import argparse
import time
import typing as tp
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from tqdm import tqdm

args_for_augs = {
    "gain": {"gains": np.linspace(-10, 10, 101, endpoint=True)},
    "time_stretching": {
        "ratios": np.exp(
            np.linspace(
                np.log(0.5),
                np.log(2.0),
                101,
                endpoint=True,
            )
        )
    },
    "pitch_shifting": {"n_steps": np.linspace(-12, 12, 101, endpoint=True)},
}

ap = argparse.ArgumentParser()
ap.add_argument(
    "--aug",
    type=str,
    choices=list(args_for_augs.keys()),
    required=True,
    help="The type of augmentation to visualize.",
)
ap.add_argument(
    "--use_class_means",
    action="store_true",
    help="Whether to use class means or the embedding of the unaugmented audio as the class mean.",
)
ap.add_argument(
    "--center_projection",
    action="store_true",
    help="Whether to center the data by the embedding of the unaugmented audio before projecting onto the top two generalized eigenvectors.",
)
ap.add_argument(
    "--class-filter",
    help="High-level classes to include in visualization. To use all, do not include this argument or set to empty string. To use multiple, separate by commas, e.g., 'm,is,sp'.",
    default="",
)
ap.add_argument(
    "--model",
    type=str,
    choices=["PANN", "CLAP"],
    default="PANN",
    help="Which model's embeddings to visualize.",
)
args = ap.parse_args()
# When true: class means are used to compute scatter matrices
# When false: the embedding of the unaugmented audio is used as the class mean
USE_CLASS_MEANS = args.use_class_means
AUG = args.aug
CENTER_PROJECTION = args.center_projection
CLASS_FILTER = args.class_filter
MODEL = args.model
if CLASS_FILTER:
    # Sort for consistency
    CLASS_FILTER = ",".join(sorted(CLASS_FILTER.split(",")))


class_id_mapping = {
    "m-sp": 0,  # Music
    "m-si": 1,
    "m-m": 2,
    "is-p": 3,  # Instrument samples
    "is-s": 4,
    "is-w": 5,
    "is-k": 6,
    "is-e": 7,
    "sp-s": 8,  # Speech
    "sp-c": 9,
    "sp-p": 10,
    "fx-o": 11,  # Sound effects
    "fx-v": 12,
    "fx-m": 13,
    "fx-h": 14,
    "fx-a": 15,
    "fx-n": 16,
    "fx-ex": 17,
    "fx-el": 18,
    "ss-n": 19,  # Soundscapes
    "ss-i": 20,
    "ss-u": 21,
    "ss-s": 22,
}

if Path(f"/home/at4219/scratch/BSD10k_{MODEL}_{AUG}.h5").exists():
    embedding_file = Path(f"/home/at4219/scratch/BSD10k_{MODEL}_{AUG}.h5")
else:
    embedding_file = Path(
        f"/Users/aramis/Heap/computed_embeddings/BSD10k_{MODEL}_{AUG}.h5"
    )
# See which system we're on
if Path("/home/at4219/scratch/").exists():
    data_dir = Path(
        f"/home/at4219/scratch/covs/{MODEL}_{AUG}_{'class_mean' if USE_CLASS_MEANS else 'orig_emb'}{'' if not CLASS_FILTER else '_' + CLASS_FILTER}"
    )
else:
    data_dir = Path(
        f"/Users/aramis/covs/{MODEL}_{AUG}_{'class_mean' if USE_CLASS_MEANS else 'orig_emb'}{'' if not CLASS_FILTER else '_' + CLASS_FILTER}"
    )

plot_dir = (
    Path("/Users/aramis/Desktop/marl_keynotes/2026-02-11_figs")
    / f"{MODEL}_{AUG}{'' if not CLASS_FILTER else '_' + CLASS_FILTER}"
)

data_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)


def timed(fn: tp.Callable) -> tp.Callable:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print(f"{fn.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


if CLASS_FILTER:
    class_filters = CLASS_FILTER.split(",")
    class_ids_to_include = []
    for cf in class_filters:
        for class_name, class_id in class_id_mapping.items():
            if class_name.startswith(cf):
                class_ids_to_include.append(class_id)
    print("Filtering to classes: ", class_ids_to_include)

with h5py.File(embedding_file, "r") as hf:
    emb: np.ndarray = hf["embedding"][
        :
    ].squeeze()  # shape (num_files, num_augs, embedding_dim)
    if CLASS_FILTER:
        all_class_ids = hf["class_id"][:]
        mask = np.isin(all_class_ids, class_ids_to_include)
        emb = emb[mask]
print("Loaded embeddings shape: ", emb.shape)
num_classes, num_augs, embedding_dim = emb.shape
# First step in FDA: globally center data
if (data_dir / "global_mean.npy").exists():
    global_mean = np.load(data_dir / "global_mean.npy")
else:
    print("Casting to float64...")
    global_mean = emb.astype(np.float64)
    print("Reducing across augmentations...")
    global_mean = global_mean.mean(axis=1)
    print("Reducing across classes...")
    global_mean = global_mean.mean(axis=0)
    np.save(data_dir / "global_mean.npy", global_mean.squeeze())
print("centering data...")
for i, emb_slice in enumerate(tqdm(emb)):
    emb[i] = emb_slice - global_mean.astype(np.float32)
emb_centered = emb
# emb_centered = emb - global_mean.astype(np.float32)
# Have the choice of centering each class relative to its mean or the embedding of the unaugmented audio
orig_audio_emb = emb_centered[:, num_augs // 2, :].astype(
    np.float64
)  # (50, embedding_dim)
class_means = (
    emb_centered.mean(axis=1).astype(np.float64) if USE_CLASS_MEANS else orig_audio_emb
)

# Compute the sum of scatter matrices within each class (audio file)
# Target shape: (num_classes, num_features, num_features) --- sum ---> (num_features, num_features)
if (data_dir / "within_class_scatter.npy").exists():
    within_class_scatter = np.load(data_dir / "within_class_scatter.npy")
else:
    within_class_scatter = np.zeros((embedding_dim, embedding_dim), dtype=np.float64)
    for emb_slice, class_mean in tqdm(
        zip(emb_centered, class_means),
        desc="Computing within-class scatter",
        total=num_classes,
    ):
        emb_slice_centered = emb_slice - class_mean[None, :]
        within_class_scatter += emb_slice_centered.T @ emb_slice_centered
    np.save(data_dir / "within_class_scatter.npy", within_class_scatter)

# Compute the scatter matrix across classes
# shape: (num_classes, features)
if (data_dir / "across_class_scatter.npy").exists():
    across_class_scatter = np.load(data_dir / "across_class_scatter.npy")
else:
    # shape: (features, features)
    across_class_scatter = num_classes * class_means.T @ class_means
    np.save(data_dir / "across_class_scatter.npy", across_class_scatter)


reg = np.eye(embedding_dim) * 1e-4
eigh = timed(scipy.linalg.eigh)

if (data_dir / "eig.npy").exists() and (data_dir / "eigv.npy").exists():
    generalized_eig = np.load(data_dir / "eig.npy")
    generalized_eigv = np.load(data_dir / "eigv.npy")
else:
    try:
        generalized_eig, generalized_eigv = eigh(
            a=within_class_scatter,
            b=across_class_scatter,
            # driver="gv",
            overwrite_a=True,
            overwrite_b=True,
            # subset_by_index=[embedding_dim - 3, embedding_dim - 1],
        )
    except np.linalg.LinAlgError as e:
        print(
            "Failed to solve generalized eigenvalue problem without regularization, retrying with regularization..."
        )
        generalized_eig, generalized_eigv = eigh(
            a=within_class_scatter,
            b=across_class_scatter + reg,
            # driver="gv",
            overwrite_a=True,
            overwrite_b=True,
            # subset_by_index=[embedding_dim - 3, embedding_dim - 1],
        )

    # generalized_eig, generalized_eigv = scipy.linalg.eigh(a=within_class_cov)

    # Reverse them so largest come first
    generalized_eigv = (generalized_eigv.T)[::-1]  # shape (num_eigv, dim)
    generalized_eig = generalized_eig[::-1]

    np.save(data_dir / "eigv.npy", generalized_eigv)
    np.save(data_dir / "eig.npy", generalized_eig)

print(f"spectral gap: {generalized_eig[0] / generalized_eig[1]:.2f}")
print("First ten generalized eigenvalues: ")
print(generalized_eig[:10])

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 6), width_ratios=[4, 1])
fig.suptitle(
    f"Generalized Eigenspectrum: {AUG}, rel. {'class means' if USE_CLASS_MEANS else 'orig emb'}"
)
ax_l.plot(generalized_eig)
ax_l.set_xlabel("Eigenvalue index")
ax_l.set_ylabel(r"$\lambda_i$")
ax_l.set_yscale("log")

ax_r.plot(generalized_eig[:32])
ax_r.set_xlabel("Eigenvalue index (zoomed)")
ax_r.set_yscale("log")
fig.tight_layout()
plt.savefig(
    plot_dir / f"{AUG}_eigenspectrum{'' if USE_CLASS_MEANS else '_orig_emb'}.png",
    dpi=300,
)
plt.clf()


cumulative_eigenvalues = np.cumsum(generalized_eig)
cutoff_idx = np.flatnonzero(
    (cumulative_eigenvalues / cumulative_eigenvalues[-1]) > 0.95
)[0]
print(cutoff_idx)
generalized_eig = generalized_eig[:cutoff_idx]
generalized_eigv = generalized_eigv[:cutoff_idx]
print(generalized_eig.shape, generalized_eigv.shape)

#
# top_two_directions = generalized_eigv[1:3]  # shape: (2, features)
top_two_directions = generalized_eigv[:2]  # shape: (2, features)
if CENTER_PROJECTION:
    emb_centered = emb_centered - orig_audio_emb[:, None, :]
projected_embeddings = np.einsum(
    "df,cbf->cbd", top_two_directions, emb_centered
)  # (num_files, num_augs, features)


emb_centered = emb_centered.reshape(num_classes * num_augs, embedding_dim)
# projected_embeddings = projected_embeddings[0]
# projected_embeddings = projected_embeddings.reshape(num_classes * num_augs, 2)

for n in range(20):
    rand_idx = np.random.choice(num_classes, 1, replace=False).item()
    colors = np.linspace(0, 1, num_augs)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        projected_embeddings[rand_idx, :, 0],
        projected_embeddings[rand_idx, :, 1],
        # alpha=0.6,
        s=20,
        c=colors,
        cmap="RdBu",
    )
    ax.set_title(f"Generalized eigenvalue projection: {AUG}, {MODEL}")
    ax.set_xlabel("Generalized Eigenvector 2")
    ax.set_ylabel("Generalized Eigenvector 3")
    ax.grid(True)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu"), ax=ax, orientation="vertical"
    )
    aug_type = "_".join(embedding_file.stem.split("_")[2:])
    cbar.set_label(f"{aug_type} strength")
    cbar.set_ticks([0, 0.5, 1])
    if aug_type == "gain":
        low_gain = f"{args_for_augs['gain']['gains'][0]:.0f} dB"
        mid_gain = "0 dB"
        hi_gain = f"{args_for_augs['gain']['gains'][-1]:.0f} dB"
        cbar.set_ticklabels([low_gain, mid_gain, hi_gain])
    elif aug_type == "time_stretching":
        low_ratio = f"{args_for_augs['time_stretching']['ratios'][0]:.2f}x (slow)"
        mid_ratio = "1.00x"
        hi_ratio = f"{args_for_augs['time_stretching']['ratios'][-1]:.2f}x (fast)"
        cbar.set_ticklabels([low_ratio, mid_ratio, hi_ratio])
    elif aug_type == "pitch_shifting":
        low_steps = f"{args_for_augs['pitch_shifting']['n_steps'][0]:.0f} st"
        mid_steps = "0 st"
        hi_steps = f"{args_for_augs['pitch_shifting']['n_steps'][-1]:.0f} st"
        cbar.set_ticklabels([low_steps, mid_steps, hi_steps])

    plt.tight_layout()
    (plot_dir / "only_one_sample").mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "only_one_sample" / f"{AUG}_embeddings_gev_{n}.png")
exit()


colors = np.tile(np.linspace(0, 1, num_augs), num_classes)

fig, ax = plt.subplots(figsize=(8, 8))
# ax.scatter(
#     projected_embeddings[:, 0],
#     projected_embeddings[:, 1],
#     alpha=0.6,
#     s=4,
#     c=colors,
#     cmap="RdBu",
# )
ax.scatter(
    projected_embeddings[:, 0],
    projected_embeddings[:, 1],
    alpha=0.6,
    s=4,
    c=colors,
    cmap="RdBu",
)
# Occasionally the axes limits get cooked by some extreme outliers
# xmin, xmax = np.quantile(projected_embeddings[:, 0], [1e-3, 1 - 1e-3])
# xmed = 0.5 * (xmin + xmax)
# xmin, xmax = xmed + 1.2 * (xmin - xmed), xmed + 1.2 * (xmax - xmed)
# ymin, ymax = np.quantile(projected_embeddings[:, 1], [1e-3, 1 - 1e-3])
# ymed = 0.5 * (ymin + ymax)
# ymin, ymax = ymed + 1.2 * (ymin - ymed), ymed + 1.2 * (ymax - ymed)
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
ax.set_title(f"Generalized eigenvalue projection: {AUG}, {MODEL}")
ax.set_xlabel("Generalized Eigenvector 2")
ax.set_ylabel("Generalized Eigenvector 3")
ax.grid(True)

cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu"), ax=ax, orientation="vertical")
aug_type = "_".join(embedding_file.stem.split("_")[2:])
cbar.set_label(f"{aug_type} strength")
cbar.set_ticks([0, 0.5, 1])
if aug_type == "gain":
    low_gain = f"{args_for_augs['gain']['gains'][0]:.0f} dB"
    mid_gain = "0 dB"
    hi_gain = f"{args_for_augs['gain']['gains'][-1]:.0f} dB"
    cbar.set_ticklabels([low_gain, mid_gain, hi_gain])
elif aug_type == "time_stretching":
    low_ratio = f"{args_for_augs['time_stretching']['ratios'][0]:.2f}x (slow)"
    mid_ratio = "1.00x"
    hi_ratio = f"{args_for_augs['time_stretching']['ratios'][-1]:.2f}x (fast)"
    cbar.set_ticklabels([low_ratio, mid_ratio, hi_ratio])
elif aug_type == "pitch_shifting":
    low_steps = f"{args_for_augs['pitch_shifting']['n_steps'][0]:.0f} st"
    mid_steps = "0 st"
    hi_steps = f"{args_for_augs['pitch_shifting']['n_steps'][-1]:.0f} st"
    cbar.set_ticklabels([low_steps, mid_steps, hi_steps])

plt.tight_layout()
plt.savefig(
    plot_dir
    / f"{AUG}_embeddings_gev{'' if USE_CLASS_MEANS else '_orig_emb'}{'' if CENTER_PROJECTION else '_uncentered'}.png",
    dpi=300,
)
