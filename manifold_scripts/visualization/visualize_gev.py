import typing as tp
from itertools import groupby, product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..consts import class_id_mapping, embedding_dir, gev_embedding_dir, id_estimate_dir


def convert_ids_to_groups(class_ids: np.ndarray) -> list[str]:
    groups: dict[str, list[int]] = {}
    for toplevel_class, lowlevel_classes in groupby(
        class_id_mapping.keys(), lambda k: k.split("-")[0]
    ):
        groups[toplevel_class] = [class_id_mapping[v] for v in lowlevel_classes]

    inverted_groups = {
        class_id: top_level_class
        for top_level_class, class_ids in groups.items()
        for class_id in class_ids
    }
    toplevel_class_labels = [inverted_groups[int(cid)] for cid in class_ids]
    return toplevel_class_labels


def visualize_eigenspectrum(
    gev_file: Path,
    save_path: Path,
    title: str,
    augmentation_range: float,
):
    return
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


def visualize_gev(
    embedding_file: Path,
    gev_file: Path,
    save_path: Path,
    title: str,
    augmentation_range: np.ndarray,
    center: bool = True,
):
    with h5py.File(embedding_file, "r") as hf:
        # Loads the dataset into memory
        emb: np.ndarray = np.array(
            hf["embedding"][:]
        ).squeeze()  # shape (num_files, num_augs, embedding_dim)
    num_classes, num_augs, embedding_dim = emb.shape
    orig_emb_idx = num_augs // 2
    orig_audio_emb = emb[:, orig_emb_idx, :].astype(np.float64)
    # file format  f"gev_embeddings_{model}_{aug}{cfg_part}{filter_part}.npz"
    aug = gev_file.stem.split("_")[3]
    if aug != "gain":
        aug = "_".join(gev_file.stem.split("_")[3:5])

    data = np.load(gev_file)
    generalized_eig = data["gev_eig"]  # (num_components,)
    generalized_eigv = data["gev_eigv"]  # (num_components, embedding_dim,)

    # top_two_directions = generalized_eigv[1:3]  # shape: (2, features)
    top_two_directions = generalized_eigv[:2]  # shape: (2, features)
    if center:
        emb -= orig_audio_emb[:, None, :]

    projected_embeddings = np.einsum(
        "df,cbf->cbd", top_two_directions, emb
    )  # (num_files, num_augs, features)

    projected_embeddings = projected_embeddings.reshape(num_classes * num_augs, 2)
    colors = np.tile(augmentation_range, (num_classes, 1)).flatten()

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        projected_embeddings[:, 0],
        projected_embeddings[:, 1],
        alpha=0.6,
        s=4,
        c=colors,
        cmap="RdBu",
    )
    ax.set_title(title)
    ax.set_xlabel("Generalized Eigenvector 1")
    ax.set_ylabel("Generalized Eigenvector 2")
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label(f"{aug} strength")
    aug_unit = {
        "gain": "dB",
        "pitch_shifting": "st",
        "time_stretching": "x",
    }[aug]
    cbar.set_ticks(
        [
            augmentation_range[0],
            augmentation_range[len(augmentation_range) // 2],
            augmentation_range[-1],
        ]
    )
    cbar.set_ticklabels(
        [
            f"{augmentation_range[0]:+.1f} {aug_unit}",
            f"{augmentation_range[len(augmentation_range) // 2]:.1f} {aug_unit}",
            f"{augmentation_range[-1]:+.1f} {aug_unit}",
        ]
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.clf()


def visualize_gev_subset(
    embedding_file: Path,
    gev_file: Path,
    save_dir: Path,
    title: str,
    augmentation_range: float,
    num_subset: int = 10,
    id_estimate_path: Path | None = None,
):
    class_ids = None
    with h5py.File(embedding_file, "r") as ctx:
        emb = ctx["embedding"]
        rand_indices = np.random.randint(0, emb.shape[0], size=num_subset)
        # h5py is picky about how you can grab multiple indices at once
        emb = np.stack(
            [emb[rand_idx, ...].squeeze() for rand_idx in rand_indices], axis=0
        )
        if "class_id" in ctx:
            class_ids = np.array(ctx["class_id"][:])
            if len(class_ids) < len(ctx["embedding"]):
                class_ids = None

    if class_ids is not None:
        class_ids = convert_ids_to_groups(class_ids)
        translation = {
            "m": "Music",
            "is": "Instrumental",
            "sp": "Speech",
            "fx": "Sound Effects",
            "ss": "Soundscapes",
        }
        class_ids = [translation[cid] for cid in class_ids]

    _, num_augs, _ = emb.shape
    orig_emb_idx = num_augs // 2
    orig_audio_emb = emb[:, orig_emb_idx, :].astype(np.float64)
    # file format  f"gev_embeddings_{model}_{aug}{cfg_part}{filter_part}.npz"
    aug = gev_file.stem.split("_")[3]
    if aug != "gain":
        aug = "_".join(gev_file.stem.split("_")[3:5])

    data = np.load(gev_file)
    # generalized_eig = data["gev_eig"]  # (num_components,)
    generalized_eigv = data["gev_eigv"]  # (num_components, embedding_dim,)

    top_two_directions = generalized_eigv[:2]  # shape: (2, features)
    emb -= orig_audio_emb[:, None, :]

    projected_embeddings = np.einsum(
        "df,cbf->cbd", top_two_directions, emb
    )  # (num_files, num_augs, features)

    if id_estimate_path and id_estimate_path.exists():
        id_data = np.load(id_estimate_path)
        id_chart = id_data["id_chart"]  # (num_perturbation_strengths, n_samples)

    for n, rand_idx in enumerate(rand_indices):
        one_sample = projected_embeddings[n]  # (n_augs, features)
        aug_strength = augmentation_range  # (n_augs,)
        # Create a DataFrame for plotting
        df = pd.DataFrame(
            one_sample, columns=[f"GEV {i+1}" for i in range(one_sample.shape[1])]
        )
        df["Augmentation Strength"] = aug_strength

        # Plot the first two principal components
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=df,
            x="GEV 1",
            y="GEV 2",
            hue="Augmentation Strength",
            palette="Spectral_r",
        )
        # Plot a special marker for the original (unaugmented) sample
        orig_idx = num_augs // 2
        ax.scatter(
            df.loc[orig_idx, "GEV 1"],
            df.loc[orig_idx, "GEV 2"],
            color="black",
            marker="*",
            s=100,
            label="Unaugmented",
        )
        # Have a line connecting all the points in order
        ax.plot(
            df["GEV 1"],
            df["GEV 2"],
            color="gray",
            alpha=0.5,
        )
        if id_estimate_path and id_estimate_path.exists():
            sample_id_chart = id_chart[:, rand_idx]  # (num_perturbation_strengths,))
            inset_ax = plt.gca().inset_axes([0.65, 0.65, 0.3, 0.3])
            sns.lineplot(
                x=np.arange(len(sample_id_chart)) + 5,
                y=sample_id_chart,
                ax=inset_ax,
            )
            inset_ax.set_title("ID Estimate")
            inset_ax.set_xlabel("Dist from center")
            inset_ax.set_ylabel("Estimated ID")
        plt.title(
            title + f", Index {rand_idx}{', ' + class_ids[n] if class_ids else ''}"
        )
        plt.xlabel("GEV 1")
        plt.ylabel("GEV 2")
        plt.grid()
        save_path = save_dir / f"{n+1}.png"
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    default_aug_ranges = {
        "gain": np.linspace(-10.0, 10.0, 101),
        "pitch_shifting": np.linspace(-12.0, 12.0, 101),
        "time_stretching": np.exp(np.linspace(np.log(0.5), np.log(2.0), 101)),
    }
    narrow_aug_ranges = {
        "gain": np.linspace(-4, 4, 101),
        "pitch_shifting": np.linspace(-4.0, 4.0, 101),
        "time_stretching": np.exp(np.linspace(np.log(1 / 1.3), np.log(1.3), 101)),
    }
    aug_units = {
        "gain": "dB",
        "pitch_shifting": "st",
        "time_stretching": "x",
    }
    plot_dir = Path("/Users/aramis/Desktop/marl_keynotes/2026-03-04_figs/gev_plots")
    (plot_dir / "uncentered").mkdir(exist_ok=True, parents=True)
    (plot_dir / "centered").mkdir(exist_ok=True, parents=True)
    model_options = ["PANN", "CLAP"]
    aug_options = ["gain", "pitch_shifting", "time_stretching"]
    config_options = [None, "narrow_config"]
    for model, aug, config in product(model_options, aug_options, config_options):
        cfg_part = f"_{config}" if config else ""

        embedding_path = embedding_dir / f"BSD10k_{model}_{aug}{cfg_part}.h5"
        gev_embedding_path = (
            gev_embedding_dir / f"gev_embeddings_{model}_{aug}{cfg_part}.npz"
        )
        id_estimate_path = (
            id_estimate_dir
            / f"intrinsic_dimensionality_{aug}_lPCA{cfg_part}_{model}.npz"
        )
        if gev_embedding_path.exists():
            aug_range = (
                narrow_aug_ranges[aug]
                if config == "narrow_config"
                else default_aug_ranges[aug]
            )
            aug_unit = aug_units[aug]
            title = f"GEV {model} - {aug.replace('_', ' ').title()} {aug_range[0]:.1f}{aug_unit} - {aug_range[-1]:+.1f}{aug_unit}"
            filename = f"gev_{model}_{aug}{cfg_part}.png"
            if not (plot_dir / "uncentered" / filename).exists():
                visualize_gev(
                    embedding_path,
                    gev_embedding_path,
                    plot_dir / "uncentered" / filename,
                    title,
                    center=False,
                    augmentation_range=aug_range,
                )
            if not (plot_dir / "centered" / filename).exists():
                visualize_gev(
                    embedding_path,
                    gev_embedding_path,
                    plot_dir / "centered" / filename,
                    title + " (Centered)",
                    center=True,
                    augmentation_range=aug_range,
                )
            (plot_dir / "uncentered" / f"{model}_{aug}{cfg_part}_subset").mkdir(
                exist_ok=True, parents=True
            )
            np.random.seed(42)  # For reproducibility of random subsets
            visualize_gev_subset(
                embedding_path,
                gev_embedding_path,
                plot_dir / "uncentered" / f"{model}_{aug}{cfg_part}_subset",
                title,
                augmentation_range=aug_range,
                num_subset=10,
                id_estimate_path=id_estimate_path,
            )
