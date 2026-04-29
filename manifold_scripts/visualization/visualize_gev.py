import typing as tp
from itertools import groupby, product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..consts import (
    aug_units,
    class_id_mapping,
    default_aug_ranges,
    embedding_dir,
    gev_embedding_dir,
    id_estimate_dir,
    narrow_aug_ranges,
    noop_index,
)

sns.set_theme("paper")


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
    augmentation_name: str,
    center: bool = True,
):
    with h5py.File(embedding_file, "r") as hf:
        # Loads the dataset into memory
        emb: np.ndarray = np.array(
            hf["embedding"][:]
        ).squeeze()  # shape (num_files, num_augs, embedding_dim)
    num_classes, num_augs, embedding_dim = emb.shape
    orig_emb_idx = noop_index[augmentation_name]
    orig_audio_emb = emb[:, orig_emb_idx, :].astype(np.float64)
    # file format  f"gev_embeddings_{model}_{aug}{cfg_part}{filter_part}.npz"
    aug = gev_file.stem.split("_")[3]
    if aug != "gain":
        aug = "_".join(gev_file.stem.split("_")[3:5])

    data = np.load(gev_file)
    generalized_eig = data["gev_eig"]  # (num_components,)
    generalized_eigv = data["gev_eigv"]  # (num_components, embedding_dim,)

    # top_two_directions = generalized_eigv[1:3]  # shape: (2, features)
    top_two_directions = generalized_eigv[:3]  # shape: (2, features)
    if center:
        emb -= orig_audio_emb[:, None, :]

    projected_embeddings = np.einsum(
        "df,cbf->cbd", top_two_directions, emb
    )  # (num_files, num_augs, features)

    projected_embeddings = projected_embeddings.reshape(
        num_classes * num_augs, top_two_directions.shape[0]
    )
    colors = np.tile(augmentation_range, (num_classes, 1)).flatten()

    # quick dataframe for seaborn
    df = pd.DataFrame(
        projected_embeddings,
        columns=[f"LD {i+1}" for i in range(projected_embeddings.shape[1])],
    )
    df["Augmentation Strength"] = colors

    fig, ax = plt.subplots(figsize=(8, 6))
    # scatter = ax.scatter(
    #     projected_embeddings[:, 0],
    #     projected_embeddings[:, 1],
    #     alpha=0.6,
    #     s=4,
    #     c=colors,
    #     cmap="RdBu",
    # )
    sns.scatterplot(
        data=df,
        x="LD 2",
        y="LD 3",
        hue="Augmentation Strength",
        palette="Spectral",
        alpha=0.5,
        ax=ax,
        size=1,
    )
    # temporarily get rid of the legend
    # ax.legend([], [], frameon=False)
    ax.set_title(title)
    # get rid of ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
    # cbar.set_label(f"{aug} strength")
    # aug_unit = aug_units[augmentation_name]
    # cbar.set_ticks(
    #     [
    #         augmentation_range[0],
    #         augmentation_range[len(augmentation_range) // 2],
    #         augmentation_range[-1],
    #     ]
    # )
    # cbar.set_ticklabels(
    #     [
    #         f"{augmentation_range[0]:+.1f} {aug_unit}",
    #         f"{augmentation_range[len(augmentation_range) // 2]:.1f} {aug_unit}",
    #         f"{augmentation_range[-1]:+.1f} {aug_unit}",
    #     ]
    # )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.clf()


def visualize_gev_subset(
    embedding_file: Path,
    gev_file: Path,
    save_dir: Path,
    title: str,
    augmentation_range: float,
    augmentation_name: str,
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
    orig_emb_idx = noop_index[augmentation_name]
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
            one_sample, columns=[f"LD {i+1}" for i in range(one_sample.shape[1])]
        )
        df["Augmentation Strength"] = aug_strength

        # Plot the first two principal components
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=df,
            x="LD 1",
            y="LD 2",
            hue="Augmentation Strength",
            palette="Spectral_r",
        )
        # Plot a special marker for the original (unaugmented) sample
        ax.scatter(
            df.loc[orig_emb_idx, "LD 1"],
            df.loc[orig_emb_idx, "LD 2"],
            color="black",
            marker="*",
            s=100,
            label="Unaugmented",
        )
        # Have a line connecting all the points in order
        ax.plot(
            df["LD 1"],
            df["LD 2"],
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
        plt.xlabel("LD 1")
        plt.ylabel("LD 2")
        plt.grid()
        save_path = save_dir / f"{n+1}.png"
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    plot_dir = Path("/Users/aramis/Desktop/marl_keynotes/2026-03-04_figs/gev_plots")
    (plot_dir / "uncentered").mkdir(exist_ok=True, parents=True)
    (plot_dir / "centered").mkdir(exist_ok=True, parents=True)
    model_options = ["PANN", "CLAP", "encodec"]
    aug_options = ["gain", "pitch_shifting", "time_stretching", "low_pass_filter"]
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
                    augmentation_name=aug,
                )
            if not (plot_dir / "centered" / filename).exists():
                visualize_gev(
                    embedding_path,
                    gev_embedding_path,
                    plot_dir / "centered" / filename,
                    title + " (Centered)",
                    center=True,
                    augmentation_range=aug_range,
                    augmentation_name=aug,
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
                augmentation_name=aug,
                num_subset=10,
                id_estimate_path=id_estimate_path,
            )
