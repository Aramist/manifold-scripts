from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..consts import (
    aug_units,
    default_aug_ranges,
    id_estimate_dir,
    narrow_aug_ranges,
    pca_embedding_dir,
)

exit()


def visualize_pca(
    pca_file: Path,
    save_path: Path,
    title: str,
    augmentation_range: float,
    center: bool = False,
):
    data = np.load(pca_file)
    pcs = data["pcs"]  # (n_samples * n_augs, n_components)
    n_augs = 101
    if center:
        pcs = pcs.reshape(-1, n_augs, pcs.shape[1])
        center_item = pcs[:, n_augs // 2, :][:, None, :]
        pcs -= center_item
        pcs = pcs.reshape(-1, pcs.shape[2])
    aug_strength = np.tile(
        augmentation_range, (pcs.shape[0] // n_augs, 1)
    ).flatten()  # (n_samples * n_augs,)

    # Create a DataFrame for plotting
    df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    df["Augmentation Strength"] = aug_strength
    # Plot the first two principal components
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        alpha=0.5,
        hue="Augmentation Strength",
        palette="Spectral",
    )
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.savefig(save_path)
    plt.clf()


def visualize_pca_subset(
    pca_file: Path,
    save_dir: Path,
    title: str,
    augmentation_range: float,
    center: bool = False,
    num_subset: int = 10,
    id_estimate_path: Path | None = None,
):
    data = np.load(pca_file)
    pcs = data["pcs"]  # (n_samples * n_augs, n_components)
    n_augs = 101
    if center:
        pcs = pcs.reshape(-1, n_augs, pcs.shape[1])
        center_item = pcs[:, n_augs // 2, :][:, None, :]
        pcs -= center_item
        pcs = pcs.reshape(-1, pcs.shape[2])

    if id_estimate_path and id_estimate_path.exists():
        id_data = np.load(id_estimate_path)
        id_chart = id_data["id_chart"]  # (num_perturbation_strengths, n_samples)

    for n in range(num_subset):
        rand_idx = np.random.randint(0, pcs.shape[0] // n_augs)
        one_sample = pcs.reshape(-1, n_augs, pcs.shape[1])[
            rand_idx
        ]  # (n_augs, n_components)
        aug_strength = augmentation_range  # (n_augs,)
        # Create a DataFrame for plotting
        df = pd.DataFrame(one_sample, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
        df["Augmentation Strength"] = aug_strength
        # Plot the first two principal components
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=df,
            x="PC1",
            y="PC2",
            hue="Augmentation Strength",
            palette="Spectral",
        )
        # Plot a special marker for the original (unaugmented) sample
        orig_idx = n_augs // 2
        ax.scatter(
            df.loc[orig_idx, "PC1"],
            df.loc[orig_idx, "PC2"],
            color="black",
            marker="*",
            s=100,
            label="Unaugmented",
        )
        ax.plot(
            df["PC1"],
            df["PC2"],
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
        plt.title(title + f" Index {rand_idx}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        save_path = save_dir / f"{n+1}.png"
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    pca_files = list(pca_embedding_dir.glob("*.npz"))
    plot_dir = Path("/Users/aramis/Desktop/marl_keynotes/2026-03-04_figs/pca_plots")
    (plot_dir / "uncentered").mkdir(exist_ok=True)
    (plot_dir / "centered").mkdir(exist_ok=True)
    model_options = ["PANN", "CLAP"]
    aug_options = ["gain", "pitch_shifting", "time_stretching"]
    config_options = [None, "narrow_config"]
    for model, aug, config in product(model_options, aug_options, config_options):
        cfg_part = f"_{config}" if config else ""

        pca_embedding_path = (
            pca_embedding_dir / f"pca_embeddings_{model}_{aug}{cfg_part}.npz"
        )
        id_estimate_path = (
            id_estimate_dir
            / f"intrinsic_dimensionality_{aug}_lPCA{cfg_part}_{model}.npz"
        )
        if pca_embedding_path.exists():
            aug_range = (
                narrow_aug_ranges[aug]
                if config == "narrow_config"
                else default_aug_ranges[aug]
            )
            aug_unit = aug_units[aug]
            title = f"PCA {model} - {aug.replace('_', ' ').title()} {aug_range[0]:.1f}{aug_unit} - {aug_range[-1]:+.1f}{aug_unit}"
            filename = f"pca_{model}_{aug}{cfg_part}.png"
            # visualize_pca(
            #     pca_embedding_path,
            #     plot_dir / "uncentered" / filename,
            #     title,
            #     center=False,
            #     augmentation_range=aug_range,
            # )
            # visualize_pca(
            #     pca_embedding_path,
            #     plot_dir / "centered" / filename,
            #     title + " (Centered)",
            #     center=True,
            #     augmentation_range=aug_range,
            # )
            (plot_dir / "uncentered" / f"{model}_{aug}{cfg_part}_subset").mkdir(
                exist_ok=True, parents=True
            )
            (plot_dir / "centered" / f"{model}_{aug}{cfg_part}_subset").mkdir(
                exist_ok=True, parents=True
            )
            np.random.seed(42)  # For reproducibility of random subsets
            visualize_pca_subset(
                pca_embedding_path,
                plot_dir / "uncentered" / f"{model}_{aug}{cfg_part}_subset",
                title,
                center=False,
                augmentation_range=aug_range,
                num_subset=10,
                id_estimate_path=id_estimate_path,
            )
            np.random.seed(42)  # For reproducibility of random subsets
            visualize_pca_subset(
                pca_embedding_path,
                plot_dir / "centered" / f"{model}_{aug}{cfg_part}_subset",
                title + " (Centered)",
                center=True,
                augmentation_range=aug_range,
                num_subset=10,
                id_estimate_path=id_estimate_path,
            )
