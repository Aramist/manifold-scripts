from itertools import groupby, product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from tqdm import tqdm

from ..consts import class_id_to_top_level, embedding_dir, id_estimate_dir

plot_dir = Path("/Users/aramis/Desktop/marl_keynotes/end_of_year_plots")
global_class_ids = None


def get_id_estimates(id_chart: np.ndarray) -> np.ndarray:
    # Id chart shape: (num_perturbation_radii, num_sounds)
    return id_chart[-1, :]  # (num_sounds,)


def get_sound_classes(embedding_path: Path) -> np.ndarray:
    global global_class_ids
    if embedding_path is not None:
        with h5py.File(embedding_path, "r") as ctx:
            if "class_id" in ctx:
                class_ids = ctx["class_id"][:]
                if len(class_ids) < len(ctx["embedding"]):
                    if global_class_ids is None:
                        raise ValueError(
                            "Class ID length does not match embedding length, and no global class IDs found. Cannot retrieve sound classes."
                        )
                    class_ids = global_class_ids
                    print(
                        "Warning: Class ID length does not match embedding length, using global class IDs instead"
                    )
                else:
                    if global_class_ids is None:
                        global_class_ids = class_ids
                return class_ids
            else:
                if global_class_ids is not None:
                    print(
                        "Warning: Class ID not found in embedding file, using global class IDs instead"
                    )
                    return global_class_ids
                raise ValueError(
                    "Class ID not found in embedding file, cannot retrieve sound classes."
                )
    else:
        raise ValueError("Embedding path is None, cannot retrieve sound classes.")


def get_summary_table(id_charts):

    # Rename some fields for better plotting
    model_translation = {
        "PANN": "PANN",
        "CLAP": "CLAP",
        "encodec": "EnCodec",
    }

    augmentation_translation = {
        "gain": "Loudness",
        "pitch_shifting": "Pitch",
        "time_stretching": "Speed",
    }

    # This one is too long to plot comfortably
    top_level_translations = lambda x: "Instrument" if x == "Instrument samples" else x

    summary_rows = []
    for chart in id_charts:
        id_estimates = get_id_estimates(np.load(chart["id_data_path"])["id_chart"])
        sound_classes = get_sound_classes(chart["embedding_path"])
        for class_id, id_estimate in zip(sound_classes, id_estimates):
            summary_rows.append(
                {
                    "model": model_translation[chart["model"]],
                    "augmentation": augmentation_translation[chart["augmentation"]],
                    "class_id": class_id,
                    "top_level_class": top_level_translations(
                        class_id_to_top_level.get(class_id, "Unknown")
                    ),
                    "id_estimate": id_estimate,
                }
            )
    return pd.DataFrame(summary_rows)


def make_summary_plot(summary_table: pd.DataFrame):
    """Makes a table showing the mean and deviation of ID estimates for each model / augmentation / class combo"""

    data = (
        summary_table.groupby(["model", "augmentation", "top_level_class"])[
            "id_estimate"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    pivoted = data.pivot_table(
        index=["model", "augmentation"], columns="top_level_class", values="mean"
    )

    num_models = data["model"].nunique()
    num_augs = data["augmentation"].nunique()
    num_classes = data["top_level_class"].nunique()

    # Mean and std table:
    annotations_arr = np.empty((num_models * num_augs, num_classes), dtype=object)
    for i, (model, aug) in enumerate(pivoted.index):
        for j, cls_name in enumerate(pivoted.columns):
            mean_val_loc = (
                (data["augmentation"] == aug)
                & (data["model"] == model)
                & (data["top_level_class"] == cls_name)
            )
            mean_val = data[mean_val_loc]["mean"].iloc[0]
            std_val = data[mean_val_loc]["std"].iloc[0]
            annotations_arr[i, j] = f"{mean_val:.1f}±{std_val:.1f}"

    fig, ax = plt.subplots(figsize=(7.5, 4))

    sns.heatmap(
        pivoted,
        annot=annotations_arr,
        fmt="s",
        cmap="viridis",
        cbar_kws={"label": "Mean ID Estimate"},
        ax=ax,
    )
    ax.set_ylabel("Model and Augmentation")
    ax.set_xlabel("Sound Class")
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Mean ID Estimates by Model, Augmentation, and Class")
    plt.tight_layout()
    plt.savefig(plot_dir / "summary_plot.png", dpi=300)
    plt.savefig(plot_dir / "summary_plot.pdf")


def analyze_id_estimates(summary_table: pd.DataFrame):
    # Model the relationship as id_estimate ~ 1 + C(model) + C(augmentation) + C(top_level_class)
    # equivalent to one-way ANOVA
    model = sm.OLS.from_formula(
        "id_estimate ~ C(model) + C(augmentation) + C(top_level_class)",
        data=summary_table,
    )
    fit = model.fit()
    anova_table = sm.stats.anova_lm(fit, typ=2)
    print(fit.summary())
    print(anova_table)

    param_names = fit.params.index
    coeffs = fit.params
    # stds = [np.sqrt(fit.cov_params().loc[param, param]) for param in param_names]
    # Red -> gray -> green colormap for coefficients
    colormap = LinearSegmentedColormap.from_list(
        "red_gray_green", ["green", "lightgray", "red"]
    )
    norm = plt.Normalize(vmin=-max(abs(coeffs)), vmax=max(abs(coeffs)))
    colors = [colormap(norm(coeff)) for coeff in coeffs]
    hex_colors = [rgb2hex(color) for color in colors]
    for name, coeff, color in zip(param_names, coeffs, hex_colors):
        print(f"Param: {name}, Coefficient: {coeff:.3f}, Color: {color[1:]}")
    breakpoint()


if __name__ == "__main__":
    plot_dir.mkdir(exist_ok=True, parents=True)
    is_local_options = [True]
    model_options = ["PANN", "CLAP", "encodec"]
    aug_options = ["gain", "pitch_shifting", "time_stretching"]
    config = None
    id_estimator = "lPCA"
    id_charts = []

    for model, aug, is_local in product(model_options, aug_options, is_local_options):
        local_part = "_local" if is_local else ""
        cfg_part = f"_{config}" if config is not None else ""
        id_data_path = (
            id_estimate_dir
            / f"intrinsic_dimensionality-{aug}{local_part}-{id_estimator}{cfg_part}-{model}.npz"
        )
        if not id_data_path.exists():
            print(f"Warning: No ID estimate found for {id_data_path.stem}")
            continue
        embedding_path = embedding_dir / f"BSD10k_{model}_{aug}{cfg_part}.h5"
        if not embedding_path.exists():
            print(f"Warning: No embedding found for {embedding_path.stem}")
            embedding_path = None
        id_charts.append(
            {
                "model": model,
                "augmentation": aug,
                "id_data_path": id_data_path,
                "embedding_path": embedding_path,
            }
        )

    summary_table = get_summary_table(id_charts)
    make_summary_plot(summary_table)
    analyze_id_estimates(summary_table)
