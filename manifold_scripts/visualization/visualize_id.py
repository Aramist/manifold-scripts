from itertools import groupby, product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..consts import class_id_mapping, embedding_dir, id_estimate_dir

plot_dir = Path("/Users/aramis/Desktop/marl_keynotes/2026-03-04_figs/id_charts")


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


def parse_filename(data_path: Path) -> dict[str, str | bool]:
    if "-" in data_path.stem:
        # New format with dashes
        # intrinsic_dimensionality-time_stretching_local-lPCA-narrow_config-CLAP.npz
        # intrinsic_dimensionality-{aug}_{local optional}-lPCA-{config optional}-{model}.npz
        split = data_path.stem.split("-")
        augmentation = split[1]
        locality = "local" in augmentation
        if locality:
            augmentation = augmentation.replace("_local", "")
        estimator = split[2]
        config = split[3] if len(split) > 4 else "default"
        model = split[-1]
        return {
            "augmentation": augmentation,
            "is_local": locality,
            "estimator": estimator,
            "config": config,
            "model": model,
        }
    # Structure of filename:
    # intrinsic_dimensionality_{augmentation}_{local (optional)}_{estimator}_{config (optional)}_{model}.npz
    name_split = data_path.stem.split("_")[
        2:
    ]  # Remove "intrinsic_dimensionality" prefix

    # Get augmentation
    augmentation = name_split[0]
    if augmentation == "gain":
        name_split = name_split[1:]  # Handle "gain" which doesn't have a second part
    if augmentation != "gain":
        augmentation = "_".join(
            name_split[:2]
        )  # Handle "pitch_shift" and "time_stretch"
        name_split = name_split[2:]  # Remove augmentation part

    # Get local flag
    if name_split[0] == "local":
        name_split = name_split[1:]  # Remove "local" part
        is_local = True
    else:
        is_local = False

    # Get estimator
    estimator = name_split[0]
    if estimator == "lPCA":
        estimator = "PCA"
    name_split = name_split[1:]  # Remove estimator part

    # Get config
    if len(name_split) > 1:
        config = "_".join(name_split[:-1])  # All parts except the last one
    else:
        config = "default"  # Default config if none specified
    model = data_path.stem.split("_")[-1]
    return {
        "augmentation": augmentation,
        "is_local": is_local,
        "estimator": estimator,
        "config": config,
        "model": model,
    }


def plot_x_label(params: dict[str, str | bool]) -> str:
    augmentation = params["augmentation"]
    if augmentation == "gain":
        return "Perturbation radius ($\\pm$ dB)"
    elif "pitch" in augmentation:
        return "Perturbation radius ($\\pm$ semitones)"
    elif "time" in augmentation:
        return "Perturbation radius (stretch factor)"
    else:
        return "Perturbation radius"


def make_plot(data_path: Path, embedding_path: Path | None = None):
    data = np.load(data_path, allow_pickle=True)
    params = parse_filename(data_path)
    augmentation = params["augmentation"]
    is_local = params["is_local"]
    estimator = params["estimator"]
    config = params["config"]
    model = params["model"]
    class_ids = None

    if embedding_path is not None:
        with h5py.File(embedding_path, "r") as ctx:
            if "class_id" in ctx:
                class_ids = np.array(ctx["class_id"][:])
                if len(class_ids) < len(ctx["embedding"]):
                    class_ids = None

    gains = data["perturbation_strength"]  # shape (num_perturbations,)
    id_chart = data["id_chart"]  # shape (num_perturbations, num_samples)
    num_perturbations, num_instances = id_chart.shape
    dists = data["dists_from_center"]
    minmax_samples = dists[0] * 2 + 1, dists[-1] * 2 + 1

    perturbation_strengths = np.repeat(gains, num_instances)
    if class_ids is None:
        class_ids = [None] * (num_instances * num_perturbations)
    else:
        class_ids = np.tile(class_ids, num_perturbations)

    df = pd.DataFrame(
        {
            "perturbation_strength": perturbation_strengths,
            "estimated_id": id_chart.flatten(),
            "class_id": class_ids,
        }
    )
    ax = sns.lineplot(
        data=df,
        x="perturbation_strength",
        y="estimated_id",
    )
    unit = "dB" if augmentation == "gain" else "st" if "pitch" in augmentation else ""
    ax.set_title(
        f"Estimated ID ({'local' if is_local else 'global'} {estimator}) across {augmentation} $\\pm${perturbation_strengths.max():.1f} {unit} for {model}"
    )
    ax.set_xlabel(plot_x_label(params))
    ax.set_ylabel("Estimated ID")
    ax.set_ylim(
        0.8, id_chart.mean(axis=1).max() * 1.1
    )  # Set y-axis limit for better visualization
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(
        plot_dir
        / f"intrinsic_dimensionality-{augmentation}-{estimator}-{config}-{model}-{'local' if is_local else 'global'}.png",
        dpi=300,
    )
    plt.close()

    if class_ids is None or class_ids[0] is None:
        return

    class_ids = convert_ids_to_groups(class_ids)
    translation = {
        "m": "Music",
        "is": "Instrumental",
        "sp": "Speech",
        "fx": "Sound Effects",
        "ss": "Soundscapes",
    }
    class_ids = [translation[cid] for cid in class_ids]

    df = pd.DataFrame(
        {
            "perturbation_strength": perturbation_strengths,
            "estimated_id": id_chart.flatten(),
            "Sound Class": class_ids,
        }
    )
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="perturbation_strength",
        y="estimated_id",
        hue="Sound Class",
        palette="tab10",
        ax=ax,
    )
    unit = "dB" if augmentation == "gain" else "st" if "pitch" in augmentation else ""
    ax.set_title(
        f"Estimated ID ({'local' if is_local else 'global'} {estimator}) across {augmentation} $\\pm${perturbation_strengths.max():.1f} {unit} for {model}"
    )
    ax.set_xlabel(plot_x_label(params))
    ax.set_ylabel("Estimated ID")
    ax.set_ylim(
        0.8, id_chart.mean(axis=1).max() * 1.1
    )  # Set y-axis limit for better visualization
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(
        plot_dir
        / "by_class"
        / f"intrinsic_dimensionality-{augmentation}-{estimator}-{config}-{model}-{'local' if is_local else 'global'}.png",
        dpi=300,
    )
    plt.close()


def local_vs_global_sidebyside(local_path: Path, global_path: Path):
    params = parse_filename(local_path)

    local_data = np.load(local_path, allow_pickle=True)
    global_data = np.load(global_path, allow_pickle=True)

    dists = local_data["dists_from_center"]  # Should be same for both
    minmax_samples = dists[0] * 2 + 1, dists[-1] * 2 + 1

    local_df = pd.DataFrame(
        {
            "perturbation_strength": np.repeat(
                local_data["perturbation_strength"], local_data["id_chart"].shape[1]
            ),
            "estimated_id": local_data["id_chart"].flatten(),
        }
    )
    global_df = pd.DataFrame(
        {
            "perturbation_strength": np.repeat(
                global_data["perturbation_strength"], global_data["id_chart"].shape[1]
            ),
            "estimated_id": global_data["id_chart"].flatten(),
        }
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.lineplot(
        data=local_df,
        x="perturbation_strength",
        y="estimated_id",
        ax=ax_l,
    )
    sns.lineplot(
        data=global_df,
        x="perturbation_strength",
        y="estimated_id",
        ax=ax_r,
    )

    fig.suptitle(
        f"Local vs Global {params['estimator']} ID across {params['augmentation']} $\\pm${local_df['perturbation_strength'].max():.1f} for {params['model']}"
    )
    ax_l.set_title(f"Local {params['estimator']}")
    ax_r.set_title(f"Global {params['estimator']}")

    ax_l.set_xlabel(plot_x_label(params))
    ax_r.set_xlabel(plot_x_label(params))
    ax_l.set_ylabel("Estimated ID")
    ax_l.set_ylim(
        0.8,
        max(
            local_data["id_chart"].mean(axis=1).max(),
            global_data["id_chart"].mean(axis=1).max(),
        )
        * 1.1,
    )  # Set y-axis limit for better visualization
    fig.tight_layout()
    plt.savefig(
        plot_dir
        / "local_vs_global"
        / f"intrinsic_dimensionality-{params['augmentation']}-{params['estimator']}-{params['config']}-{params['model']}-localvsglobal.png",
    )


def pann_vs_clap_sidebyside(pann_path: Path, clap_path: Path):
    # Similar structure to local_vs_global_sidebyside but comparing PANN vs CLAP for the same config
    params = parse_filename(pann_path)
    pann_data = np.load(pann_path, allow_pickle=True)
    clap_data = np.load(clap_path, allow_pickle=True)

    dists = pann_data["dists_from_center"]  # Should be same for both
    minmax_samples = dists[0] * 2 + 1, dists[-1] * 2 + 1

    pann_df = pd.DataFrame(
        {
            "perturbation_strength": np.repeat(
                pann_data["perturbation_strength"], pann_data["id_chart"].shape[1]
            ),
            "estimated_id": pann_data["id_chart"].flatten(),
        }
    )
    clap_df = pd.DataFrame(
        {
            "perturbation_strength": np.repeat(
                clap_data["perturbation_strength"], clap_data["id_chart"].shape[1]
            ),
            "estimated_id": clap_data["id_chart"].flatten(),
        }
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.lineplot(
        data=pann_df,
        x="perturbation_strength",
        y="estimated_id",
        ax=ax_l,
    )
    sns.lineplot(
        data=clap_df,
        x="perturbation_strength",
        y="estimated_id",
        ax=ax_r,
    )
    fig.suptitle(
        f"PANN vs CLAP {params['estimator']} ID across {params['augmentation']} $\\pm${pann_df['perturbation_strength'].max():.1f} for {params['model']}"
    )
    ax_l.set_title(f"PANN {params['estimator']}")
    ax_r.set_title(f"CLAP {params['estimator']}")
    ax_l.set_xlabel(plot_x_label(params))
    ax_r.set_xlabel(plot_x_label(params))
    ax_l.set_ylabel("Estimated ID")
    ax_l.set_ylim(
        0.8,
        max(
            pann_data["id_chart"].mean(axis=1).max(),
            clap_data["id_chart"].mean(axis=1).max(),
        )
        * 1.1,
    )  # Set y-axis limit for better visualization
    fig.tight_layout()
    plt.savefig(
        plot_dir
        / "pann_vs_clap"
        / f"intrinsic_dimensionality-{params['augmentation']}-{params['estimator']}-{params['config']}-{params['model']}-pannvsclap.png",
    )
    plt.close()


if __name__ == "__main__":
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "local_vs_global").mkdir(
        exist_ok=True
    )  # Create subdirectory for local vs global comparison
    (plot_dir / "pann_vs_clap").mkdir(
        exist_ok=True
    )  # Create subdirectory for PANN vs CLAP comparison
    (plot_dir / "by_class").mkdir(
        exist_ok=True
    )  # Create subdirectory for class-wise plots
    all_data = list(id_estimate_dir.glob("intrinsic_dimensionality_*.npz"))

    is_local_options = [True, False]
    model_options = ["PANN", "CLAP", "encodec"]
    # model_options = ["encodec"]
    aug_options = ["gain", "pitch_shifting", "time_stretching"]
    id_options = ["lPCA"]
    config_options = [None, "narrow_config"]

    for model, aug, id_estimator, config, is_local in product(
        model_options, aug_options, id_options, config_options, is_local_options
    ):
        local_part = "_local" if is_local else ""
        cfg_part = f"_{config}" if config is not None else ""
        id_data_path = (
            id_estimate_dir
            / f"intrinsic_dimensionality-{aug}{local_part}-{id_estimator}{cfg_part}-{model}.npz"
        )
        if not id_data_path.exists():
            print(f"Warning: No ID estimate found for {id_data_path.stem}")
            continue
        embedding_path = (
            embedding_dir
            / f"BSD10k_{model}_{aug}{'_' + config if config is not None else ''}.h5"
        )
        if not embedding_path.exists():
            print(f"Warning: No embedding found for {embedding_path.stem}")
            embedding_path = None

        try:
            make_plot(id_data_path, embedding_path=embedding_path)
        except Exception as e:
            print(f"Error processing {id_data_path.stem}: {e}")

    exit()
    # Make local vs global comparison plots
    local_paths = [p for p in all_data if "local" in p.stem]
    for local_path in tqdm(local_paths):
        params = parse_filename(local_path)
        estimator = params["estimator"] if params["estimator"] != "PCA" else "lPCA"
        config = "_" + params["config"] if params["config"] != "default" else ""
        global_path = (
            id_estimate_dir
            / f"intrinsic_dimensionality_{params['augmentation']}_{estimator}{config}_{params['model']}.npz"
        )
        if global_path.exists():
            local_vs_global_sidebyside(local_path, global_path)
        else:
            print(f"Warning: No global counterpart found for {local_path.stem}")
            print(global_path)
            print()

    # Make PANN vs CLAP comparison plots
    # pann_paths = [p for p in all_data if "PANN" in p.stem]
    # for pann_path in tqdm(pann_paths):
    #     params = parse_filename(pann_path)
    #     estimator = params["estimator"] if params["estimator"] != "PCA" else "lPCA"
    #     config = "_" + params["config"] if params["config"] != "default" else ""
    #     clap_path = (
    #         id_estimate_dir
    #         / f"intrinsic_dimensionality_{params['augmentation']}_{'local_' if params['is_local'] else ''}{estimator}{config}_CLAP.npz"
    #     )
    #     if clap_path.exists():
    #         pann_vs_clap_sidebyside(pann_path, clap_path)
    #     else:
    #         print(f"Warning: No CLAP counterpart found for {pann_path.stem}")
