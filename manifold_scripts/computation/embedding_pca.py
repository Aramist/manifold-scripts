import argparse
import json
import typing as tp
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA


def convert_list_to_numpy(data: dict) -> None:
    """Recursively converts lists in a dictionary to numpy arrays. Modifies the dictionary in place."""
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = np.array(value)
        elif isinstance(value, dict):
            convert_list_to_numpy(value)


def convert_numpy_to_list(data: dict) -> None:
    """Recursively converts numpy arrays in a dictionary to lists. Modifies the dictionary in place."""
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
        elif isinstance(value, dict):
            convert_numpy_to_list(value)


with open("default_config.json", "r") as ctx:
    default_args = json.load(ctx)
    convert_list_to_numpy(default_args)
with open("class_id_mapping.json", "r") as ctx:
    class_id_mapping = json.load(ctx)
with open("class_longnames.json", "r") as ctx:
    class_longnames = json.load(ctx)

running_on_cluster = Path(f"/home/at4219/scratch/").exists()
EMBEDDING_DIR = (
    Path("/home/at4219/scratch/computed_embeddings/")
    if running_on_cluster
    else Path("/Users/aramis/Heap/computed_embeddings/")
)


def get_args() -> dict[str, tp.Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aug",
        type=str,
        choices=list(default_args.keys()),
        required=True,
        help="The type of augmentation to visualize.",
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
    ap.add_argument(
        "--config",
        type=Path,
        help="Path to json file containing config used to generate augmentations.",
    )
    args = ap.parse_args()
    AUG = args.aug
    CLASS_FILTER = args.class_filter
    MODEL = args.model
    CONFIG = args.config
    CONFIG_NAME = CONFIG.stem if CONFIG else None
    if CLASS_FILTER:
        class_filters = CLASS_FILTER.split(",")
        # Sort for consistency
        class_filters.sort()
        class_ids_to_include = []
        for cf in class_filters:
            for class_name, class_id in class_id_mapping.items():
                if class_name.startswith(cf):
                    class_ids_to_include.append(class_id)

    if CONFIG:
        with open(CONFIG, "r") as f:
            args_for_augs = json.load(f)
        convert_list_to_numpy(args_for_augs)

    return {
        "augmentation": AUG,
        "class_filter": class_ids_to_include if CLASS_FILTER else None,
        "model": MODEL,
        "config_name": CONFIG_NAME,
        "config": args_for_augs if CONFIG else None,
    }


def make_pca_embeddings(
    embedding_path: Path, class_ids_to_include: tp.Optional[tp.List[int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(embedding_path, "r") as hf:
        emb: np.ndarray = hf["embedding"][
            :
        ].squeeze()  # shape (num_files, num_augs, embedding_dim)
        if class_ids_to_include is not None:
            all_class_ids = hf["class_id"][:]
            mask = np.isin(all_class_ids, class_ids_to_include)
            emb = emb[mask]
    print("Loaded embeddings shape: ", emb.shape)
    num_classes, num_augs, embedding_dim = emb.shape

    # Global PCA
    pca = PCA(n_components=0.95)
    emb_flat = emb.reshape(-1, embedding_dim)
    emb_pca = pca.fit_transform(emb_flat)
    ev_ratios = pca.explained_variance_ratio_

    return emb_pca, ev_ratios


def run(args: dict[str, tp.Any]) -> None:
    # see which system we're on
    if running_on_cluster:
        raise RuntimeError("Don't run this on the cluster")
        cfg_part = f"_{args['config_name']}" if args["config_name"] else ""
        embedding_file = (
            EMBEDDING_DIR
            / f"BSD10k_{args['model']}_{args['augmentation']}{cfg_part}.h5"
        )
        data_dir = Path(
            f"/home/at4219/scratch/covs/{args['model']}_{args['augmentation']}_{'class_mean' if args['config']['use_class_means'] else 'orig_emb'}{'' if not args['class_filter'] else '_' + ','.join(map(str, args['class_filter']))}"
        )
        plot_dir = None
    else:
        cfg_part = f"_{args['config_name']}" if args["config_name"] else ""
        embedding_file = (
            EMBEDDING_DIR
            / f"BSD10k_{args['model']}_{args['augmentation']}{cfg_part}.h5"
        )
        filter_part = (
            ""
            if not args["class_filter"]
            else "_" + ",".join(map(str, args["class_filter"]))
        )

    save_path = (
        Path("/Users/aramis/Heap/computed_embeddings/pca_embeddings")
        / f"pca_embeddings_{args['model']}_{args['augmentation']}{cfg_part}{filter_part}.npz"
    )
    if save_path.exists():
        print(f"PCA embeddings already exist at {save_path}, skipping computation.")
        return

    pcs, ev_ratios = make_pca_embeddings(embedding_file, args["class_filter"])
    np.savez(save_path, pcs=pcs, ev_ratios=ev_ratios)
    print(f"Saved PCA embeddings to {save_path}")


if __name__ == "__main__":
    model_options = ["PANN", "CLAP"]
    aug_options = list(default_args.keys())
    config_options = [None, "narrow_config"]
    for model, aug, config in product(model_options, aug_options, config_options):
        print(f"Running with model={model}, aug={aug}, config={config}")
        try:
            if config:
                with open("narrow_config.json", "r") as f:
                    args_for_augs = json.load(f)
                convert_list_to_numpy(args_for_augs)
            args = {
                "augmentation": aug,
                "class_filter": None,
                "model": model,
                "config_name": config,
                "config": args_for_augs if config else None,
            }
            run(args)
        except Exception as e:
            print(f"Error running with model={model}, aug={aug}, config={config}: {e}")
