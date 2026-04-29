import argparse
import json
import os
import typing as tp
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

from ..consts import (
    class_id_mapping,
    class_longnames,
    convert_list_to_numpy,
    convert_numpy_to_list,
    default_aug_ranges,
    embedding_dir,
    pca_embedding_dir,
)


def get_args() -> dict[str, tp.Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aug",
        type=str,
        choices=list(default_aug_ranges.keys()),
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
    cfg_part = f"_{args['config_name']}" if args["config_name"] else ""
    embedding_file = (
        embedding_dir / f"BSD10k_{args['model']}_{args['augmentation']}{cfg_part}.h5"
    )
    filter_part = (
        ""
        if not args["class_filter"]
        else "_" + ",".join(map(str, args["class_filter"]))
    )

    save_path = (
        pca_embedding_dir
        / f"pca_embeddings_{args['model']}_{args['augmentation']}{cfg_part}{filter_part}.npz"
    )
    if save_path.exists():
        print(f"PCA embeddings already exist at {save_path}, skipping computation.")
        return

    pcs, ev_ratios = make_pca_embeddings(embedding_file, args["class_filter"])
    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(save_path, pcs=pcs, ev_ratios=ev_ratios)
    print(f"Saved PCA embeddings to {save_path}")


if __name__ == "__main__":
    model_options = ["PANN", "CLAP", "encodec"]
    aug_options = list(default_aug_ranges.keys())
    config_options = [None, "narrow_config"]
    for model, aug, config in product(model_options, aug_options, config_options):
        print(f"Running with model={model}, aug={aug}, config={config}")
        try:
            if config:
                with open("manifold_scripts/narrow_config.json", "r") as f:
                    args_for_augs = json.load(f)
                convert_list_to_numpy(args_for_augs)
            args = {
                "augmentation": aug,
                "class_filter": None,
                "model": model,
                "config_name": config,
            }
            run(args)
        except Exception as e:
            print(f"Error running with model={model}, aug={aug}, config={config}: {e}")
