import json
import typing as tp
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm

from ..consts import (
    convert_list_to_numpy,
    convert_numpy_to_list,
    embedding_dir,
    gev_embedding_dir,
)


def compute_gev(
    embedding_path: Path, class_ids_to_include: tp.Optional[tp.List[int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the set of vectors which solve the generalized eigenvector problem maximizing
    x'Bx/x'Ax for the within-class scatter matrix B and across class scatter matrix A

    Args:
        embedding_path (Path): Path to the h5 file containing audio embeddings
        class_ids_to_include (tp.Optional[tp.List[int]], optional): If specified, only include samples whose class_id is in this list. If None, include all samples. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Eigenvectors and eigenvalues, sorted in descending order of eigenvalue magnitude
    """
    with h5py.File(embedding_path, "r") as hf:
        # Loads the dataset into memory
        emb: np.ndarray = np.array(
            hf["embedding"][:]
        ).squeeze()  # shape (num_files, num_augs, embedding_dim)
        if class_ids_to_include:
            all_class_ids = np.array(hf["class_id"][:])
            mask = np.isin(all_class_ids, class_ids_to_include)
            emb = emb[mask]
    print("Loaded embeddings shape: ", emb.shape)
    num_classes, num_augs, embedding_dim = emb.shape
    # First step in FDA: globally center data
    print("Centering data globally...")
    print("Casting to float64...")
    global_mean = emb.astype(np.float64)
    print("Reducing across augmentations...")
    global_mean = global_mean.mean(axis=1)
    print("Reducing across classes...")
    global_mean = global_mean.mean(axis=0)
    print("Centering...")
    for i, emb_slice in enumerate(tqdm(emb)):
        emb[i] = emb_slice - global_mean.astype(np.float32)
    emb_centered = emb
    orig_audio_emb = emb_centered[:, num_augs // 2, :].astype(
        np.float64
    )  # (num_classes, embedding_dim)

    # Compute the sum of scatter matrices within each class (audio file)
    # Target shape: (num_classes, num_features, num_features) --- sum ---> (num_features, num_features)
    within_class_scatter = np.zeros((embedding_dim, embedding_dim), dtype=np.float64)
    for emb_slice, class_mean in tqdm(
        zip(emb_centered, orig_audio_emb),
        desc="Computing within-class scatter",
        total=num_classes,
    ):
        emb_slice_centered = emb_slice - class_mean[None, :]
        within_class_scatter += emb_slice_centered.T @ emb_slice_centered

    # Compute the scatter matrix across classes
    # shape: (num_classes, features)
    # shape: (features, features)
    across_class_scatter = num_classes * orig_audio_emb.T @ orig_audio_emb

    reg = np.eye(embedding_dim) * 1e-4

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

    # Reverse them so largest come first
    generalized_eig = generalized_eig[::-1]
    generalized_eigv = (generalized_eigv.T)[::-1]  # shape (num_eigv, dim)
    return generalized_eigv, generalized_eig


if __name__ == "__main__":
    gev_embedding_dir.mkdir(exist_ok=True)
    model_options = ["PANN", "CLAP"]
    aug_options = ["gain", "pitch_shifting", "time_stretching"]
    config_options = [None, "narrow_config"]
    class_filters = []
    for model, aug, config in tqdm(
        product(model_options, aug_options, config_options),
        total=len(model_options) * len(aug_options) * len(config_options),
    ):
        print(f"Running with model={model}, aug={aug}, config={config}")
        try:
            args_for_augs = None
            if config:
                with open("narrow_config.json", "r") as f:
                    args_for_augs = json.load(f)
                convert_list_to_numpy(args_for_augs)

            cfg_part = f"_{config}" if config else ""
            embedding_file = (
                gev_embedding_dir / f"gev_embeddings_{model}_{aug}{cfg_part}.npz"
            )
            filter_part = (
                "" if not class_filters else "_" + ",".join(map(str, class_filters))
            )

            save_path = (
                gev_embedding_dir
                / f"gev_embeddings_{model}_{aug}{cfg_part}{filter_part}.npz"
            )
            if save_path.exists():
                print(
                    f"GEV embeddings already exist at {save_path}, skipping computation."
                )
                continue
            embedding_path = embedding_dir / f"BSD10k_{model}_{aug}{cfg_part}.h5"
            if not embedding_path.exists():
                print(f"Embedding file {embedding_path} does not exist, skipping.")
                continue
            gev_eigv, gev_eig = compute_gev(embedding_path)
            np.savez(embedding_file, gev_eigv=gev_eigv, gev_eig=gev_eig)
            print(f"Saved GEV embeddings to {embedding_file}")

        except Exception as e:
            print(f"Error running with model={model}, aug={aug}, config={config}: {e}")
