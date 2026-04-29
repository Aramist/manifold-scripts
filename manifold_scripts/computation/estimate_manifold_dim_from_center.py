import argparse
import time
import typing as tp
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import skdim
from joblib import Parallel, delayed
from tqdm import tqdm

from ..consts import aug_parameter_names, default_aug_ranges, id_estimate_dir

min_dist_from_center = 3  # Minimum number of augmentations from the center to consider for dimension estimation
ESTIMATOR_CLS = skdim.id.lPCA
ESTIMATOR_NAME = ESTIMATOR_CLS.__name__
RUN_PARALLEL: bool = False


def make_output_path(
    cfg_name: str, aug: str, using_local_estimator: bool, model: str
) -> Path:

    cfg_section = f"_{cfg_name}" if cfg_name else ""
    pointwise_section = "_local" if using_local_estimator else ""
    output_path = (
        id_estimate_dir
        / f"intrinsic_dimensionality-{aug}{pointwise_section}-{ESTIMATOR_NAME}{cfg_section}-{model}.npz"
    )
    return output_path


def parse_embedding_file_name(embedding_file: Path) -> tp.Tuple[str, str, str]:
    # File is named like: BSD10k_{MODEL}_{aug_multiple_words}_{cfg_name_optional}
    file_stem = embedding_file.stem

    MODEL = file_stem.split("_")[1]
    aug_options = list(aug_parameter_names.keys())
    AUG = next(filter(lambda aug: aug in file_stem, aug_options), None)
    if AUG is None:
        raise ValueError(
            f"Could not parse augmentation from file name {embedding_file}. Expected one of {aug_options} to be in the file name."
        )
    num_words_in_aug = len(AUG.split("_"))
    CFG_NAME = "_".join(file_stem.split("_")[2 + num_words_in_aug :])
    return MODEL, AUG, CFG_NAME


def check_chart(id_chart_path: Path) -> bool:
    """Checks an intrinsic dimensionality chart to see if it has been
    fully computed already

    Args:
        id_chart_path (Path): Path to program output

    Returns:
        bool: True if the chart looks valid and complete
    """

    if id_chart_path.exists():
        try:
            loaded = np.load(id_chart_path)
            id_chart = loaded["id_chart"]
            incomplete_rows = np.isnan(id_chart).any(axis=1)
            if incomplete_rows.any():
                return False
        except:
            return False
    else:
        return False

    return True


def custom_pointwise_estimator(X: np.ndarray, neighbor_radius=5) -> float:
    """Computes local ID estimates for a manifold

    Args:
        X (np.ndarray): The data to compute the ID for. Shape (num_points, embedding_dim)
        neighbor_radius (int, optional): The radius of the neighborhood to consider for each point. Defaults to 5.

    Returns:
        float: The average local ID estimate across all points
    """
    num_samples = X.shape[0]
    dims = np.empty(num_samples - 2 * neighbor_radius, dtype=np.float64)
    for center_idx in range(neighbor_radius, num_samples - neighbor_radius):
        neighborhood = X[
            center_idx - neighbor_radius : center_idx + neighbor_radius + 1
        ]
        neighborhood_dim = ESTIMATOR_CLS().fit(neighborhood).dimension_
        dims[center_idx - neighbor_radius] = neighborhood_dim
    return dims.mean()


def job(aug_subset: np.ndarray, use_local_estimator: bool) -> np.ndarray:
    # Given a bunch of samples (num_samples, num_augs, embedding_dim), compute the intrinsic dimensionality
    # of the manifold for each sample and returns an array of shape (num_samples,)
    output = np.zeros(aug_subset.shape[0], dtype=np.float64)
    for j, sample in enumerate(aug_subset):
        if use_local_estimator:
            # intrinsic_dim = np.mean(
            #     ESTIMATOR_CLS().fit_pw(sample, n_neighbors=3).dimension_pw_
            # )
            # sample: (n_augs, embedding_dim)
            intrinsic_dim = custom_pointwise_estimator(sample, neighbor_radius=3)
        else:
            intrinsic_dim = ESTIMATOR_CLS().fit(sample).dimension_
        output[j] = intrinsic_dim
    return output


def run_for_embedding_file(
    embedding_file: Path,
    use_pointwise_estimate: bool,
) -> None:
    # Parse embedding file to produce output path
    model, aug, cfg_name = parse_embedding_file_name(embedding_file)
    output_path = make_output_path(
        cfg_name=cfg_name,
        using_local_estimator=use_pointwise_estimate,
        model=model,
        aug=aug,
    )

    if check_chart(output_path):
        print(f"Chart at {output_path} looks complete, skipping computation.")
        return

    hf = h5py.File(embedding_file, "r")
    emb_dataset = hf["embedding"]  # shape (num_files, num_augs, embedding_dim)
    num_classes, num_augs, _ = emb_dataset.shape

    def arg_generator():
        for dist_from_center in reversed(range(min_dist_from_center, num_augs // 2)):
            subset = emb_dataset[
                :,
                num_augs // 2 - dist_from_center : num_augs // 2 + dist_from_center + 1,
                :,
            ]
            yield subset

    # Also save the magnitude of the perturbations
    dists_from_center = np.array(list(range(min_dist_from_center, num_augs // 2)))
    aug_param_key = aug_parameter_names[aug][
        0
    ]  # e.g. "gains" for gain, "ratios" for time_stretching, etc.
    if aug_param_key is None:
        raise ValueError(f"No augmentation parameter found for AUG={aug}")
    if aug_param_key in hf:
        aug_param_values = hf[aug_param_key][:]
    elif f"args/{aug_param_key}" in hf:
        aug_param_values = hf[f"args/{aug_param_key}"][:]
    else:
        aug_param_values = default_aug_ranges[aug]
    perturb_magnitudes = aug_param_values[dists_from_center + num_augs // 2]

    id_chart = np.full((len(dists_from_center), num_classes), np.nan, dtype=np.float64)

    # Running the largest jobs first so if it's gonna die it dies early
    if RUN_PARALLEL:
        results = Parallel(return_as="generator")(
            delayed(job)(subset, use_pointwise_estimate) for subset in arg_generator()
        )
        for i, result in zip(
            reversed(range(len(dists_from_center))),
            tqdm(results, total=len(dists_from_center)),
        ):
            id_chart[i] = result
            # Save after each row in case the script dies
            np.savez(
                output_path,
                id_chart=id_chart,
                perturbation_strength=perturb_magnitudes,
                dists_from_center=dists_from_center,
            )
    else:
        for i, subset in zip(
            reversed(range(len(dists_from_center))),
            tqdm(arg_generator(), total=len(dists_from_center)),
        ):
            id_chart[i] = job(subset, use_pointwise_estimate)
            np.savez(
                output_path,
                id_chart=id_chart,
                perturbation_strength=perturb_magnitudes,
                dists_from_center=dists_from_center,
            )

    np.savez(
        output_path,
        id_chart=id_chart,
        perturbation_strength=perturb_magnitudes,
        dists_from_center=dists_from_center,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "embedding_file",
        type=Path,
        help="Path to the h5 file containing the embeddings. The file should have a dataset named 'embedding' of shape (num_files, num_augs, embedding_dim).",
    )
    ap.add_argument(
        "--pointwise",
        action="store_true",
        help="Whether to use pointwise intrinsic dimensionality estimation (fit_pw) or not (fit). Pointwise estimation computes the intrinsic dimensionality for each point and averages them, while non-pointwise estimation computes a single intrinsic dimensionality for the entire dataset.",
    )
    args = ap.parse_args()
    embedding_file = args.embedding_file
    use_pointwise_estimate = args.pointwise

    run_for_embedding_file(embedding_file, use_pointwise_estimate)
