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
# Assuming the file is named like "BSD10k_{MODEL}_{AUG}_{cfg_name_here}.h5"
MODEL = embedding_file.stem.split("_")[1]
AUG = embedding_file.stem.split("_")[2]
if AUG.lower() != "gain":
    # Two word augmentation, e.g. time_stretching or pitch_shifting
    AUG = "_".join(embedding_file.stem.split("_")[2:4])
    CFG_NAME = "_".join(embedding_file.stem.split("_")[4:])
else:
    CFG_NAME = "_".join(embedding_file.stem.split("_")[3:])

min_dist_from_center = 2  # Minimum number of augmentations from the center to consider for dimension estimation
ESTIMATOR_CLS = skdim.id.lPCA
ESTIMATOR_NAME = ESTIMATOR_CLS.__name__
USE_POINTWISE_ESTIMATE = args.pointwise
RUN_PARALLEL: bool = False


if Path("/scratch/at4219/").exists():
    data_dir = Path(f"/scratch/at4219/covs/")
else:
    data_dir = Path(f"/Users/aramis/covs/")


data_dir.mkdir(parents=True, exist_ok=True)
cfg_section = f"_{CFG_NAME}" if CFG_NAME else ""
pointwise_section = "_local" if USE_POINTWISE_ESTIMATE else ""
output_path = (
    data_dir
    / f"intrinsic_dimensionality_{AUG}{pointwise_section}_{ESTIMATOR_NAME}{cfg_section}_{MODEL}.npz"
)

if output_path.exists():
    try:
        # If all of this works, we can exit without wasting compute
        print(f"Output file {output_path} already exists. Checking if it's valid...")
        loaded = np.load(output_path)
        id_chart = loaded["id_chart"]
        zero_rows = (np.abs(id_chart) < 1e-4).all(axis=1)
        if zero_rows.any():
            print(
                f"Warning: Found {zero_rows.sum()} rows in the intrinsic dimensionality chart that are all zeros. This might indicate a problem with the estimation. Please check the output file {output_path}."
            )
        if zero_rows.sum() > 5:
            print(
                f"Error: Found {zero_rows.sum()} rows in the intrinsic dimensionality chart that are all zeros. Re-running estimation..."
            )
        else:
            print("Loaded intrinsic dimensionality chart shape: ", id_chart.shape)
            print()
            print()
            exit()
    except:
        pass

hf = h5py.File(embedding_file, "r")
emb_dataset = hf["embedding"]  # shape (num_files, num_augs, embedding_dim)
print("Loaded embeddings shape: ", emb_dataset.shape)
num_classes, num_augs, embedding_dim = emb_dataset.shape


def job(aug_subset):
    # Given a bunch of samples (num_samples, num_augs, embedding_dim), compute the intrinsic dimensionality
    # of the manifold for each sample and returns an array of shape (num_samples,)
    output = np.zeros(aug_subset.shape[0], dtype=np.float64)
    for j, sample in enumerate(aug_subset):
        if USE_POINTWISE_ESTIMATE:
            intrinsic_dim = np.mean(
                ESTIMATOR_CLS().fit_pw(sample, n_neighbors=3).dimension_pw_
            )
        else:
            intrinsic_dim = ESTIMATOR_CLS().fit(sample).dimension_
        output[j] = intrinsic_dim
    return output


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
aug_param_key = list(args_for_augs[AUG].keys())[0]
if aug_param_key is None:
    raise ValueError(f"No augmentation parameter found for AUG={AUG}")
print(hf.keys())
if aug_param_key in hf:
    aug_param_values = hf[aug_param_key][:]
elif f"args/{aug_param_key}" in hf:
    aug_param_values = hf[f"args/{aug_param_key}"][:]
else:
    aug_param_values = args_for_augs[AUG][list(args_for_augs[AUG].keys())[0]]
perturb_magnitudes = aug_param_values[dists_from_center + num_augs // 2]

id_chart = np.full((len(dists_from_center), num_classes), np.nan, dtype=np.float64)

# Running the largest jobs first so if it's gonna die it dies early
if RUN_PARALLEL:
    results = Parallel(n_jobs=2, return_as="generator")(
        delayed(job)(subset) for subset in arg_generator()
    )
    for i, result in zip(
        reversed(range(len(dists_from_center))),
        tqdm(results, total=len(dists_from_center)),
    ):
        id_chart[i] = result
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
        id_chart[i] = job(subset)
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
