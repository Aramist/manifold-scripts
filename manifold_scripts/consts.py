import json
from pathlib import Path

import numpy as np

embedding_dir = Path("/Users/aramis/Heap/computed_embeddings")
gev_embedding_dir = Path("/Users/aramis/Heap/computed_embeddings/gev_embeddings")
pca_embedding_dir = Path("/Users/aramis/Heap/computed_embeddings/pca_embeddings")
id_estimate_dir = Path(
    "/Users/aramis/Heap/computed_embeddings/intrinsic_dimensionality"
)


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


class_id_mapping = {}
with open(Path(__file__).parent / "class_id_mapping.json", "r") as f:
    class_id_mapping = json.load(f)

class_longnames = {}
with open(Path(__file__).parent / "class_longnames.json", "r") as f:
    class_longnames = json.load(f)
