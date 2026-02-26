import argparse
import json
import os
import signal
import sys
import typing as tp
from pathlib import Path

import h5py
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audiomanifolds import embeddings, transformations

args_for_augs = {
    "gain": {"gains": np.linspace(-10, 10, 101, endpoint=True)},
    "time_stretching": {
        "ratios": np.exp(
            np.linspace(
                np.log(0.5),
                np.log(2.0),
                101,
            )
        )
    },
    "pitch_shifting": {"n_steps": np.linspace(-12, 12, 101, endpoint=True)},
    "low_pass_filter": {
        "cutoff_frequencies": np.linspace(1000, 16000, 101, endpoint=True)[
            ::-1
        ]  # Reversed so the less-impactful augmentations come first
    },
}

module_lookup = {
    "gain": transformations.Gain,
    "time_stretching": transformations.TimeStretching,
    "pitch_shifting": transformations.PitchShifting,
    "low_pass_filter": transformations.LowPassFilter,
}

model_lookup = {
    "PANN": embeddings.PannEmbedder,
    "CLAP": embeddings.CLAPAudioEmbedder,
}

static_file: h5py.File | None = None


def sigterm_handler(signal, frame):
    """
    Handles the SIGTERM signal by performing cleanup and exiting gracefully.
    """
    print("SIGTERM received. Performing graceful shutdown...")

    if static_file is not None:
        print("Closing HDF5 file...")
        static_file.close()

    print("Cleanup complete. Exiting.")
    sys.exit(0)  # Exit the program with a status code


signal.signal(signal.SIGTERM, sigterm_handler)


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


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_paths: list[Path],
        target_sr: float,
        clip_length: float | None = None,
        augmentation: tp.Callable | None = None,
        start_idx: int = 0,
    ):
        self.audio_paths = audio_paths
        self.target_sr = target_sr
        self.clip_length = clip_length
        self.augmentation = augmentation
        self.start_idx = start_idx

    def __len__(self):
        return len(self.audio_paths) - self.start_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        """Grabs audio sample and its sample rate

        Args:
            idx (int): Index of the audio sample to grab (within the audio_paths list)

        Returns:
            tuple[torch.Tensor, float]: Audio and sample rate
        """
        audio_path = self.audio_paths[idx + self.start_idx]
        audio, sr = sf.read(audio_path, always_2d=True)
        audio = audio[:, 0]  # use only one channel
        audio = audio[None, :]  # (channels, samples)
        if abs(sr - self.target_sr) > 1e-3:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        if self.clip_length is not None:
            clip_length_samples = int(sr * self.clip_length)
            audio = audio[..., :clip_length_samples]

        if audio.dtype in (np.int16, np.int32):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val

        audio_tensor = torch.from_numpy(audio).float()

        if self.augmentation is not None:
            audio_tensor, sr = self.augmentation((audio_tensor, sr))

        audio_tensor = audio_tensor.squeeze(
            0
        )  # Remove the channel dimension since we're only using one channel

        return (audio_tensor, sr)


def find_start_idx(save_to: Path) -> int:
    """Finds the first index for which an embedding does not exist.
    Creates the h5 file if it does not exist and returns 0.

    Args:
        save_to (Path): Path to the h5 file where embeddings are being saved

    Returns:
        int: First index for which an embedding does not exist
    """

    if not save_to.exists():
        with h5py.File(save_to, "w") as hf:
            pass  # Create the file if it doesn't exist
        return 0

    try:
        with h5py.File(save_to, "r") as hf:
            pass
    except:
        # File is corrupted
        save_to.unlink()
        return 0

    with h5py.File(save_to, "r") as hf:
        if "embedding" not in hf:
            return 0
        if "last_written_index" not in hf.attrs:
            return 0
        last_written_index = hf.attrs["last_written_index"]
        return last_written_index + 1


def run(
    audio_paths: list[Path],
    augmentation_name: str,
    model_name: str,
    save_to: Path,
    *,
    clip_length: float | None = None,
):
    kwargs = args_for_augs[augmentation_name]
    if save_to is None:
        raise ValueError("save_to path must be provided")

    augment_module: transformations.AudioTransformation

    if augmentation_name not in module_lookup:
        raise ValueError(f"Unknown augmentation: {augmentation_name}")

    augment_module_class = module_lookup[augmentation_name]
    augment_module = augment_module_class(**kwargs)

    if model_name not in model_lookup:
        raise ValueError(f"Unknown model: {model_name}")
    embedding_module_class = model_lookup[model_name]
    embedding_module = embedding_module_class.from_pretrained()
    embedding_module.eval()
    sr = embedding_module.expected_sample_rate
    if sr is None:
        raise ValueError("Failed to infer expected sample rate for embedder.")

    if torch.cuda.is_available():
        embedding_module = embedding_module.cuda()

    # See if we have been pre-empted
    start_idx = find_start_idx(save_to)
    print(f"Determined start index to be {start_idx}")

    dset = AudioDataset(
        audio_paths,
        target_sr=sr,
        clip_length=clip_length,
        augmentation=augment_module,
        start_idx=start_idx,
    )
    try:
        num_avail_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        num_avail_cpu = os.cpu_count() or 1
    num_workers = max(1, num_avail_cpu - 2)
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    with h5py.File(save_to, "a") as hf:
        global static_file
        static_file = hf  # For signal handler access
        try:
            for n, data in tqdm(
                enumerate(iter(dloader), start=start_idx),
                total=len(dloader),
                desc="Processing audio files",
            ):
                augmented_audio, sr = data
                with torch.no_grad():
                    if torch.cuda.is_available():
                        augmented_audio = augmented_audio.cuda()
                    # augmented_audio shape: num_augs, clip_len

                    # embedding shape: (num_augs, embedding_dim)
                    emb = embedding_module((augmented_audio, sr)).squeeze()

                if "embedding" not in hf:
                    hf.create_dataset(
                        "embedding",
                        shape=(len(audio_paths), *emb.shape),
                        dtype=np.float32,
                    )
                hf["embedding"][n] = emb.cpu().numpy()
                hf.attrs["last_written_index"] = (
                    n  # Store the last written index as an attribute for quick access
                )
        except KeyboardInterrupt:
            return
        audio_ids = [int(p.stem) for p in audio_paths]
        if "sound_id" not in hf:
            hf.create_dataset("sound_id", data=np.array(audio_ids), dtype=np.int32)
            for key, value in kwargs.items():
                hf.create_dataset(
                    f"args/{key}", data=np.array(value)
                )  # Store which augmentation params were used


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "augmentation",
        type=str,
        help="Type of augmentation to apply",
        choices=list(args_for_augs.keys()),
    )
    ap.add_argument(
        "--model",
        type=str,
        default="PANN",
        help="Which embedding model to use (default: PANN)",
        choices=list(model_lookup.keys()),
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="Augmentation params to use",
    )
    args = ap.parse_args()
    augmentation_name = args.augmentation
    model_name = args.model
    cfg_name: str | None = None
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        convert_list_to_numpy(cfg)
        args_for_augs.update(cfg)
        print(
            f"Using augmentation parameters from {args.config}: {args_for_augs[augmentation_name]}"
        )
        cfg_name = args.config.stem
    BSD_AUDIO_PATH = Path("/ext3/BSD10k_audio/")
    BSD_METADATA_PATH = Path("/ext3/bsd_id_to_class_mapping.csv")
    output_dir = Path("/scratch/at4219/")

    print(f"Using CUDA: {torch.cuda.is_available()}")
    # get lengths of all audio files to filter for longer clips
    audio_paths = sorted(BSD_AUDIO_PATH.glob("*.wav"))
    class_id_df = pd.read_csv(BSD_METADATA_PATH)
    if "durations" not in class_id_df.columns:
        durations = {}
        for audio_path in tqdm(audio_paths, desc="Getting audio durations..."):
            info = sf.info(audio_path)
            sound_id = int(audio_path.stem)
            durations[sound_id] = info.frames / info.samplerate
        class_id_df["durations"] = class_id_df["sound_id"].map(durations)
        class_id_df.to_csv(BSD_METADATA_PATH, index=False)
    else:
        durations = {
            int(row["sound_id"]): row["durations"] for _, row in class_id_df.iterrows()
        }

    min_duration = 2.0  # seconds
    max_duration = 10.0
    audio_paths = list(
        filter(lambda p: durations[int(p.stem)] >= min_duration, audio_paths)
    )

    output_path = (
        output_dir
        / f"BSD10k_{model_name}_{augmentation_name}{'_' + cfg_name if cfg_name else ''}.h5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        audio_paths,
        augmentation_name,
        model_name,
        save_to=output_path,
        clip_length=max_duration,
    )

    class_id_lookup = {
        int(row["sound_id"]): int(row["class_idx"]) for _, row in class_id_df.iterrows()
    }
    # Append information about class id to the h5 file
    # Additionally, store information about the augmentation parameters used for reproducibility
    with h5py.File(output_path, "a") as hf:
        audio_ids = hf["sound_id"][:]
        class_ids = np.array([class_id_lookup[int(audio_id)] for audio_id in audio_ids])
        if "class_id" not in hf:
            hf.create_dataset("class_id", data=class_ids)
        for key, value in args_for_augs[augmentation_name].items():
            if key not in hf:
                hf.create_dataset(
                    f"{augmentation_name}_{key}", data=np.array(value)
                )  # Store which augmentation params were used
