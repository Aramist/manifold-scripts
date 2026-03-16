import argparse
import time
from pathlib import Path

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import soundfile as sf
import torch
from sklearn.decomposition import PCA
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
}

module_lookup = {
    "gain": transformations.Gain,
    "time_stretching": transformations.TimeStretching,
    "pitch_shifting": transformations.PitchShifting,
}


def load_audio(
    audio_path: Path, crop_length: float | None = None
) -> tuple[torch.Tensor, float]:
    """Loads an audio file and returns its waveform and sample rate

    Returns:
        tuple[torch.Tensor, float]: Audio tensor (batch, samples) and its sample rate
    """
    # note: SF reads audio as (samples, channels)
    audio, sr = sf.read(audio_path, always_2d=True)  # 5 seconds of music
    # audio = audio.mean(axis=1)
    audio = audio[:, 0]  # only use one channel

    if crop_length is not None:
        audio = audio[: min(len(audio), int(sr * crop_length))]

    audio = audio[None, None, :]  # (batch, channels, samples)
    audio = torch.from_numpy(audio).float()
    audio_with_sr = (audio, sr)
    return audio_with_sr


def convert_sr(
    audio: tuple[torch.Tensor, float], target_sr: float
) -> tuple[torch.Tensor, float]:
    """Converts audio to target sample rate

    Returns:
        tuple[torch.Tensor, float]: Audio tensor (batch, samples) and its sample rate
    """

    if abs(target_sr - audio[1]) < 1e-3:
        return audio

    orig_device = audio[0].device
    orig_dtype = audio[0].dtype
    waveform = audio[0].cpu().numpy()

    new_waveform = librosa.resample(waveform, orig_sr=audio[1], target_sr=target_sr)
    new_waveform = torch.from_numpy(new_waveform).to(orig_device).to(orig_dtype)

    return new_waveform, target_sr


def run(
    audio_dir: Path,
    augmentation: str,
    clip_length: float | None = None,
    save_to: Path | None = None,
):
    kwargs = args_for_augs[augmentation]

    loaded_audio = [
        load_audio(audio_path) for audio_path in sorted(audio_dir.glob("*.wav"))
    ]

    if not all([sr == loaded_audio[0][1] for _, sr in loaded_audio]):
        target_sr = min([sr for _, sr in loaded_audio])
        loaded_audio = [
            convert_sr(audio, target_sr=target_sr) for audio in loaded_audio
        ]

    sr = loaded_audio[0][1]

    clip_len = min([audio.shape[-1] for audio, _ in loaded_audio])
    if clip_length is not None:
        clip_len = min(clip_len, int(sr * clip_length))
    stacked_audio = (
        torch.concatenate([audio[..., :clip_len] for audio, _ in loaded_audio], dim=0),
        sr,
    )  # new shape: (batch, channels, clip_len)

    augment_module: transformations.AudioTransformation

    if augmentation not in module_lookup:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    augment_module_class = module_lookup[augmentation]
    augment_module = augment_module_class(**kwargs)

    augmented_audio = augment_module(stacked_audio)
    # augmented_audio shape: (batch, num_augs, channels, clip_len)
    print("Augmented audio: ", augmented_audio[0].shape)

    embedding_module = embeddings.PannEmbedder.from_pretrained()
    embedding_module.eval()
    start_time = time.time()
    emb = []
    for audio in tqdm(augmented_audio[0], desc="Computing embeddings"):
        emb.append(embedding_module((audio, sr)))
    emb = torch.stack(emb, dim=0)
    end_time = time.time()
    print(f"Embedding computation time: {end_time - start_time:.2f} seconds")
    # embeddings shape: (batch, num_augs, embedding_dim)
    print("Embeddings shape: ", emb.shape)
    if save_to is None:
        save_to = Path("embeddings.h5")
    with h5py.File(save_to, "w") as hf:
        hf.create_dataset("embeddings", data=emb.numpy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("augmentation", type=str, choices=list(args_for_augs.keys()))
    ap.add_argument("--clip-length", type=float, default=1.0)
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    dataset_name = args.audio_dir.stem
    augmentation_name = args.augmentation
    output_path = (
        args.output_dir / f"embedding_table_{dataset_name}_{augmentation_name}.h5"
    )
    if not Path(output_path).exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run(
            args.audio_dir,
            args.augmentation,
            clip_length=args.clip_length,
            save_to=output_path,
        )

    visualize_embeddings(output_path)
