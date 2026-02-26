from pathlib import Path

parent_dir = Path("/scratch/at4219/")
files = [
    "BSD10k_CLAP_gain.h5",
    "BSD10k_CLAP_gain_narrow_config.h5",
    "BSD10k_CLAP_pitch_shifting.h5",
    "BSD10k_CLAP_pitch_shifting_narrow_config.h5",
    "BSD10k_CLAP_time_stretching.h5",
    "BSD10k_CLAP_time_stretching_narrow_config.h5",
    "BSD10k_PANN_gain.h5",
    "BSD10k_PANN_gain_narrow_config.h5",
    "BSD10k_PANN_pitch_shifting.h5",
    "BSD10k_PANN_pitch_shifting_narrow_config.h5",
    "BSD10k_PANN_time_stretching.h5",
    "BSD10k_PANN_time_stretching_narrow_config.h5",
]
files = list(map(lambda f: parent_dir / f, files))

fmt = '''srun singularity exec --nv --fakeroot \\
            --overlay /scratch/at4219/manifold_project.ext3:ro \\
            $CUDA_IMAGE \\
            /bin/bash -c "source /ext3/audio-embeddings/.venv/bin/activate; python ~/estimate_manifold_dim_from_center.py {}"'''

fmt_pointwise = '''srun singularity exec --nv --fakeroot \\
            --overlay /scratch/at4219/manifold_project.ext3:ro \\
            $CUDA_IMAGE \\
            /bin/bash -c "source /ext3/audio-embeddings/.venv/bin/activate; python ~/estimate_manifold_dim_from_center.py {} --pointwise"'''


for n, file in enumerate(files, start=1):
    print(f"echo 'Processing {file.stem} ({n}/{len(files)*2})'")
    print(fmt.format(file))
    print()


for n, file in enumerate(files, start=1 + len(files)):
    print(f"echo 'Processing {file.stem} ({n}/{len(files)*2})'")
    print(fmt_pointwise.format(file))
    print()


from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def squeeze_h5_files_in_directory(directory: Path):
    for h5_file in tqdm(list(directory.glob("*.h5"))):
        try:
            with h5py.File(h5_file, "r+") as hf:
                for dataset_name in hf.keys():
                    cur_shape = hf[dataset_name].shape
                    target_shape = tuple(dim for dim in cur_shape if dim != 1)
                    if target_shape == cur_shape:
                        continue  # already squeezed
                    data = hf[dataset_name][()]
                    squeezed_data = np.squeeze(data)
                    del hf[dataset_name]
                    hf.create_dataset(dataset_name, data=squeezed_data)
                    print(
                        f"Squeezed dataset '{dataset_name}' in file '{h5_file.name}'."
                    )
        except OSError as e:
            h5_file.unlink()  # file si corrupted, delete it


target_dir = Path("/scratch/at4219")
squeeze_h5_files_in_directory(target_dir)
