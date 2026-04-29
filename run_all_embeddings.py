from itertools import product

script_fmt = """#!/bin/bash
#SBATCH --job-name=comp_emb
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=2-0
#SBATCH --account=torch_pr_529_general
#SBATCH -o compute_emb_gpu.log


compute()
{
    AUG=$1
    MODEL=$2
    srun --ntasks=1 --exclusive \\
        singularity exec --nv --fakeroot \\
	    --overlay /scratch/at4219/BSD10k_audio_img.sqf:ro \\
	    --overlay /scratch/at4219/manifold_project.ext3:ro \\
	    $CUDA_IMAGE \\
	    /bin/bash -c "source /ext3/audio-embeddings/.venv/bin/activate; python ~/make_bsd_embeddings.py $AUG --model $MODEL" &
}

"""


augmentations = ["low_pass_filter", "time_stretching", "pitch_shifting", "gain"]
models = ["CLAP", "PANN", "encodec"]
combos = []
for aug, model in product(augmentations, models):
    if model == "encodec":
        combos.append((aug, model))
    elif aug == "low_pass_filter":
        combos.append((aug, model))
    elif aug == "gain" and model == "PANN":
        combos.append((aug, model))  # had to redo this one

with open("compute_embeddings.sh", "w") as f:
    f.write(script_fmt)
    total_jobs = len(combos)
    for n, (aug, model) in enumerate(combos, start=1):
        f.write(f"echo 'Computing {aug} with {model} (job {n}/{total_jobs})'\n")
        f.write(f"compute {aug} {model}\n")

    f.write("wait\n")
