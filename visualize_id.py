from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plot_dir = Path("/Users/aramis/Desktop/marl_keynotes/2026-02-25_figs/id_charts")
plot_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path("/Users/aramis/Heap/computed_embeddings/intrinsic_dimensionality")
options = list(data_dir.glob("*.npz"))
print("Available options:")
for i, option in enumerate(options):
    print(f"{i}: {option.stem}")
choice = int(input("Enter the number corresponding to the desired option: "))
data_path = options[choice]
data = np.load(data_path, allow_pickle=True)

# Structure of filename:
# intrinsic_dimensionality_{augmentation}_{estimator}_{config (optional)}_{model}.npz
augmentation = data_path.stem.split("_")[2]
estimator = data_path.stem.split("_")[3]
config = (
    "_".join(data_path.stem.split("_")[4:-1])
    if len(data_path.stem.split("_")) > 5
    else "default"
)
model = data_path.stem.split("_")[-1]
print(data.files)
gains = data["perturbation_strength"]  # shape (num_perturbations,)
id_chart = data["id_chart"]  # shape (num_perturbations, num_samples)
num_perturbations, num_instances = id_chart.shape
dists = data["dists_from_center"]
print(gains)
print(dists)
exit()
minmax_samples = dists[0] * 2 + 1, dists[-1] * 2 + 1


mean_id = id_chart.mean(axis=1)

perturbation_strengths = np.repeat(gains, num_instances)
df = pd.DataFrame(
    {
        "perturbation_strength": perturbation_strengths,
        "estimated_id": id_chart.flatten(),
    }
)
# df = pd.DataFrame({"perturbation_strength": gains, "mean_id": mean_id})
ax = sns.lineplot(
    data=df,
    x="perturbation_strength",
    y="estimated_id",
)
unit = "dB" if augmentation == "gain" else "st" if "pitch" in augmentation else "units"
ax.set_title(
    f"Estimated ID ({estimator}) across {augmentation} $\\pm${perturbation_strengths.max():.1f} {unit} for {model}"
)
ax.set_xlabel(
    f"Perturbation radius ($\\pm$ {unit}, {minmax_samples[0]}-{minmax_samples[1]} samples)"
)
ax.set_ylabel("Estimated ID")
fig = plt.gcf()
fig.tight_layout()
plt.savefig(
    plot_dir
    / f"intrinsic_dimensionality_{augmentation}_{estimator}_{config}_{model}.png",
    dpi=300,
)

# fig, ax = plt.subplots()
# ax.plot(gains, mean_id)
# ax.set_xlabel("Perturbation range ($\\pm$ dB)")
# ax.set_ylabel("Mean Intrinsic Dimensionality")

# plt.show()

# num_bins =
# bin_min, bin_max = np.quantile(id_chart.flatten(), [0.01, 0.99])

# # bins = np.linspace(bin_min, bin_max, num_bins + 1, endpoint=True)o
# bins = np.arange(id_chart.max() + 1)

# histogram_image = np.zeros((id_chart.shape[0], len(bins) - 1))
# for i in range(id_chart.shape[0]):
#     bin_memberships, _ = np.histogram(id_chart[i], bins=bins)
#     histogram_image[i] = bin_memberships

# fig, ax = plt.subplots()
# im = ax.imshow(
#     histogram_image,
#     aspect="auto",
#     origin="lower",
#     extent=[bin_min, bin_max, gains[0], gains[-1]],
#     cmap="viridis",
#     norm=mpl.colors.LogNorm(vmin=1, vmax=histogram_image.max()),
# )

# cbar = fig.colorbar(im, ax=ax)

# ax.set_xlabel("lPCA estimated ID")
# ax.set_ylabel("Perturbation range ($\\pm$ dB)")

# plt.show()
