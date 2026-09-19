from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

data_dir = Path("data/segthor_part1/train")
patients = sorted(data_dir.glob("Patient_*"))

all_values = []
fg_values = []  # fg = heart, aorta, trachea, esophagus

for patient in patients:
    ct = np.asarray(nib.load(str(patient / f"{patient.name}.nii.gz")).dataobj)
    gt = np.asarray(nib.load(str(patient / "GT.nii.gz")).dataobj)

    all_values.append(ct.flatten())
    fg_values.append(ct[gt > 0])  # filter only voxel that labelled

    print(patient.name, "HU range:", ct.min(), ct.max())

all_values = np.concatenate(all_values)
fg_values = np.concatenate(fg_values)

p_low, p_high = np.percentile(fg_values, [0.5, 99.5])
print("Foreground Percentile 0.5:", p_low, "99.5:", p_high)

# plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

ax1.hist(all_values, bins=100)
ax1.set_title("All Voxels")
ax1.set_xlabel("Hounsfield Units (HU)")
ax1.set_ylabel("Voxel Count")

ax2.hist(fg_values, bins=100)
ax2.axvline(p_low, color="red", linestyle="--", label=f"p0.5 = {p_low:.0f}")
ax2.axvline(p_high, color="red", linestyle="--", label=f"p99.5 = {p_high:.0f}")
ax2.set_title("Foreground Voxels")
ax2.set_xlabel("Hounsfield Units (HU)")
ax2.set_ylabel("Voxel Count")
ax2.legend()

# log scale
ax3.hist(all_values, bins=100)
ax3.set_yscale("log")
ax3.set_title("All Voxels (log scale)")
ax3.set_xlabel("Hounsfield Units (HU)")
ax3.set_ylabel("Voxel Count (log scale)")

ax4.hist(fg_values, bins=100)
ax4.axvline(p_low, color="red", linestyle="--", label=f"p0.5 = {p_low:.0f}")
ax4.axvline(p_high, color="red", linestyle="--", label=f"p99.5 = {p_high:.0f}")
ax4.set_yscale("log")
ax4.set_title("Foreground Voxels (log scale)")
ax4.set_xlabel("Hounsfield Units (HU)")
ax4.set_ylabel("Voxel Count (log scale)")
ax4.legend()

plt.tight_layout()
plt.savefig("EDA/hu_histogram.png")
