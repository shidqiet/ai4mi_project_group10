from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

data_dir = Path("data/segthor_part1/train")
patients = sorted(data_dir.glob("Patient_*"))

dx_list = []
dy_list = []
dz_list = []

for patient in patients:
    ct = nib.load(str(patient / f"{patient.name}.nii.gz"))
    dx, dy, dz = ct.header.get_zooms()

    dx_list.append(round(float(dx), 3))
    dy_list.append(round(float(dy), 3))
    dz_list.append(round(float(dz), 3))

    print(patient.name, "spacing:", round(dx, 3), round(dy, 3), round(dz, 3))

print()
print("dx counts:", Counter(dx_list))
print("dy counts:", Counter(dy_list))
print("dz counts:", Counter(dz_list))

# use median instead of mean to handle outlier
print()
print("median dx:", np.median(dx_list))
print("median dy:", np.median(dy_list))
print("median dz:", np.median(dz_list))
