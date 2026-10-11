# Augmentation, weighted sampler, postprocessing

3D metrics after stitching back to the original space, 10 validation patients, mean ± std over 3 runs.
Baseline: ENet on original data. Aug: UNet k32, lr 5e-4, DiceCE, batch 8, preprocessed data. Aug + WS: same with the weighted sampler, batch 32.

Note: due to a label bug, the WS runs oversampled trachea instead of esophagus (fixed in the code since).

## Overall

| Config | 3D Dice | 3D NSW | Eso. | Heart | Trach. | Aorta | HD95 mm | HD mm | ASSD mm |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.693 ± 0.012 | 0.676 ± 0.015 | 0.462 ± 0.032 | 0.837 ± 0.011 | 0.726 ± 0.008 | 0.745 ± 0.012 | 18.7 ± 1.2 | 52.0 ± 3.1 | 4.27 ± 0.15 |
| Aug | 0.863 ± 0.004 | 0.860 ± 0.004 | 0.753 ± 0.011 | 0.926 ± 0.004 | 0.853 ± 0.003 | 0.919 ± 0.003 | 18.9 ± 6.7 | 56.7 ± 4.6 | 2.43 ± 0.29 |
| Aug + WS | 0.870 ± 0.004 | 0.867 ± 0.004 | 0.757 ± 0.006 | 0.937 ± 0.001 | 0.868 ± 0.005 | 0.916 ± 0.005 | 12.8 ± 2.5 | 58.9 ± 15.7 | 1.99 ± 0.12 |

## Per organ

HD95 mm

| Config | Eso. | Heart | Trach. | Aorta |
|---|---|---|---|---|
| Baseline | 15.0 ± 0.4 | 14.3 ± 0.7 | 15.2 ± 1.2 | 30.3 ± 2.8 |
| Aug | 8.4 ± 0.9 | 9.1 ± 0.8 | 47.8 ± 26.4 | 10.1 ± 2.6 |
| Aug + WS | 9.1 ± 2.1 | 12.2 ± 7.4 | 10.5 ± 3.4 | 19.3 ± 3.9 |

HD mm

| Config | Eso. | Heart | Trach. | Aorta |
|---|---|---|---|---|
| Baseline | 44.7 ± 18.1 | 50.6 ± 12.6 | 38.9 ± 6.1 | 73.8 ± 6.0 |
| Aug | 68.8 ± 16.3 | 19.8 ± 1.4 | 94.5 ± 15.7 | 43.8 ± 18.5 |
| Aug + WS | 93.6 ± 48.5 | 23.7 ± 10.0 | 64.1 ± 17.2 | 54.4 ± 12.0 |

ASSD mm

| Config | Eso. | Heart | Trach. | Aorta |
|---|---|---|---|---|
| Baseline | 4.01 ± 0.17 | 5.71 ± 0.28 | 2.79 ± 0.07 | 4.57 ± 0.33 |
| Aug | 2.02 ± 0.20 | 2.70 ± 0.17 | 3.48 ± 0.87 | 1.51 ± 0.19 |
| Aug + WS | 1.88 ± 0.21 | 2.55 ± 0.44 | 1.63 ± 0.10 | 1.89 ± 0.13 |

The weighted sampler makes HD higher and much more variable (overall 56.7 ± 4.6 → 58.9 ± 15.7 mm, esophagus 68.8 ± 16.3 → 93.6 ± 48.5 mm).

## Postprocessing

LCC: keep the largest 3D connected component per organ. LCC + 50 mm: also keep components within 50 mm of it.

| Config | 3D Dice | Eso. | Heart | Trach. | Aorta | HD95 mm | HD mm | ASSD mm |
|---|---|---|---|---|---|---|---|---|
| Aug + LCC | 0.864 ± 0.006 | 0.748 ± 0.014 | 0.926 ± 0.004 | 0.863 ± 0.003 | 0.918 ± 0.007 | 10.5 ± 1.0 | 20.0 ± 1.4 | 2.00 ± 0.28 |
| Aug + LCC + 50 mm | 0.864 ± 0.004 | 0.754 ± 0.011 | 0.926 ± 0.004 | 0.856 ± 0.003 | 0.919 ± 0.003 | 9.3 ± 0.4 | 21.8 ± 0.7 | 1.87 ± 0.04 |
| Aug + WS + LCC | 0.869 ± 0.001 | 0.745 ± 0.005 | 0.938 ± 0.001 | 0.873 ± 0.004 | 0.918 ± 0.006 | 8.9 ± 0.9 | 18.2 ± 1.5 | 1.86 ± 0.29 |
| Aug + WS + LCC + 50 mm | 0.870 ± 0.004 | 0.757 ± 0.006 | 0.938 ± 0.001 | 0.869 ± 0.005 | 0.917 ± 0.005 | 7.6 ± 0.6 | 22.0 ± 2.2 | 1.64 ± 0.05 |

LCC alone cuts off correct parts of organs predicted in two pieces (esophagus Dice drops), while LCC + 50 mm keeps them and still removes distant false positives. Aug + WS + LCC + 50 mm scores highest.