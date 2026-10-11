# AI for Medical Imaging — Fall 2026 group project

This repository contains our SegTHOR multi-organ CT segmentation project. It
supports the full workflow: prepare 3D CT volumes, train 2D or 2.5D models,
evaluate volume-level metrics, reconstruct predictions in the original image
space, and reproduce our experiment sweeps.

SegTHOR contains thoracic CT scans labelled for the esophagus, heart, trachea,
and aorta. The project began from the teaching baseline maintained by Hoel
Kervadec and is a fork of the
[MICCAI weakly supervised segmentation tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial).

## What this project adds

Since the initial baseline, the project has gained:

- **SegTHOR preprocessing:** percentile-based CT clipping, 3D CLAHE, common
  spacing resampling, fixed field-of-view crop/pad handling, and both
  preprocessed and original-space slice datasets.
- **Architectures:** ShallowNet, ENet, U-Net, and Swin-UNet; U-Net supports
  2D and neighbouring-slice 2.5D input.
- **Training controls:** configurable architecture settings, seed and batch
  size, Adam/AdamW/Nesterov SGD, polynomial scheduling, early stopping,
  weighted foreground sampling, and consistent 2.5D augmentation.
- **Losses and selection:** CE, Dice, DiceCE, and NSW DiceCE; model selection
  by 2D Dice, 3D Dice, HD95, or ASSD.
- **Volume evaluation:** per-patient/per-organ 3D Dice, HD95, HD, and ASSD,
  plus automatic plots, CSV/JSON summaries, and elapsed-time recording.
- **Inference:** original-space NIfTI reconstruction and optional per-organ
  connected-component postprocessing.
- **Reproducible experiments:** Slurm templates, plotting scripts, and reports
  for architecture, ENet, loss, optimiser, adjacent-slice, augmentation,
  sampling, and postprocessing studies.

Detailed methodology and results are in [`docs/experiments/`](docs/experiments/).

## Development timeline

This timeline records the substantive work merged to `origin/master` after the
project moved beyond Hoel Kervadec's teaching baseline. Dates are commit dates.
It describes the implementation path, not just the resulting command-line options.

### 13–16 September — make training configurable and introduce Dice objectives

The first training changes added AdamW and Nesterov SGD alongside the existing
Adam optimiser, plus polynomial learning-rate decay. These were exposed as
training arguments so later runs could compare optimiser/schedule combinations
without modifying the loop. Dice loss and combined Dice–cross-entropy support
followed, including repair of the aorta label in the merged SegTHOR masks and a
`--loss` selector in the training entry point. This established alternatives to
the original cross-entropy-only baseline.

### 17–21 September — build a geometry-aware SegTHOR pipeline

Exploratory checks of Hounsfield-unit and voxel-spacing distributions informed
the preprocessing design. The slicer clips CT intensities to robust
percentiles, applies 3D CLAHE, and resamples both CT and ground truth to the
dataset's median spacing before producing PNG slices. It saves the original
geometry and preprocessing statistics while doing so.

Those saved statistics are crucial: stitching first reverses the resize and
crop/pad operations, then resamples the prediction back into the original NIfTI
space. Fixed-size centre crop/pad was added to give models a consistent
in-plane field of view, with the inverse operation added to stitching at the
same time. This phase also added Swin-UNet and a U-Net implementation that can
operate in 2D or 2.5D: neighbouring slices are stacked as input channels rather
than processed as an independent 3D volume.

### 27 September–1 October — evaluate the 3D clinical object, not only slices

Validation was extended from slice-wise Dice to per-patient reconstructed
volumes. The training loop collects validation slices by patient, reconstructs
each volume, and calculates foreground 3D Dice. It then added spacing-aware
surface metrics—HD95, full HD, and ASSD—using shared directed surface-distance
calculations. The saved preprocessing spacing makes those distances meaningful
in millimetres rather than pixels.

Checkpoints can now be selected by 2D Dice, 3D Dice, HD95, or ASSD; Dice scores
are maximised while distances are minimised. Early stopping was added on the
selected validation metric, and the default epoch cap became 50.

### 2–4 October — make fairness across organs visible and selectable

The Nash social welfare (NSW) aggregator was introduced to avoid a high average
Dice concealing a poorly segmented organ. For positive per-organ scores it uses
the geometric mean; in the loss it is implemented in log form for numerical
stability. `NSWDiceCELoss` combines that pooled Dice term with cross-entropy.
NSW can also aggregate Dice for model selection.

This feature was paired with automatic metric reporting: foreground curves,
per-organ plots, selected-epoch patient distributions, heatmaps, correlations,
and CSV summaries. Seeds and batch-size overrides were added so comparisons are
repeatable and practical, and plot label handling was corrected.

### 5–7 October — make richer evaluation affordable and reproducible

Computing volume metrics every epoch can be expensive. Distance calculations
were reduced to a bounding box around the relevant masks, and
`--distance-metrics-at best` was added to defer HD95/HD/ASSD until the selected
epoch when appropriate. Validation PNGs are likewise retained for the best
epoch only. Compound-loss terms and total wall-clock time are recorded in run
artifacts, making sweep results easier to audit.

The same period added Slurm array-job templates and figure-generation scripts
for architecture and ENet sweeps, followed by their reports. The preprocessing
script was extended to emit both preprocessed and original-space variants,
while augmentation added rotations, translations, scaling, brightness changes,
and noise to training data only.

### 8–10 October — consolidate the experimental recipe and improve robustness

The SegTHOR split was fixed to the 30/10 patient recipe used across the
experiments. A weighted random sampler then increased the frequency of
foreground and particularly esophagus-containing slices, addressing the
background-heavy training distribution.

2.5D augmentation was first blocked because independent transformations of
neighbouring slices would destroy their spatial correspondence; it was then
implemented correctly by sampling one transform and applying it consistently
to every slice in the 2.5D input. The final inference improvements added
per-organ 3D postprocessing: keep the largest connected component and nearby
components within a physical-distance threshold, while warning if more than
10% of an organ is removed. Stitching original-space slices no longer requires
preprocessing statistics because no inverse resampling is needed.

This phase also completed the adjacent-slice, DiceCE-vs-NSW, optimiser, and
augmentation/sampling/postprocessing reports. Together, the reports document
how the final experimental choices were tested rather than presenting a single
unexplained configuration.

## Layout

```text
docs/                  Documentation, figures, and experiment reports
src/ai4mi/             Installable Python package
  data/                Dataset loading
  models/              Segmentation architectures
  preprocessing/       SegTHOR slicing and preprocessing
  training/            Training loop and losses
  inference/           2D-to-3D reconstruction
  evaluation/          Metric plotting and reporting
tools/                 Data utilities, plots, HPC jobs, and viewer submodule
  data/                Local datasets and versioned checksum files
```

Generated data, results, Slurm logs, and local environments are ignored by
Git. SegTHOR scans are not distributed in this repository.

## Setup

### Prerequisites

- Python 3.10 or newer
- Git
- `make`, `unzip`, and `sha256sum` for the data workflow
- A CUDA-capable PyTorch installation for practical SegTHOR training (CPU is
  sufficient for development and smoke tests)

Clone with the viewer submodule:

```bash
git clone --recurse-submodules https://github.com/shidqiet/ai4mi_project_group10.git
cd ai4mi_project_group10
```

For an existing clone, initialise the submodule instead:

```bash
git submodule update --init --recursive
```

Create a local environment and install both dependencies and the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the installation:

```bash
python -m ai4mi.training.main --help
python -m ai4mi.preprocessing.slice_segthor --help
python -m ai4mi.inference.stitch --help
```

Conda is also suitable, particularly on HPC systems: create a Python 3.10+
environment and run the same two `pip` installation commands within it. Avoid
installing the same dependency through both Conda and pip.

## Prepare data

### Synthetic TOY2 dataset

Generate the synthetic two-circles dataset for a quick local test:

```bash
make tools/data/TOY2
```

This creates `tools/data/TOY2/`. The legacy `tools/data/TOY` Make target is
not maintained; use `TOY2`.

### Full SegTHOR training dataset

Obtain the full SegTHOR training dataset through the
[course-provided SegTHOR link](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/IQBJLXRY5wedSYEuofqRtuylAWiiHp2ciems5XSCu3DFMkA?e=qa3Ujf)
and extract it under `tools/data/` so that the source scans are available at
`tools/data/segthor_train_full/`. The dataset is not versioned and requires
the appropriate UvA access.

Once the full dataset is extracted, generate the slice datasets:

```bash
make tools/data/SEGTHOR
```

The latter command uses the standard deterministic 30/10 train/validation
split, 256 × 256 in-plane shape, and seed 0. It creates:

- `tools/data/SEGTHOR/` — preprocessed slices and `preprocess_stats.pkl`, required
  to restore predictions to their original geometry.
- `tools/data/SEGTHOR_ORIGINAL/` — slices retaining original geometry.

Inspect all preprocessing options with:

```bash
python -m ai4mi.preprocessing.slice_segthor --help
```

For direct preprocessing, use a temporary output directory and rename it only
after success; `--process -1` uses all available CPU cores.

## Train

All commands use `python -m ai4mi...` and should write outputs under ignored
`results/`.

### Local smoke test

```bash
python -m ai4mi.training.main \
  --dataset TOY2 \
  --net shallow \
  --epochs 1 \
  --debug \
  --dest results/smoke/toy2
```

Append `--gpu` only if the active PyTorch installation can access CUDA.

### SegTHOR example

This representative full-supervision 2.5D U-Net run selects by mean 3D Dice:

```bash
python -m ai4mi.training.main \
  --dataset SEGTHOR \
  --net unet \
  --net_kwargs '{"kernels": 64}' \
  --adjacent_slices 1 \
  --mode full \
  --loss dicece \
  --optimizer adam \
  --lr 2.5e-4 \
  --lr-scheduler polynomial \
  --selection-metric dice3d \
  --distance-metrics-at best \
  --epochs 50 \
  --patience 0 \
  --seed 0 \
  --gpu \
  --dest results/segthor/unet_k64_25d_s0
```

Run `python -m ai4mi.training.main --help` for all options. The principal
groups are:

| Purpose | Options |
| --- | --- |
| Dataset/model | `--dataset`, `--net`, `--net_kwargs`, `--adjacent_slices` |
| Optimisation | `--optimizer`, `--lr`, `--weight-decay`, `--lr-scheduler`, `--batch-size` |
| Duration | `--epochs`, `--patience`, `--min_delta`, `--seed` |
| Objective | `--loss`, `--mode`, `--weighted_sampler`, `--augment` |
| Selection | `--selection-metric`, `--selection-aggregation`, `--distance-metrics-at` |

`--adjacent_slices 0` is 2D; a positive value supplies 2n + 1 slices. Dice is
maximised, whereas HD95 and ASSD are minimised. Distance calculations are
expensive: `--distance-metrics-at best` calculates them only at the selected
epoch and cannot be used if HD95 or ASSD selects the checkpoint.

Each run writes `bestweights.pt`, `bestmodel.pkl`, `best_epoch.txt`,
`summary.json`, validation predictions, NumPy metric arrays, and automatic
figures in `plots/`.

## Evaluate and inspect

Training generates its principal report automatically. Plot a specific metric:

```bash
python -m ai4mi.evaluation.plot \
  --metric_file results/segthor/unet_k64_25d_s0/dice_val.npy \
  --dest results/segthor/unet_k64_25d_s0/dice_val.png \
  --headless
```

Compare 2D predictions with the viewer submodule:

```bash
python tools/vendor/viewer/viewer.py \
  --img_source tools/data/SEGTHOR/val/img \
  tools/data/SEGTHOR/val/gt \
  results/segthor/unet_k64_25d_s0/best_epoch/val \
  -n 2 -C 5 \
  --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
  --legend \
  --class_names background esophagus heart trachea aorta
```

The graphical viewer needs X forwarding over SSH. Use
[3D Slicer](https://www.slicer.org/) or [ITK-SNAP](http://www.itksnap.org/)
for NIfTI inspection.

Stitch preprocessed validation predictions into original-space NIfTI volumes:

```bash
python -m ai4mi.inference.stitch \
  --data_folder results/segthor/unet_k64_25d_s0/best_epoch/val \
  --source_scan_pattern "tools/data/segthor_train_full/train/{id_}/GT.nii.gz" \
  --dest_folder volumes/segthor/unet_k64_25d_s0 \
  --grp_regex "(Patient_\d\d)_\d\d\d\d" \
  --num_classes 255 \
  --preprocess_stats tools/data/SEGTHOR/preprocess_stats.pkl \
  --lcc
```

`--lcc` retains the largest component per organ and nearby components, warning
if it removes more than 10% of an organ. Omit `--preprocess_stats` when
stitching `SEGTHOR_ORIGINAL` predictions.

## Experiments and HPC

[`tools/hpc/`](tools/hpc/) contains Slurm templates for architecture, ENet,
optimiser, adjacent-slice, and augmentation/sampling/postprocessing studies.
They are specific to the project HPC environment: review account, Conda
environment, partition, GPU, and time-limit settings before submission.

Experiment reports and reproducible figures are versioned in
[`docs/experiments/`](docs/experiments/):

- [architecture sweep](docs/experiments/arch_sweep/sweep_report.md)
- [ENet sweep](docs/experiments/enet_sweep/enet_report.md)
- [loss sweep](docs/experiments/loss_sweep/loss_report.md)
- [optimiser sweep](docs/experiments/optimiser_sweep/optimiser_report.md)
- [adjacent-slice sweep](docs/experiments/adj_sweep/adj_report.md)
- [augmentation, sampling, and postprocessing](docs/experiments/augmentation_sampler_postprocessing/report.md)

## Development notes

- Project code lives in `src/ai4mi/`; use module entry points rather than
  running source files by path.
- `tools/plotting/` reproduces experiment figures, and checksum files live in
  `tools/data/checksums/`.
- See [`docs/git.md`](docs/git.md) for Git workflow guidance.
- Run `git submodule update --init --recursive` after cloning or switching to a
  revision that changes the viewer submodule.

## License and acknowledgements

Source files retain the MIT license notice from the project baseline. Preserve
applicable notices when reusing or modifying code. SegTHOR access and use remain
subject to the course and dataset terms.
