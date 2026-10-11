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
  spacing resampling, fixed field-of-view crop/pad handling.
- **Architectures:** U-Net, and Swin-UNet.
- **Support for 2.5D input**: adjacent slices can be passed to the network. Architecture-agnostic. Prediction is performed only for the central slice.
- **Training controls:** configurable architecture settings through CLI, seed and batch
  size, Adam/AdamW/Nesterov SGD, polynomial scheduling, early stopping,
  weighted foreground sampling.  

- **Data augmentation:** optional training-only random affine transforms, intensity scaling, and Gaussian noise, applied consistently across adjacent slices and the mask.
- **Losses and selection:** CE, Dice, DiceCE, and NSW DiceCE; model selection
  by 2D Dice, 3D Dice (organ mean or NSW), HD95, or ASSD.
- **Volume evaluation:** per-patient/per-organ 3D Dice, HD95, HD, and ASSD,
  plus automatic plots, CSV/JSON summaries, and elapsed-time recording.
- **Inference:** original-space NIfTI reconstruction and optional per-organ
  connected-component postprocessing.
- **Final model and test inference:** training on all 40 patients without a
  validation split, prediction on the unlabelled test scans, and validation
  metrics in the submission format.
- **Reproducible experiments:** Slurm templates, plotting scripts, and reports
  for architecture, ENet, loss, optimiser, 2.5D, augmentation,
  sampling, and postprocessing studies.

Detailed methodology and results are in [`docs/experiments/`](docs/experiments/).



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

The latter command uses a deterministic 30/10 train/validation
split, 256 × 256 in-plane shape, and seed 0. It creates:

- `tools/data/SEGTHOR/` — preprocessed slices and `preprocess_stats.pkl`, required
  to restore predictions to their original geometry.
- `tools/data/SEGTHOR_ORIGINAL/` — slices retaining original geometry.

Inspect all preprocessing options with:

```bash
python -m ai4mi.preprocessing.slice_segthor --help
```


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

This representative 2.5D U-Net run selects by mean 3D Dice:

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
  --augment \
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
| Duration | `--epochs`, `--patience`, `--min_delta`, `--seed`, `--no-val` |
| Objective | `--loss`, `--mode`, `--weighted_sampler`, `--augment` |
| Selection | `--selection-metric`, `--selection-aggregation`, `--distance-metrics-at` |

`--adjacent_slices 0` is 2D; a positive value supplies 2n + 1 slices. For`selection-metric` Dice is
maximised, whereas HD95 and ASSD are minimised. Distance calculations are
expensive: `--distance-metrics-at best` calculates them only at  the end, for the best
epoch, and cannot be used if HD95 or ASSD selects the checkpoint.

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

## Final model and test predictions

The final model reuses the best configuration from the
[best-config evaluation](docs/experiments/best_config_eval/best_config_report.md)
but trains on all 40 patients. With no validation set, it trains for a fixed 50
epochs and keeps the last-epoch weights.


Slice all 40 patients into the training set (`--retains 0`). If the test data is present at tools/data/segthor_train_full/test/Patient_XX.nii.gz, it is sliced into tools/data/SEGTHOR_val0/test/ with the training set's preprocessing statistics:
```bash
python -m ai4mi.preprocessing.slice_segthor \
  --source_dir tools/data/segthor_train_full \
  --dest_dir tools/data/SEGTHOR_val0 \
  --shape 256 256 \
  --retains 0
```

To add the test scans to an existing `SEGTHOR_val0` later, rerun the same command
with `--test_only`. That reuses its `preprocess_stats.pkl`, so the test slices
match the training preprocessing.

Train with `--no-val`, which skips validation, ignores `--patience` and
`--selection-metric`, and writes `lastmodel.pkl` and `lastweights.pt`:

```bash
python -m ai4mi.training.main \
  --dataset SEGTHOR_val0 \
  --no-val \
  --net unet \
  --net_kwargs '{"kernels": 64}' \
  --adjacent_slices 1 \
  --mode full \
  --loss dicece \
  --optimizer adam \
  --lr 2.5e-4 \
  --weight-decay 1e-4 \
  --lr-scheduler polynomial \
  --augment \
  --epochs 50 \
  --seed 0 \
  --gpu \
  --dest results/segthor/final/unet_k64_s0
```

Predict the test slices (no ground truth), then stitch and postprocess them
into original-space volumes:

```bash
python -m ai4mi.inference.predict \
  --model results/segthor/final/unet_k64_s0/lastmodel.pkl \
  --data_root tools/data/SEGTHOR_val0 \
  --subset test \
  --adjacent_slices 1 \
  --dest results/segthor/final/unet_k64_s0/test \
  --gpu

python -m ai4mi.inference.stitch \
  --data_folder results/segthor/final/unet_k64_s0/test \
  --source_scan_pattern "tools/data/segthor_train_full/test/{id_}.nii.gz" \
  --dest_folder volumes/segthor/final/unet_k64_s0 \
  --grp_regex "(Patient_\d\d)_\d\d\d\d" \
  --num_classes 255 \
  --preprocess_stats tools/data/SEGTHOR_val0/preprocess_stats.pkl \
  --lcc \
  --lcc_max_dist 50
```

`--adjacent_slices` must match the value the model was trained with.

Score stitched volumes that have ground truth (here, the validation patients)
with 3D Dice, HD95, HD and ASSD per patient and organ:

```bash
python -m ai4mi.evaluation.evaluate_volumes \
  --pred_folder volumes/segthor/unet_k64_25d_s0 \
  --gt_pattern "tools/data/segthor_train_full/train/{id_}/GT.nii.gz" \
  --dest results/segthor/unet_k64_25d_s0/eval \
  --npz_dir results/segthor/unet_k64_25d_s0/eval
```

This writes `per_patient.csv` and `summary.json`. With `--npz_dir`, it also
writes `dice.npz`, `hd95.npz`, `hd.npz` and `assd.npz`; each maps a patient ID to
a per-class array, background included.

[`tools/hpc/final_best_k64.job`](tools/hpc/final_best_k64.job) runs the training
step. [`tools/hpc/submission.job`](tools/hpc/submission.job) runs prediction,
stitching and evaluation, and assembles the submission folder.

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
- [best-config evaluation](docs/experiments/best_config_eval/best_config_report.md)

## Development notes

- Project code lives in `src/ai4mi/`; use module entry points rather than
  running source files by path.
- `tools/plotting/` reproduces experiment figures, and checksum files live in
  `tools/data/checksums/`.
-  [`docs/git.md`](docs/git.md) contains  Git workflow guidance as given by the course.
- Run `git submodule update --init --recursive` after cloning or switching to a
  revision that changes the viewer submodule.

## License and acknowledgements

Source files retain the MIT license notice from the project baseline. Preserve
applicable notices when reusing or modifying code. SegTHOR access and use remain
subject to the course and dataset terms.
