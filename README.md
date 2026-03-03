# Reproducibility Package
## GAN-based Bone Suppression Using a Combined Loss Function

This repository now contains a CLI-first, modular Python implementation for reproducing bone suppression experiments from the original notebooks.

## Project structure

- `train.py`: main training entrypoint.
- `evaluate.py`: standalone evaluation/report entrypoint.
- `configs/base.yaml`: default experiment configuration.
- `src/bonesuppression/`
  - `data/`: paired image loading and dataset splitting.
  - `models/`: autoencoder and U-Net builders, losses, model factory.
  - `training/`: training/evaluation utilities and artifact plotting.
  - `eval/`: final report + methods summary writers.
  - `utils/`: reproducibility and run directory utilities.
- `experiments/`: archival copy of original notebooks (`AE_UNET_BS.ipynb`, `GAN_BS.ipynb`).
- root notebooks are preserved for backward compatibility.

## Dataset

JSRT is not redistributed. Download and place paired images in directories such as:

- `Dataset/source`
- `Dataset/target`

File names must match between source/target folders.

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Config-driven run

```bash
python train.py --config configs/base.yaml
```

### CLI override run

```bash
python train.py \
  --source-dir Dataset/source \
  --target-dir Dataset/target \
  --model unet \
  --loss mixed_l2 \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-3 \
  --seed 42 \
  --experiment-name unet_mixedl2
```

### Resume from checkpoint

```bash
python train.py --config configs/base.yaml --resume outputs/ae_mixed_l2/<run_id>/checkpoints/best.keras
```

## Evaluation

```bash
python evaluate.py \
  --config configs/base.yaml \
  --checkpoint outputs/ae_mixed_l2/<run_id>/checkpoints/best.keras \
  --output-dir outputs/ae_mixed_l2/<run_id>/posthoc_eval
```

## Reproducibility features

- Global seed control (`random`, `numpy`, `tensorflow`).
- Deterministic TensorFlow ops enabled where available.
- Run-resolved config exported as YAML and JSON.
- Metrics and artifacts saved to a versioned run directory:

```
outputs/<experiment_name>/<timestamp>/
  checkpoints/
  metrics/
  figures/
  predictions/
  reports/
```

## Artifact outputs

Each training run saves:

- Model checkpoints (`best.keras`, `last.keras`).
- Keras epoch logs (`metrics/keras_history.csv`).
- Training curves (`figures/training_curves.png`).
- Predicted sample images (`predictions/*.png`).
- Final evaluation report (`reports/evaluation_report.json`).
- Methods-ready summary (`reports/reproducibility_methods_summary.md`).

## Methodology mapping to paper sections

The generated `reproducibility_methods_summary.md` maps run settings to manuscript sections:

- **Data**: source/target directories and preprocessing resolution.
- **Training setup**: model, loss, optimizer, learning rate, epochs, batch size, seed.
- **Metrics**: final test metrics (loss, PSNR, MS-SSIM) for reporting tables.

## Legacy notebooks

Original notebooks are retained at root and copied into `experiments/` for archival reproducibility:

- `AE_UNET_BS.ipynb`
- `GAN_BS.ipynb`

These notebooks are no longer the primary execution path.
