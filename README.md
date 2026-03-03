# Reproducibility Package — GAN-based Bone Suppression Using a Combined Loss Function

This repository contains the source code, trained models, and configuration files for:

Jochymek L., Vašinková M., Doležíl V., Gajdoš P.  
**GAN-based bone suppression using a combined loss function.** (2026)

The goal is to enable full reproducibility of the experiments reported in the paper.

## What this repository reproduces

The experiments compare rib suppression approaches for chest radiographs (CXRs):

- Convolutional Autoencoders (AE)
- U-Net architectures
- Generative Adversarial Networks (GAN)

The best-performing model is a Wasserstein GAN trained with a combined loss:

- Wasserstein + L1 + Perceptual + Sobel

## Step-by-step reproduction

### 1) Obtain the dataset

This work uses the publicly available JSRT dataset:

**Japanese Society of Radiological Technology (JSRT) — Standard Digital Image Database**  
http://db.jsrt.or.jp/eng.php

**Note:** The JSRT dataset is not included in this repository due to licensing restrictions.

### 2) Preprocess the data

Apply the preprocessing and resizing described in the paper (Section 2).
Images are resized to 1024×1024 or 512×512, depending on the model configuration.

### 3) Set up the environment

The original experiments used:

- Python 3.6.8
- TensorFlow 2.6.2
- Segmentation Models v1.0.1
- CUDA 11.4
- NVIDIA Tesla V100 (32GB)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

GPU training requires CUDA and compatible NVIDIA drivers. Newer Python versions may require dependency adjustments.

### 4) Run the experiments

Open and execute the provided Jupyter notebooks:

- `GAN_BS.ipynb` — GAN experiments (Wasserstein + L1 + Perceptual + Sobel)
- `AE_UNET_BS.ipynb` — AE and U-Net experiments (paper tables referenced inside the notebook)

Each notebook includes preprocessing, model definitions, training, and evaluation.

### 5) Evaluate results

Metrics reported:

- PSNR
- MS-SSIM

Metrics are computed as described in the paper (Section 2.5).

## Trained models included

The `models/` directory contains trained weights for the best-performing configurations reported in the paper.

Best GAN configuration:

- Loss: Wasserstein + L1 + Perceptual + Sobel
- Epochs: 750
- Training size: 3980 samples
- Resolution: 512×512×3

## Repository contents

- `GAN_BS.ipynb` — GAN workflow
- `AE_UNET_BS.ipynb` — AE/U-Net workflow
- `models/` — trained weights
- `configs/` — configuration files (if applicable)
- `requirements.txt` — dependencies
- `LICENSE` — MIT license
- `CITATION.cff` — citation metadata

## Funding

Supported by:

- Center for Artificial Intelligence and Quantum Computing in System Brain Research (CLARA), Grant No. 101136607-02
- Research Platform for Digital Transformation and Society 5.0, Grant No. CZ.02.01.01/00/23 021/0012599

## Citation

If you use this repository (code or weights), please cite the associated publication.

## License

MIT License — see `LICENSE`.

## Disclaimer

For research use only. Bone-suppressed radiographs must not be used for clinical decision-making without appropriate regulatory validation and clinical assessment.
