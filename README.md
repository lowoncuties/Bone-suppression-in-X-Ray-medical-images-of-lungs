## GAN-based Bone Suppression Using a Combined Loss Function

This archive contains the source code, trained models, and configuration files  
associated with the accepted publication:

Jochymek L., Vašinková M., Doležíl V., Gajdoš P.  
**GAN-based bone suppression using a combined loss function.**  
(2026)

This package is intended to ensure full transparency and reproducibility of the experiments reported in the article.

---

## 1. Overview

The goal of this study is to evaluate different deep learning paradigms for rib suppression in chest radiographs (CXRs), including:

- Convolutional Autoencoders (AE)  
- U-Net architectures  
- Generative Adversarial Networks (GAN)  

The best-performing configuration is based on a Wasserstein GAN augmented with L1, perceptual, and Sobel loss components.

All experiments described in the publication can be reproduced directly using the provided Jupyter notebooks.

---

## 2. Dataset

This study uses the publicly available JSRT dataset:

**Japanese Society of Radiological Technology (JSRT)**  
Standard Digital Image Database  
http://db.jsrt.or.jp/eng.php

**IMPORTANT:**  
The original JSRT dataset is **NOT redistributed** in this archive due to licensing restrictions. Users must obtain the dataset directly from the official JSRT source.

The experiments were conducted on the augmented JSRT dataset described in Rajaraman et al. (2021), resized to 1024×1024 or 512×512 depending on the model configuration.

### Dataset layout (CLI runs)

For CLI-based runs, place paired images in directories such as:

- `Dataset/source`  
- `Dataset/target`  

File names must match between the source and target folders.

---

## 3. System Requirements

Experiments were conducted using:

- Python 3.6.8  
- TensorFlow 2.6.2  
- Segmentation Models v1.0.1  
- CUDA 11.4  
- NVIDIA Tesla V100 (32GB)

Exact Python dependencies are listed in:

```bash
pip install -r requirements.txt
```

CUDA and compatible NVIDIA drivers are required for GPU training.

**Note:**  
The experiments were originally conducted under Python 3.6.8. Newer Python versions may require minor dependency adjustments.

---

## 4. Repository Structure

- `GAN_BS.ipynb`  
  Jupyter notebook for GAN-based bone suppression  
  (Wasserstein + L1 + Perceptual + Sobel loss)

- `AE_UNET_BS.ipynb`  
  Jupyter notebook for Autoencoder and U-Net experiments  
  (corresponding to Tables 1–4 and 6–8 in the paper)

- `train.py`  
  Training entrypoint (CLI)

- `evaluate.py`  
  Evaluation/report entrypoint (CLI)

- `configs/base.yaml`  
  Default experiment configuration (CLI)

- `src/bonesuppression/`
  - `data/`: paired image loading and dataset splitting
  - `models/`: autoencoder and U-Net builders, losses, model factory
  - `training/`: training/evaluation utilities and artifact plotting
  - `eval/`: final report + methods summary writers
  - `utils/`: reproducibility and run directory utilities

- `experiments/`  
  Archival copy of the original notebooks (`AE_UNET_BS.ipynb`, `GAN_BS.ipynb`)

- `models/`  
  Trained model weights for best-performing configurations

- `configs/`  
  Optional configuration files (if applicable)

- `requirements.txt`  
  Python dependency specification

- `LICENSE`  
  Software license (MIT)

- `CITATION.cff`  
  Citation metadata

---

## 5. Reproducing the Experiments

### Step 1 – Obtain the JSRT dataset

Download from:  
http://db.jsrt.or.jp/eng.php

### Step 2 – Perform preprocessing and resizing

(as described in Section 2 of the paper)

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Run the experiments

#### Option A: Execute the notebooks

- `GAN_BS.ipynb` – reproduces the GAN experiments
- `AE_UNET_BS.ipynb` – reproduces AE and U-Net experiments

Each notebook contains:

- preprocessing steps
- model definition
- training procedure
- evaluation metrics computation

#### Option B: Run training from the CLI

**Config-driven run**

```bash
python train.py --config configs/base.yaml
```

**CLI override run**

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

**Resume from checkpoint**

```bash
python train.py --config configs/base.yaml --resume outputs/ae_mixed_l2/<run_id>/checkpoints/best.keras
```

### Step 5 – Evaluate model performance

Metrics:

- PSNR
- MS-SSIM

Metrics are computed as described in Section 2.5 of the paper.

**CLI evaluation**

```bash
python evaluate.py \
  --config configs/base.yaml \
  --checkpoint outputs/ae_mixed_l2/<run_id>/checkpoints/best.keras \
  --output-dir outputs/ae_mixed_l2/<run_id>/posthoc_eval
```

---

## 6. Trained Models

The archive includes trained model weights corresponding to the  
best-performing configurations reported in Tables 6–8 of the paper.

**Best GAN configuration:**

- Loss: Wasserstein + L1 + Perceptual + Sobel
- Epochs: 750
- Training size: 3980 samples
- Resolution: 512×512×3

---

## 7. Reproducibility Features (CLI)

- Global seed control (`random`, `numpy`, `tensorflow`)
- Deterministic TensorFlow ops enabled where available
- Run-resolved config exported as YAML and JSON
- Metrics and artifacts saved to a versioned run directory:

```text
outputs/<experiment_name>/<timestamp>/
  checkpoints/
  metrics/
  figures/
  predictions/
  reports/
```

### Artifact outputs

Each training run saves:

- Model checkpoints (`best.keras`, `last.keras`)
- Keras epoch logs (`metrics/keras_history.csv`)
- Training curves (`figures/training_curves.png`)
- Predicted sample images (`predictions/*.png`)
- Final evaluation report (`reports/evaluation_report.json`)
- Methods-ready summary (`reports/reproducibility_methods_summary.md`)

### Methodology mapping to paper sections

The generated `reproducibility_methods_summary.md` maps run settings to manuscript sections:

- **Data**: source/target directories and preprocessing resolution
- **Training setup**: model, loss, optimizer, learning rate, epochs, batch size, seed
- **Metrics**: final test metrics (loss, PSNR, MS-SSIM) for reporting tables

---

## 8. Funding

This work was supported by:

- Center for Artificial Intelligence and Quantum Computing in System Brain Research (CLARA), Grant No. 101136607-02
- Research Platform for Digital Transformation and Society 5.0, Grant No. CZ.02.01.01/00/23 021/0012599

---

## 9. Citation

If you use this code or trained models, please cite the associated publication.

---

## 10. License

This software is distributed under the MIT License.  
See the `LICENSE` file for details.

---

## 11. Disclaimer

This work is intended for research purposes only.  
Bone-suppressed radiographs must not be used for clinical decision-making  
without appropriate regulatory validation and clinical assessment.

---

## 12. Legacy notebooks

Original notebooks are retained at root and copied into `experiments/` for archival reproducibility:

- `AE_UNET_BS.ipynb`
- `GAN_BS.ipynb`
