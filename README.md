# Reproducibility Package  
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

---

## 3. System Requirements

Experiments were conducted using:

- Python 3.6.8  
- TensorFlow 2.6.2  
- Segmentation Models v1.0.1  
- CUDA 11.4  
- NVIDIA Tesla V100 (32GB)

Exact Python dependencies are listed in:
pip install -r requirements.txt


CUDA and compatible NVIDIA drivers are required for GPU training.

**Note:**  
The experiments were originally conducted under Python 3.6.8. Newer Python versions may require minor dependency adjustments.

---
## 4. Repository Structure

- **GAN_BS.ipynb**  
  Jupyter notebook for GAN-based bone suppression  
  (Wasserstein + L1 + Perceptual + Sobel loss)

- **AE_UNET_BS.ipynb**  
  Jupyter notebook for Autoencoder and U-Net experiments  
  (corresponding to Tables 1–4 and 6–8 in the paper)

- **models/**  
  Trained model weights for best-performing configurations

- **configs/**  
  Optional configuration files (if applicable)

- **requirements.txt**  
  Python dependency specification

- **LICENSE**  
  Software license (MIT)

- **CITATION.cff**  
  Citation metadata


---

## 5. Reproducing the Experiments

### Step 1 – Obtain the JSRT dataset

Download from:  
http://db.jsrt.or.jp/eng.php  

### Step 2 – Perform preprocessing and resizing  
(as described in Section 2 of the paper)

### Step 3 – Install dependencies
pip install -r requirements.txt


### Step 4 – Open and execute the notebooks

- `GAN_BS.ipynb` – reproduces the GAN experiments  
- `AE_UNET_BS.ipynb` – reproduces AE and U-Net experiments  

Each notebook contains:

- preprocessing steps  
- model definition  
- training procedure  
- evaluation metrics computation  

### Step 5 – Evaluate model performance

Metrics:

- PSNR  
- MS-SSIM  

Metrics are computed as described in Section 2.5 of the paper.

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

## 7. Funding

This work was supported by:

- Center for Artificial Intelligence and Quantum Computing in System Brain Research (CLARA), Grant No. 101136607-02  
- Research Platform for Digital Transformation and Society 5.0, Grant No. CZ.02.01.01/00/23 021/0012599  

---

## 8. Citation

If you use this code or trained models, please cite the associated publication.

---

## 9. License

This software is distributed under the MIT License.  
See the `LICENSE` file for details.

---

## 10. Disclaimer

This work is intended for research purposes only.  
Bone-suppressed radiographs must not be used for clinical decision-making  
without appropriate regulatory validation and clinical assessment.

