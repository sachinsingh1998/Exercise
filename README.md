# COMP9517 Group Project – Aerial Image Classification

## Overview

This project focuses on classifying aerial images into 15 distinct landscape categories using a combination of traditional machine learning pipelines and modern deep learning architectures. The SkyView dataset from Kaggle was used to explore and compare performance across various models and techniques.

---

## Dataset

- **Source**: [SkyView Dataset on Kaggle](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- **Classes**: 15 categories (800 images each)
- **Splits**:
  - 70% Training / 15% Validation / 15% Test (stratified)
  - Stratified K-Fold used for traditional ML cross-validation

---

## Directory Structure

```
project/
│
├── Submission.ipynb             # Consolidated jupyter notebook for all methods
├── Aerial_Landscapes/              # Dataset folder with 15 subfolders
├── README.md                       # This file
```

**Important**: The dataset folder must be named `Aerial_Landscapes` and placed in the same directory as the notebook.

---
## Methods

### 1.Traditional Machine Learning

- Combines handcrafted features (HOG + Color Histogram)
- Uses an ensemble of `XGBoost` and `Random Forest` for prediction
- Evaluated via both **70/15/15 split**


#### Features Extracted

- **Color Histogram**
  - 8×8×8 bins on RGB channels, normalized and flattened
  - Image resized to 512×512

- **HOG (Histogram of Oriented Gradients)**
  - Grayscale image resized to 128×128
  - `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `orientations=9`

- **PCA for HOG**
  - Dimensionality reduction to 1000 components

- **Feature Fusion**
  - Combined: `[PCA-HOG || Color Histogram]`
  - Standardized with `StandardScaler`

### Models Used

- **XGBoost**
  - `gpu_hist`, `gpu_predictor`
- **Random Forest**
  - 1000 trees, `max_depth=10`, class-balanced
- **Voting Classifier**
  - Soft voting ensemble of RF + XGBoost

### Evaluation

- Stratified 70/15/15 split
- Accuracy, Classification Report, and Confusion Matrix (raw & normalized)

---

## Deep Learning Models

### 1. VGG-16 and ResNet-50 (Transfer Learning & Explainable AI)

- **Backbones**: Pretrained on ImageNet
- **Modifications**: Final FC layer changed for 15 classes
- **Training**: First few layers frozen; deeper layers fine-tuned
- **Loss**: CrossEntropyLoss
- **XAI**: Class Activation Maps generated from final convolutional layers
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix

### 2. EfficientNetV2-S

- **Transformations**: Augmented with flips, jitter, normalization
- **Class Imbalance**: Handled using `WeightedRandomSampler`
- **Optimizations**: Early stopping and LR scheduling
- **Metrics**: Accuracy, F1-Score, Precision, Recall

### 3. DenseNet-121

- **Structure**: 3-way split (train/val/test)
- **Performance**: Excellent on fine-grained features like texture
- **Metrics**: Accuracy, Classification Report, Confusion Matrix

### 4. EfficientNet-B0

- **Strengths**: Lightweight, fast convergence
- **Setup**: Folder-based train/val/test structure
- **Metrics**: Accuracy and per-class F1-score

---

## Hardware Used

- **GPU**: NVIDIA RTX 5080 (Blackwell, sm_120)
- **CPU**: Intel Core i7-14700K
- **RAM**: 32 GB DDR5
- **Environment**: Ubuntu 22.04, Python 3.11
- **Note**: PyTorch built from source with nightly version for `sm_120` GPU support

---

## Installation

Install the required packages:

```
pip install torch torchvision thundersvm opencv-python scikit-learn scikit-image xgboost matplotlib seaborn tqdm pandas PIL torchinfo
```

---

## Usage

1. Ensure dataset is placed inside a folder named `Aerial_Landscapes`.
2. Run the notebook `Project_final.ipynb` from start to end.
3. Adjust device setting if necessary:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Contributors

- 

---

## Acknowledgements

- SkyView dataset from Kaggle  
- PyTorch & TorchVision pretrained models  
- ThunderSVM (GPU-accelerated SVM)  
- Kaggle community reference for data loading: https://www.kaggle.com/code/bryamblasrimac/skyview-eda-vit-accuracy-96-33  
- scikit-learn, OpenCV, and XGBoost libraries  
- Special thanks to the UNSW COMP9517 teaching team
