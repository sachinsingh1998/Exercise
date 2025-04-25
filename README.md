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

---

##  Dataset Location & Access

- Dataset folder **must be named `Aerial_Landscapes`** and placed in the **same directory** as the notebook.
- The code accesses the images using:
  ```python
  path = "Aerial_Landscapes"
  os.path.join(path, category)
  ```
- All category folders should contain `.jpg` images only.
- Folder names should exactly match the class labels:
  ```
  ['Agriculture', 'Airport', 'Beach', 'City', 'Desert',
   'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
   'Parking', 'Port', 'Railway', 'Residential', 'River']
  ```

---

---
## Methods

### 1.Traditional Machine Learning

- Combines handcrafted features (HOG + Color Histogram)
- Uses an ensemble of `XGBoost` and `Random Forest` for prediction
- Evaluated via both **70/15/15 split**


#### Features Extracted

- **Color Histogram**
  - Image resized to 512×512
  - 8×8×8 bins for RGB channels
  - Normalized and flattened
- **HOG (Histogram of Oriented Gradients)**
  - Image resized to 128×128 and converted to grayscale
  - `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `orientations=9`
- **Dimensionality Reduction**
  - PCA applied to HOG features (`n_components=1000`)
- **Feature Fusion**
  - Final feature vector = `[PCA-HOG || Histogram]` concatenated
  - Standardized using `StandardScaler`

### Models Used

- **XGBoost** with:
  - `gpu_hist` tree method
  - `predictor='gpu_predictor'`
- **Random Forest**
  - `n_estimators=1000`, `max_depth=10`, class-balanced
- Combined using `VotingClassifier` (soft voting)

### Evaluation

- **Train/Validation/Test Split**: 70% / 15% / 15% (stratified)
- Metrics reported:
  - Accuracy
  - Classification Report
  - Confusion Matrix (normalized and raw)

---

## Deep Learning Models

### 1. VGG-16 and ResNet-50 (Transfer Learning & Explainable AI)

- **Backbones**: VGG-16 and ResNet-50 from `torchvision.models`
- **Weights**: Pretrained on ImageNet  
- **Training**: Frozen first few layers and trained the later Convolutional layers and FC layers
- **Loss**: CrossEntropyLoss
- **Metrics**: Accuracy, Confusion Matrix, Macro Averaged Precision, Recall and F1 scores
- **XAI**: Capture Activated Features from last Convolutional layer and display the Class Activation Map.

### 2. EfficientNetV2 (with data augmentation and class imbalance handling)

- **Transformations**: Augmented with flips, jitter, normalization
- **Backbones**: EfficientNetV2 from `torchvision.models`
- **Weights**: Pretrained on ImageNet  
- **Training**: Replacing the final output layer with a Linear layer mapped to the number of output classes (15).
- **Class Imbalance**: Handled using `WeightedRandomSampler`according to the class weight ensuring that underrepresented classes are      equally likely to be seen during training.
- **Loss**: CrossEntropyLoss
- **Optimizations**: Early stopping and LR scheduling
- **Metrics**: Accuracy, Confusion Matrix, Macro Averaged Precision, Recall and F1 scores
- 
## 3. DenseNet-121 (Three-Way Split)
- Architecture: DenseNet-121, known for its dense connections that promote feature reuse and stronger gradient flow.
- Used pretrained weights from ImageNet and replaced the classifier for 15 classes.
- Initial training with frozen base layers, followed by fine-tuning.
- Weighted loss function applied to account for minor class imbalance.
- Achieved high classification performance, especially on texture-rich classes like desert, forest, and grass.
- Confusion matrix and classification report included in the report.

## 4. EfficientNet-B0 (Three-Way Split)
- Architecture: EfficientNet-B0, which uses compound scaling to balance network depth, width, and resolution.
- Fine-tuned using the same train/val/test split and preprocessing as DenseNet.
- Optimized for performance with low parameter count (~5M).
- Reached test accuracy above 98% and trained faster than DenseNet.
- Strong performance on clearly distinct categories such as airport and runway.
- Evaluation included confusion matrix and full classification metrics.


---

---

## Evaluation

- Evaluation performed on held-out test sets
- Plotted confusion matrices and metric comparisons
- Qualitative visualizations for each model's performance
- Metrics used across models:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix

---


## Hardware Used

- **GPU**: NVIDIA RTX 5080 (Blackwell, sm_120)
- **CPU**: Intel Core i7-14700K
- **RAM**: 32 GB DDR5
- **Environment**: Ubuntu 24.04.2, Python 3.11
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

- Aryan Tiwari  
- Soundhar Krishnamoorthy
- Srividhya Parthasarathy
- Shayan Ziaei
- Sachin Singh

---

## Acknowledgements

- SkyView dataset from Kaggle  
- Pretrained weights PyTorch  
- Scikit-learn and OpenCV libraries for traditional ML 
- Kaggle Code for Custom DataLoading into Pandas Dataframe https://www.kaggle.com/code/bryamblasrimac/skyview-eda-vit-accuracy-96-33
- Special thanks to the UNSW COMP9517 teaching team
