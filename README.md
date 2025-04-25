# Aerial Scene Classification using HOG, Color Histograms, and Ensemble Learning

---

##  Project Summary
- Multi-class classification of aerial scenes (15 categories)
- Combines handcrafted features (HOG + Color Histogram)
- Uses an ensemble of `XGBoost` and `Random Forest` for prediction
- Evaluated via both **70/15/15 split**

---

##  Directory Structure

```
project/
│
├── Submission.ipynb        # Main Jupyter notebook
├── Aerial_Landscapes/          # Dataset folder (15 subfolders, each a class)
├── README.md                   # This file

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

## Features Extracted

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

---

## Models Used

- **XGBoost** with:
  - `gpu_hist` tree method
  - `predictor='gpu_predictor'`
- **Random Forest**
  - `n_estimators=1000`, `max_depth=10`, class-balanced
- Combined using `VotingClassifier` (soft voting)

---

## Evaluation

- **Train/Validation/Test Split**: 70% / 15% / 15% (stratified)
- Metrics reported:
  - Accuracy
  - Classification Report
  - Confusion Matrix (normalized and raw)

---

## Hardware Used

- **GPU**: NVIDIA RTX 5080 (sm_120)
  - Required building PyTorch from source using **nightly version**
  - Used patch to enable `sm_120` (Blackwell architecture) support
- **CPU**: Intel Core i7-14700K
- **RAM**: 32 GB DDR5
- **Environment**: Ubuntu 22.04, Python 3.11

---


Essential libraries:
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `xgboost`
- `matplotlib`, `seaborn`, `tqdm`
- `torch` (built from source with `sm_120` support(nightly version))

---

## Notes

- External libraries are used for feature extraction and modeling.
- Any external tools or modifications (e.g., PyTorch nightly) are listed in this README.
