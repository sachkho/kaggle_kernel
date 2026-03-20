# Kernel Methods for Image Classification

**MVA / IASD Data Challenge** — 10-class image classification on a CIFAR-10 subset using kernel methods only.

> No ML libraries (scikit-learn, libsvm, etc.) are used. Only NumPy, cvxopt, and matplotlib.

---

## Results

| Method | Validation Accuracy |
|--------|-------------------|
| Raw pixels + Kernel PCA | ~18% |
| HOG + KRR + RBF | ~50% |
| HOG + LBP + KRR + RBF | ~57% |
| **HOG + LBP + Color + MKL-KRR + Augmentation** | **~70%** |

---

## Method Overview

The pipeline has five main components:

### 1. Feature Extraction (`features.py`)

Three complementary feature groups are extracted from each image:

- **HOG** (Histogram of Oriented Gradients) — encodes local edge orientations in 4×4 pixel cells with L2-Hys block normalisation. Robust to brightness variations. Computed on the full image and a 2×2 spatial pyramid. Also computed on **opponent color channels** (O1 = R−G, O2 = R+G−2B) to capture colour edges independently of luminance.

- **LBP** (Local Binary Patterns) — for each pixel, compares its intensity to 8 neighbours on a circle and builds a histogram of the resulting binary codes. Captures micro-texture (fur, feathers, scales) that HOG misses. Computed at two scales (radius=1 and radius=2) on the full image and a 2×2 spatial pyramid.

- **Opponent colour histograms** — per-channel intensity histograms in the opponent colour space, which decorrelates colour from luminance. More discriminative than raw RGB histograms on pre-processed dark images.

Each group is normalised independently (zero mean, unit variance) fitted on the training set only.

### 2. Data Augmentation (`augment.py`)

Applied to the **training set only** before feature extraction:
- Horizontal flip (×1)
- Translations of ±2 pixels in 4 directions (×4)

This multiplies the training set by 6 (5 000 → 30 000 images) at no label cost, since all augmentations preserve the class.

### 3. Multiple Kernel Learning (`kernel_svm.py` — `MKLRidgeClassifier`)

Instead of a single kernel on concatenated features, one kernel is applied per feature group with its own bandwidth γ, then the kernels are combined as a weighted sum:

```
K_combined = w_hog · K_hog + w_lbp · K_lbp + w_color · K_color
```

Each γ is set automatically as `1 / (D_group × var_group)`, which avoids the problem of a single γ being simultaneously too large for one group and too small for another.

### 4. Kernel Ridge Regression (KRR)

The classifier solves:

```
α = (K + λI)⁻¹ Y
```

where Y is the one-hot label matrix. Predictions are `argmax(K_test · α)`. Much faster than SVM (single linear solve vs. iterative QP) while achieving comparable accuracy.

**Nyström approximation** is used to make the O(n³) solve tractable on the augmented training set: m anchor points are sub-sampled and a low-rank approximation of K is built, reducing complexity to O(m²·n).

### 5. Hyperparameter Tuning (`cross_validate_mkl.py`)

5-fold cross-validation grid search over:
- λ ∈ {1e-5, 1e-4, 1e-3}
- γ multiplier ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- Kernel type per group: RBF or χ² (chi-squared)
- MKL weights: 4 configurations

Results are saved to `cv_results.csv` with the best configuration printed at the end.

---

## File Structure

```
├── main.py                # Full training pipeline → produces Yte.csv
├── features.py            # HOG, LBP, opponent colour, grouped extraction
├── kernel_svm.py          # Kernel functions, KRR, SVM, MKLRidgeClassifier
├── augment.py             # Horizontal flip + translation augmentation
├── cross_validate_mkl.py  # 5-fold CV grid search over all hyperparameters
└── data/
    ├── Xtr.csv            # 5000 training images (3072 values each)
    ├── Xte.csv            # 2000 test images
    └── Ytr.csv            # Training labels (Id, Prediction)
```

---

## Installation

```bash
pip install numpy cvxopt matplotlib
```

No other dependencies required.

---

## Usage

**1. Train and generate submission:**
```bash
python main.py
# → outputs Yte.csv
```

**2. Run hyperparameter search before final submission:**
```bash
python cross_validate_mkl.py
# → outputs cv_results.csv with best config
```

**Important:** delete the `cache/` folder whenever you modify `features.py`, since extracted features are cached as `.npy` files to avoid recomputation.

```bash
rm -rf cache/
```

---

## Key Design Choices

**Why not raw pixels?** The images are pre-processed and appear very dark. Pixel-level kernels are sensitive to absolute intensities and are not robust to shifts — two images of the same object slightly translated look completely different in pixel space.

**Why separate kernels per feature group?** HOG, LBP, and colour histograms have very different dimensionalities and variances. A single γ cannot be simultaneously optimal for all three. MKL assigns each group its own γ computed as `1/(D·var)`.

**Why KRR instead of SVM?** KRR is 10–50× faster than SVM (one linear solve vs. an iterative QP per class) while achieving comparable accuracy on this dataset. This makes cross-validation feasible.

**Why Nyström?** The augmented training set has ~30 000 samples. An exact KRR solve would require inverting a 30 000×30 000 matrix (27 billion entries). Nyström with m=2000 anchors reduces this to a 2000×2000 solve with minimal accuracy loss.
