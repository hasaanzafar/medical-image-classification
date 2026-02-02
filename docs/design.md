# Design Decisions and Rationale

This document records the major design choices made in this project, along with the reasoning behind each decision.

The goal is not only performance, but clarity, reproducibility, and robustness in a medical imaging context.

---

## Dataset Choice

**Selected dataset:** Chest X-ray Pneumonia (Kaggle)

**Rationale:**
- Public and widely studied medical imaging dataset
- Binary classification simplifies error analysis
- Class imbalance reflects real-world medical data
- Folder-based structure integrates well with PyTorch `ImageFolder`

---

## Image Resolution

**Chosen resolution:** 224 × 224

**Rationale:**
- Compatible with standard CNN architectures
- Enables transfer learning if needed
- Balances spatial detail with computational efficiency

---

## Data Augmentation Strategy

**Applied augmentations:**
- Random horizontal flipping (training only)

**Rationale:**
- Improves generalization
- Preserves medical realism
- Avoids unrealistic transformations (e.g., rotations, color jitter)

---

## Normalization

**Statistics used:** ImageNet mean and standard deviation

**Rationale:**
- Ensures compatibility with pretrained CNN backbones
- Standard practice in medical imaging transfer learning

---

## Data Pipeline Design

**Key choices:**
- Separate transforms for training and evaluation
- Script-based dataset loading
- No preprocessing stored in Git

**Rationale:**
- Prevents data leakage
- Guarantees reproducibility
- Keeps repository lightweight

---

## Evaluation Metrics

**Primary metrics:**
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

**Rationale:**
- Accuracy alone is insufficient for imbalanced medical datasets
- Recall is especially important in clinical contexts

---

## Reproducibility Measures

- Fixed random seeds
- Deterministic dataloading where possible
- Explicit dependency versions
  

---
---

## Model Architecture

**Chosen model:** Custom lightweight CNN

**Rationale:**
- Medical imaging datasets are often small
- Shallow architectures reduce overfitting
- Explicit architecture improves interpretability

**Key design choices:**
- Three convolutional blocks with increasing channel depth
- Batch normalization after each convolution
- Global average pooling instead of fully connected layers

