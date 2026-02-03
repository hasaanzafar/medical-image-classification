# Medical Image Classification

This repository contains an end-to-end deep learning pipeline for medical image classification using convolutional neural networks (CNNs).

The project focuses on model design, evaluation, and reproducibility rather than just achieving high accuracy.

## Motivation

Medical imaging datasets are often limited in size and class-imbalanced.
This project explores how architectural choices, data augmentation, and evaluation metrics affect classification performance.

## Dataset

The pipeline supports publicly available medical imaging datasets (e.g., chest X-ray, histopathology, or MRI datasets).
Raw images are not included due to size constraints.

## Approach

- Data preprocessing and normalization
- CNN-based image classification using PyTorch
- Experiment tracking with MLflow
- Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices

## Reproducibility

- Fixed random seeds
- Script-driven training and evaluation
- Dockerized environment (optional)

## Repository Structure

(Structure diagram here)


## Results

The model was trained on the full Chest X-Ray Pneumonia dataset using GPU acceleration.
Evaluation was performed on a held-out test set.

- Accuracy: 83%
- Precision (Pneumonia): 80%
- Recall (Pneumonia): 98%
- F1-score (Pneumonia): 88%
- ROC-AUC: 0.95

High recall was prioritized to minimize false negatives, which is critical in medical screening tasks.


## Design Decisions

Detailed architectural and experimental design choices are documented in  
[`docs/design.md`](docs/design.md).

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Due to size constraints, raw images are not included in this repository.

To reproduce results:
1. Download the dataset from Kaggle
2. Extract it into `data/raw/chest_xray/`
3. Run `train.py` and `evaluate.py`

