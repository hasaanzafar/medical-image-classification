# Medical Image Classification

This repository contains an end-to-end deep learning pipeline for medical image classification using convolutional neural networks (CNNs), including **model training, evaluation, and deployment**.

The project emphasizes **robust evaluation, reproducibility, and practical deployment considerations**, rather than optimizing for accuracy alone.


## Motivation

Medical imaging datasets are often class-imbalanced, and false negatives can carry high real-world cost.
This project explores how architectural choices, transfer learning, and evaluation metrics impact performance in a medical screening context, with an emphasis on **high recall for critical classes**.


## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Due to dataset size, raw images are **not included** in this repository.

Expected directory structure:

chest_xray/
├── train/
├── val/
└── test/

## Approach

- Image preprocessing and normalization
- CNN-based classification using PyTorch
- Transfer learning with a pretrained ResNet-18 backbone
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Model deployment using FastAPI for inference

## Reproducibility

- Script-driven training and evaluation
- Environment-aware dataset paths (local or Kaggle)
- Fixed model architecture and evaluation protocol
- Containerized deployment using Docker

## Repository Structure

medical-image-classification/
├── README.md
├── src/
│ ├── datasets.py
│ ├── model.py
│ ├── train.py
│ └── evaluate.py
├── app/
│ ├── main.py
│ └── requirements.txt
├── docs/
│ └── design.md
├── figures/
│ ├── confusion_matrix.png
│ └── roc_curve.png
└── Dockerfile


## Results

The model was trained on the full Chest X-Ray Pneumonia dataset using GPU acceleration.
Evaluation was performed on a held-out test set.

- Accuracy: 83%
- Precision (Pneumonia): 80%
- Recall (Pneumonia): 98%
- F1-score (Pneumonia): 88%
- ROC-AUC: 0.95

High recall was intentionally prioritized to minimize false negatives, which is critical in medical screening applications.


## Design Decisions

Detailed architectural and experimental design choices are documented in  
[`docs/design.md`](docs/design.md).


## Running the Project

The dataset is automatically mounted at:
/kaggle/input/chest-xray-pneumonia/chest_xray

Run:
```bash
python src/train.py
python src/evaluate.py


