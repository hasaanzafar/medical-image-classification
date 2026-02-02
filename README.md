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

Detailed quantitative and qualitative analysis provided in notebooks and figures.

## Design Decisions

Detailed architectural and experimental design choices are documented in  
[`docs/design.md`](docs/design.md).

