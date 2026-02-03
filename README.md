# Medical Image Classification

This repository contains an end-to-end deep learning pipeline for medical image classification using convolutional neural networks (CNNs), covering **model training, evaluation, and deployment as a containerized inference service**.

The project emphasizes **robust evaluation, reproducibility, and practical deployment readiness**, rather than optimizing for accuracy alone.


## Motivation

Medical imaging datasets are often class-imbalanced, and false negatives can carry high real-world cost.
This project explores how architectural choices, transfer learning, and evaluation metrics impact performance in a medical screening context, with a deliberate focus on **high recall for critical classes**.


## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Due to dataset size, raw images are **not included** in this repository.

<img width="128" height="115" alt="image" src="https://github.com/user-attachments/assets/b29d9062-2d84-4c5e-a3bc-23e3ec7df4be" />

## Approach

- Image preprocessing and normalization
- CNN-based image classification using PyTorch
- Transfer learning with a pretrained ResNet-18 backbone
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Deployment of the trained model as a RESTful inference service using FastAPI


## Reproducibility

- Script-driven training and evaluation
- Environment-aware dataset paths (local or Kaggle)
- Fixed model architecture and evaluation protocol
- Containerized deployment using Docker for reproducible inference


## Repository Structure
<img width="239" height="448" alt="image" src="https://github.com/user-attachments/assets/f8a21ce7-bd28-4075-875a-54ba33b42e1c" />


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

When using Kaggle, the dataset is automatically mounted at:

/kaggle/input/chest-xray-pneumonia/chest_xray

## Model Deployment (FastAPI + Docker)

The trained model is deployed as a lightweight RESTful inference service using FastAPI and Docker. The service loads a frozen trained PyTorch model, applies the same preprocessing pipeline used during training, and exposes an HTTP endpoint for image-based predictions.

This deployment is designed to demonstrate deployment readiness, reproducibility, and clean separation between training and inference

## Local Deployment (Docker)

Build the Docker image:
-docker build -t medical-image-classifier .


Run the inference service:
-docker run -p 8000:8000 medical-image-classifier


Once running, the API documentation is available at: 
- http://localhost:8000/docs

## Example API Usage:

- Endpoint: POST /predict

- Input: Chest X-ray image file

- Output

{
  "prediction": "PNEUMONIA",
  "confidence": 0.93
}



This deployment demonstrates:

- Separation of training and inference code
- Consistent preprocessing across environments
- Model inference exposed via a REST API
- Reproducible deployment using Docker

Note: This deployment is intended as a demonstration of deployment readiness and is not a production medical system.
