import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import numpy as np

from datasets import get_dataloader
from model import SimpleCNN


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def main():
    data_dir = "data/raw/chest_xray/test"
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        train=False
    )

    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))

    labels, preds, probs = evaluate(model, test_loader, device)

    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall:", recall_score(labels, preds))
    print("F1 Score:", f1_score(labels, preds))
    print("ROC-AUC:", roc_auc_score(labels, probs))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
