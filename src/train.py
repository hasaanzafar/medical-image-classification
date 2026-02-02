import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from datasets import get_dataloader
from model import SimpleCNN


def set_seed(seed=42):
    """
    Ensures reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    # Configuration
    data_dir = "data/raw/chest_xray/train"
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        train=True
    )

    model = SimpleCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
