import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(train=True):
    """
    Returns image transformations for training or evaluation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloader(data_dir, batch_size=32, train=True):
    """
    Creates a DataLoader for medical image classification.
    """
    transform = get_transforms(train=train)

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )

    return loader
