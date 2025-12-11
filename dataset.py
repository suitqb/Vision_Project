# dataset.py
import os
from torchvision import datasets, transforms
import torch

def get_anime_dataloader(data_root="data/anime_faces", image_size=64,
                           batch_size=32, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)   # [-1, 1]
    ])

    ds = datasets.ImageFolder(root=data_root, transform=transform)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dl, len(ds.classes)
