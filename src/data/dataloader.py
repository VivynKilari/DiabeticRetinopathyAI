from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import RetinopathyDataset
from src.data.transforms import train_transform, val_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = PROJECT_ROOT / "datasets"


def create_dataloaders(batch_size=8):

    train_dataset = RetinopathyDataset(
        csv_file=DATASET / "train_1.csv",
        image_dir=DATASET / "train_images" / "train_images",
        transform=train_transform
    )

    val_dataset = RetinopathyDataset(
        csv_file=DATASET / "valid.csv",
        image_dir=DATASET / "val_images" / "val_images",
        transform=val_transform
    )

    train_df = pd.read_csv(DATASET / "train_1.csv")

    class_counts = train_df["diagnosis"].value_counts().sort_index()

    class_weights = 1.0 / class_counts

    sample_weights = train_df["diagnosis"].map(class_weights).to_numpy(copy=True)
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader