from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset

from .preprocessing import crop_black_border, apply_clahe


class RetinopathyDataset(Dataset):
    """
    PyTorch Dataset for the APTOS 2019 Diabetic Retinopathy dataset.
    """

    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir / f"{row['id_code']}.png"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"OpenCV failed to load image:\n{image_path}"
            )

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --------------------------
        # Medical Image Preprocessing
        # --------------------------

        image = crop_black_border(image)

        image = apply_clahe(image)

        # --------------------------

        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label