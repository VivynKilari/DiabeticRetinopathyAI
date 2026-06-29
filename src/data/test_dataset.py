from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import RetinopathyDataset
from src.data.transforms import train_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = PROJECT_ROOT / "datasets"

dataset = RetinopathyDataset(
    csv_file=DATASET / "train_1.csv",
    image_dir=DATASET / "train_images" / "train_images",
    transform=train_transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(loader))

print("Image Batch Shape :", images.shape)
print("Labels :", labels)