import torch

from src.models.model import DRClassifier
from src.utils.config import DEVICE

model = DRClassifier().to(DEVICE)

dummy = torch.randn(8, 3, 512, 512).to(DEVICE)

output = model(dummy)

print(output.shape)