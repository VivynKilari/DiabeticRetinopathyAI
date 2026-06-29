from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from preprocessing import crop_black_border, apply_clahe

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = PROJECT_ROOT / "datasets"

image_path = DATASET / "train_images" / "train_images" / "1ae8c165fd53.png"

image = cv2.imread(str(image_path))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cropped = crop_black_border(image)

enhanced = apply_clahe(cropped)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cropped)
plt.title("Border Removed")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(enhanced)
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()

plt.show()