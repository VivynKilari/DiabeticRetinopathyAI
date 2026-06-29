import cv2
import numpy as np


def crop_black_border(image, threshold=10):
    """
    Remove black borders around retinal fundus images.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mask = gray > threshold

    if mask.any():
        coords = np.argwhere(mask)

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        image = image[y0:y1, x0:x1]

    return image


def apply_clahe(image):
    """
    Improve local contrast using CLAHE.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image