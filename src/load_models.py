import os
import cv2
import numpy as np

from typing import Dict, List

from preprocessing import standardize_char
from config import ALLOWED_MODEL_SUFFIXES


def load_models(model_dir: str) -> Dict[str, List[np.ndarray]]:
    """Lê modelos <LABEL>_<SUFIXO>.*; binariza INV e padroniza; retorna dict[label]=[imgs]."""
    models: Dict[str, List[np.ndarray]] = {}
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for fn in os.listdir(model_dir):
        if not fn.lower().endswith(valid_ext):
            continue

        stem = os.path.splitext(fn)[0]
        parts = stem.split("_", 1)

        if len(parts) != 2 or parts[1] not in ALLOWED_MODEL_SUFFIXES:
            continue

        label = parts[0].upper()
        img = cv2.imread(os.path.join(model_dir, fn), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        _, bin_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        models.setdefault(label, []).append(standardize_char(bin_inv))

    return models
