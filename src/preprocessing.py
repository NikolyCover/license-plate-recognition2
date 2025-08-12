import cv2
import numpy as np

from typing import List, Tuple

from config import CHAR_SIZE


def process_plate(plate_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cinza -> Otsu INV -> erosão leve. Retorna (eroded, gray, thresh_inv)."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    eroded = cv2.erode(thresh_inv, np.ones((3, 3), np.uint8), iterations=1)

    return eroded, gray, thresh_inv


def standardize_char(
    char_img: np.ndarray, size: Tuple[int, int] = CHAR_SIZE
) -> np.ndarray:
    """Centraliza o caractere em canvas fixo (preserva proporção)."""
    h, w = char_img.shape

    if h == 0 or w == 0:
        return np.zeros(size[::-1], dtype=np.uint8)

    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(char_img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(size[::-1], dtype=np.uint8)

    x0 = (size[0] - nw) // 2
    y0 = (size[1] - nh) // 2

    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    return canvas


def segment_characters(
    binary_image: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Segmenta contornos com h>25 e w>10; retorna (chars_std, caixas)."""
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    items = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h > 25 and w > 10:
            std = standardize_char(binary_image[y : y + h, x : x + w])

            items.append((x, std, (x, y, w, h)))

    items.sort(key=lambda t: t[0])

    return [t[1] for t in items], [t[2] for t in items]
