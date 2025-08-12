import cv2
import numpy as np
from config import CHAR_SIZE2
from typing import List, Tuple, Optional

def calculate_character_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
    contours1: List[np.ndarray],
    contours2: List[np.ndarray],
    alpha: float = 0.7,
) -> float:
    shape = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I3, 0.0)
    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    return alpha * shape + (1 - alpha) * hist

def preprocess_char_for_comparison(
    char_img: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    _, thr = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    return cv2.resize(thr, CHAR_SIZE2, interpolation=cv2.INTER_NEAREST), contours