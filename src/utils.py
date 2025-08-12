import os
import cv2
import numpy as np

from typing import Optional

from artifacts import PipelineArtifacts


def save_artifacts(artifacts: PipelineArtifacts, out_dir: str) -> None:
    """Salva imagens intermediárias e caracteres segmentados em `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    def _save(name: str, img: Optional[np.ndarray]):
        if img is not None:
            cv2.imwrite(os.path.join(out_dir, name), img)

    _save("00_original_bgr.png", artifacts.original_bgr)
    _save("01_gray.png", artifacts.gray)
    _save("02_binary_inv_otsu.png", artifacts.binary_inv_otsu)
    _save("03_eroded.png", artifacts.eroded)

    for i, seg in enumerate(artifacts.segmented):
        _save(f"char_{i:02d}_raw.png", seg.raw_crop)
        _save(f"char_{i:02d}_std.png", seg.standardized)

        if seg.proc50 is not None:
            _save(f"char_{i:02d}_proc50.png", seg.proc50)


def show_debug_windows(artifacts: PipelineArtifacts) -> None:
    """Exibe janelas OpenCV das etapas (opcional para debug)."""
    if artifacts.original_bgr is not None:
        cv2.imshow("00 Original", artifacts.original_bgr)

    if artifacts.gray is not None:
        cv2.imshow("01 Gray", artifacts.gray)

    if artifacts.binary_inv_otsu is not None:
        cv2.imshow("02 Binary INV Otsu", artifacts.binary_inv_otsu)

    if artifacts.eroded is not None:
        cv2.imshow("03 Eroded", artifacts.eroded)

    for i, seg in enumerate(artifacts.segmented):
        cv2.imshow(f"char {i:02d} - raw", seg.raw_crop)
        cv2.imshow(f"char {i:02d} - std", seg.standardized)

        if seg.proc50 is not None:
            cv2.imshow(f"char {i:02d} - proc50", seg.proc50)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
