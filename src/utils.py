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
