from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class SegmentArtifact:
    """Etapas por caractere: crop original, padronizado e 50x50 p/ classificador."""
    bbox_x: int
    raw_crop: np.ndarray
    standardized: np.ndarray
    proc50: Optional[np.ndarray] = None
    contours: Optional[List[np.ndarray]] = None  # para matchShapes

@dataclass
class PipelineArtifacts:
    """Imagens intermediárias do pipeline para a GUI."""
    original_bgr: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    binary_inv_otsu: Optional[np.ndarray] = None
    eroded: Optional[np.ndarray] = None
    segmented: List[SegmentArtifact] = field(default_factory=list)
