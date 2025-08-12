from typing import List, Optional, Tuple
import cv2

from artifacts import PipelineArtifacts, SegmentArtifact
from preprocessing import preprocess_plate, segment_characters
from similarity import preprocess_char_for_comparison
from load_models import load_models
from classify import classify_character
from utils import save_artifacts
from config import PLATE_PATTERN

def recognize_plate( 
    image_path: str,
    models_path: str,
    pattern: Optional[List[str]] = None,
    collect_artifacts: bool = True,
) -> Tuple[str, Optional[PipelineArtifacts]]:
    img = cv2.imread(image_path)

    if img is None:
        return "", None

    eroded, gray, binary = preprocess_plate(img)
    chars_std, boxes = segment_characters(eroded)

    artifacts = PipelineArtifacts(
        original_bgr=img.copy() if collect_artifacts else None,
        gray=gray if collect_artifacts else None,
        binary_inv_otsu=binary if collect_artifacts else None,
        eroded=eroded if collect_artifacts else None,
    )

    if collect_artifacts:
        for (x, y, w, h), std in zip(boxes, chars_std):
            raw = eroded[y : y + h, x : x + w]
            proc50, cnts = preprocess_char_for_comparison(std)

            artifacts.segmented.append(
                SegmentArtifact(
                    bbox_x=x,
                    raw_crop=raw,
                    standardized=std,
                    proc50=proc50,
                    contours=cnts,
                )
            )

    if not chars_std:
        return "", artifacts
    
    models = load_models(models_path)

    if not models:
        return "", artifacts

    pattern = pattern or PLATE_PATTERN
    result = []

    for i, char in enumerate(chars_std):
        allowed = pattern[i] if i < len(pattern) else None
        
        result.append(classify_character(char, models, allowed_type=allowed))

    if artifacts:
        save_artifacts(artifacts, "out_artifacts")

    return "".join(result), artifacts


if __name__ == "__main__":
    text, arts = recognize_plate(
        "mock/PLATE_1.png", "characters", collect_artifacts=True
    )

    print("Placa reconhecida:", text)

    if arts:
        save_artifacts(arts, "out_artifacts")
