from typing import List, Optional, Tuple
import cv2

from artifacts import PipelineArtifacts, SegmentArtifact
from preprocessing import process_plate, segment_characters
from similarity import preprocess_char_for_comparison
from load_models import load_models
from classify import classify_character
from utils import save_artifacts


def recognize_plate(
    image_path: str,
    models_path: str,
    pattern: Optional[List[str]] = None,
    collect_artifacts: bool = True,
) -> Tuple[str, Optional[PipelineArtifacts]]:
    """Executa o fluxo completo e (opcionalmente) retorna todas as imagens intermediárias."""
    img = cv2.imread(image_path)
    if img is None:
        return "", None

    eroded, gray, thr = process_plate(img)
    chars_std, boxes = segment_characters(eroded)

    arts = PipelineArtifacts(
        original_bgr=img.copy() if collect_artifacts else None,
        gray=gray if collect_artifacts else None,
        binary_inv_otsu=thr if collect_artifacts else None,
        eroded=eroded if collect_artifacts else None,
    )

    if collect_artifacts:
        for (x, y, w, h), std in zip(boxes, chars_std):
            raw = eroded[y : y + h, x : x + w]
            proc50, cnts = preprocess_char_for_comparison(std)
            arts.segmented.append(
                SegmentArtifact(
                    bbox_x=x,
                    raw_crop=raw,
                    standardized=std,
                    proc50=proc50,
                    contours=cnts,
                )
            )

    if not chars_std:
        return "", arts
    models = load_models(models_path)
    if not models:
        return "", arts

    pattern = pattern or ["L", "L", "L", "D", "L", "D", "D"]
    out = []

    for i, ch in enumerate(chars_std):
        allowed = pattern[i] if i < len(pattern) else None
        out.append(classify_character(ch, models, allowed_type=allowed))

    if arts:
        save_artifacts(arts, "out_artifacts")

    return "".join(out), arts


if __name__ == "__main__":
    text, arts = recognize_plate(
        "mock/PLATE_1.png", "characters", collect_artifacts=True
    )

    print("Placa reconhecida:", text)

    if arts:
        save_artifacts(arts, "out_artifacts")
