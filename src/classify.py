from typing import Dict, List, Optional, Tuple
import numpy as np

from similarity import preprocess_char_for_comparison, calculate_character_similarity
from heuristics import o_vs_q, r_vs_x
from config import F_SCORE, LABEL_HOLES
from heuristics import count_holes 

def best_label_by_median(
    char_proc, char_contours, models
) -> Tuple[str, float]:
    """Mediana dos 3 melhores scores entre variantes por rótulo (menor=melhor)."""
    best_label, best_score = "?", float("inf")

    for labels, variants in models.items():
        ds = []

        for m in variants:
            m_proc, m_cnt = preprocess_char_for_comparison(m)

            if m_proc is None or not m_cnt:
                continue

            ds.append(calculate_character_similarity(char_proc, m_proc, char_contours, m_cnt))
        if ds:
            score = float(np.median(sorted(ds)[:3]))

            if score < best_score:
                best_score, best_label = score, labels

    return best_label, best_score


def classify_character(
    char_img, models: Dict[str, List], allowed_type: Optional[str] = None
) -> str:
    """Classifica caractere respeitando filtro por tipo ('L','D') e desempates O/Q, R/X."""

    # Verificar se é letra ou dígito e filtrar os modelos
    if allowed_type == "L":
        models = {k: v for k, v in models.items() if k.isalpha()}
    elif allowed_type == "D":
        models = {k: v for k, v in models.items() if k.isdigit()}

    if not models:
        return "?"

    char_proc, char_cnt = preprocess_char_for_comparison(char_img)

    if char_proc is None or not char_cnt:
        return "?"
    
    num_holes = count_holes(char_proc)
    models = {k: v for k, v in models.items() if LABEL_HOLES.get(k, None) == num_holes}

    if not models:
        return "?"

    print('Caracter com', num_holes,' e Modelos após filtro:', list(models.keys()))

    label, score = best_label_by_median(char_proc, char_cnt, models)

    if label in ("O", "Q"):
        t = o_vs_q(char_proc)

        label = t or label
    elif label in ("R", "X"):
        t = r_vs_x(char_proc)

        label = t or label

    return label if score <= F_SCORE else "?"
