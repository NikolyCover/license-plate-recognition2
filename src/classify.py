from typing import Dict, List, Optional, Tuple
import numpy as np
from similarity import preprocess_char_for_comparison, character_similarity
from heuristics import make_hole_adjust_fn, o_vs_q, r_vs_x
from config import F_SCORE

def best_label_by_median(char_proc, char_contours, models, adjust_fn=None) -> Tuple[str, float]:
    """Mediana dos 3 melhores scores entre variantes por rótulo (menor=melhor)."""
    best_label, best_score = "?", float("inf")
    for label, variants in models.items():
        ds = []
        for m in variants:
            m_proc, m_cnt = preprocess_char_for_comparison(m)
            if m_proc is None or not m_cnt: continue
            ds.append(character_similarity(char_proc, m_proc, char_contours, m_cnt))
        if ds:
            score = float(np.median(sorted(ds)[:3]))
            if adjust_fn is not None: score += float(adjust_fn(label))
            if score < best_score: best_score, best_label = score, label
    return best_label, best_score

def classify_character(char_img, models: Dict[str, List], allowed_type: Optional[str]=None) -> str:
    """Classifica caractere respeitando filtro por tipo ('L','D') e desempates O/Q, R/X."""
    if allowed_type == 'L':
        models = {k:v for k,v in models.items() if k.isalpha()}
    elif allowed_type == 'D':
        models = {k:v for k,v in models.items() if k.isdigit()}
    if not models: return "?"

    char_proc, char_cnt = preprocess_char_for_comparison(char_img)
    if char_proc is None or not char_cnt: return "?"

    label, score = best_label_by_median(char_proc, char_cnt, models, adjust_fn=make_hole_adjust_fn(char_proc))
    if label in ("O","Q"):
        t = o_vs_q(char_proc);  label = t or label
    elif label in ("R","X"):
        t = r_vs_x(char_proc); label = t or label

    return label if score <= F_SCORE else "?"
