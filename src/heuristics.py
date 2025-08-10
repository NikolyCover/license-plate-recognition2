from typing import List, Optional, Tuple
import cv2
import numpy as np
from config import LABEL_HOLES, HOLE_MATCH_BONUS, HOLE_MISMATCH_PENALTY

def count_holes(img_bin_255: np.ndarray) -> int:
    """Conta furos com RETR_CCOMP (foreground=255)."""
    cnts, hier = cv2.findContours(img_bin_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return 0
    return sum(1 for h in hier[0] if h[3] != -1)

def make_hole_adjust_fn(char_proc_bin_255: np.ndarray):
    """Cria função delta(score) com base na diferença de furos observados vs esperados."""
    ch = count_holes(char_proc_bin_255)
    def adjust(label: str) -> float:
        diff = abs(ch - LABEL_HOLES.get(label, 0))
        return HOLE_MATCH_BONUS if diff == 0 else HOLE_MISMATCH_PENALTY * diff
    return adjust

def _quadrant_sums(img: np.ndarray) -> Tuple[int,int,int,int]:
    h, w = img.shape; mh, mw = h//2, w//2
    q = lambda a: int(np.count_nonzero(a))
    return q(img[:mh,:mw]), q(img[:mh,mw:]), q(img[mh:,:mw]), q(img[mh:,mw:])

def _left_right_ratio(img: np.ndarray) -> float:
    h, w = img.shape
    return np.count_nonzero(img[:,:w//2]) / (np.count_nonzero(img[:,w//2:]) + 1e-6)

def _center_cross_density(img: np.ndarray) -> float:
    h, w = img.shape; ch, cw = int(0.35*h), int(0.35*w)
    ys, xs = (h-ch)//2, (w-cw)//2
    center = img[ys:ys+ch, xs:xs+cw]
    return np.count_nonzero(center) / (center.size + 1e-6)

def _hole_offset_norm(img: np.ndarray) -> Optional[float]:
    """Distância entre centroides do contorno externo e do furo."""
    cnts, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not cnts: return None
    outer = max(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]))
    hole = next((i for i,(_,_,_,p) in enumerate(hier[0]) if p==outer), None)
    if hole is None: return None
    def centroid(c):
        m = cv2.moments(c); 
        return (m["m10"]/m["m00"], m["m01"]/m["m00"]) if m["m00"] else (None,None)
    cx_o, cy_o = centroid(cnts[outer]); cx_h, cy_h = centroid(cnts[hole])
    if None in (cx_o,cy_o,cx_h,cy_h): return None
    return float(np.hypot(cx_h-cx_o, cy_h-cy_o))

def o_vs_q(img: np.ndarray) -> Optional[str]:
    """Desempate O/Q combinando BR, μ11 e deslocamento do furo (mesmos limiares atuais)."""
    img = (img>0).astype(np.uint8)*255
    TL,TR,BL,BR = _quadrant_sums(img); total = TL+TR+BL+BR+1e-6
    br = BR/total
    m = cv2.moments((img>0).astype(np.uint8), True)
    diag = m["mu11"] / (np.sqrt((m["mu20"]+1e-6)*(m["mu02"]+1e-6)) + 1e-12)
    shift = _hole_offset_norm(img) or 0.0
    if br>=0.290 or (br>=0.272 and shift>=1.0) or shift>=1.6: return "Q"
    if max(abs((v/total)-0.25) for v in (TL,TR,BL,BR))<=0.03 and shift<=0.7 and abs(diag)<0.02: return "O"
    return None

def r_vs_x(img: np.ndarray) -> Optional[str]:
    """Desempate R/X por tronco à esquerda (R) e cruzamento central (X)."""
    lr = _left_right_ratio(img); cross = _center_cross_density(img)
    if lr >= 1.10: return "R"
    if cross >= 0.70 and lr < 1.05: return "X"
    return None
