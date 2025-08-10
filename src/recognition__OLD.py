# -*- coding: utf-8 -*-
"""
Reconhecimento de caracteres de placa com OpenCV.
Refatorado para separar responsabilidades, documentar funções e expor imagens/artefatos
para uso em interfaces gráficas (GUI).
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# ==============================
# Constantes e configuração
# ==============================

CHAR_SIZE: Tuple[int, int] = (50, 80)  # (largura, altura) do canvas dos caracteres
F_SCORE: float = 0.6                    # limiar de aceitação do rótulo (menor = melhor)

LABEL_HOLES: Dict[str, int] = {
    # Dígitos
    "0": 0, "1": 0, "2": 0, "3": 0, "4": 0,
    "5": 0, "6": 1, "7": 0, "8": 2, "9": 0,
    # Letras
    "A": 1, "B": 2, "C": 0, "D": 1, "E": 0,
    "F": 0, "G": 0, "H": 0, "I": 0, "J": 0,
    "K": 0, "L": 0, "M": 0, "N": 0, "O": 1,
    "P": 1, "Q": 1, "R": 0, "S": 0, "T": 0,
    "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0,
    "Z": 0,
}

HOLE_MISMATCH_PENALTY: float = 0.25  # penalidade por diferença de 1 buraco
HOLE_MATCH_BONUS: float = -0.05      # bônus quando coincide número de buracos

# Sufixos aceitos nos arquivos de modelo (mantido da sua lógica)
ALLOWED_MODEL_SUFFIXES = {"00", "01", "02", "05"}


# ==============================
# Estruturas para GUI / Artefatos
# ==============================

@dataclass
class SegmentArtifact:
    """Representa um caractere segmentado em diferentes estágios."""
    bbox_x: int
    raw_crop: np.ndarray           # recorte original binário (antes de padronizar)
    standardized: np.ndarray       # canvas padronizado (CHAR_SIZE)
    proc50: Optional[np.ndarray] = None   # imagem 50x50 usada na comparação
    contours: Optional[List[np.ndarray]] = None  # contornos usados em matchShapes


@dataclass
class PipelineArtifacts:
    """Coleção de imagens/etapas geradas ao processar a placa."""
    original_bgr: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    binary_inv_otsu: Optional[np.ndarray] = None
    eroded: Optional[np.ndarray] = None
    segmented: List[SegmentArtifact] = field(default_factory=list)


# ==============================
# Pré-processamento
# ==============================

def process_plate(plate_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converte a imagem da placa para cinza, aplica Otsu + binarização invertida
    e uma erosão leve (kernel 3x3), retornando (eroded, gray, thresh_inv).

    Retorno:
        eroded (np.ndarray): imagem binária (caractere branco/255, fundo preto/0) pós-erosão.
        gray   (np.ndarray): imagem em tons de cinza.
        thresh (np.ndarray): imagem binária invertida (Otsu) antes da erosão.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh_inv, kernel, iterations=1)
    return eroded, gray, thresh_inv


def standardize_char(char_img: np.ndarray, size: Tuple[int, int] = CHAR_SIZE) -> np.ndarray:
    """
    Centraliza e normaliza um caractere em um canvas fixo (size), preservando proporção.

    Args:
        char_img: imagem binária 0/255 do caractere.
        size: (largura, altura) do canvas.

    Returns:
        np.ndarray: imagem binária 0/255 do caractere centralizado no canvas.
    """
    h, w = char_img.shape
    if h == 0 or w == 0:
        return np.zeros(size[::-1], dtype=np.uint8)

    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(char_img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(size[::-1], dtype=np.uint8)  # (altura, largura)
    x_offset = (size[0] - nw) // 2
    y_offset = (size[1] - nh) // 2
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas


def segment_characters(binary_image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    """
    Segmenta contornos relevantes (caracteres) da placa.

    Critério:
      - apenas contornos com h > 25 e w > 10 (mesma lógica original).

    Returns:
        chars_std: lista de imagens binárias padronizadas (CHAR_SIZE).
        boxes: lista de bounding boxes (x, y, w, h) de cada caractere (não padronizado).
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items: List[Tuple[int, np.ndarray, Tuple[int,int,int,int]]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 25 and w > 10:
            char_img = binary_image[y:y + h, x:x + w]
            std = standardize_char(char_img)
            items.append((x, std, (x, y, w, h)))

    # ordenar por posição x (esquerda->direita)
    items.sort(key=lambda it: it[0])

    chars_std = [it[1] for it in items]
    boxes = [it[2] for it in items]
    return chars_std, boxes


# ==============================
# Preparação / Similaridade
# ==============================

def preprocess_char_for_comparison(char_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Prepara um caractere padronizado para comparação:
      - aplica threshold (garantia de binário),
      - encontra contornos (RETR_EXTERNAL),
      - redimensiona para 50x50 (igual à sua lógica).

    Returns:
        resized_50 (np.ndarray|None): imagem 50x50 binária para comparação.
        contours (List[np.ndarray]|None): contornos encontrados (para matchShapes).
    """
    _, thresh = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    resized = cv2.resize(thresh, (50, 50), interpolation=cv2.INTER_NEAREST)
    return resized, contours


def character_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
    contours1: List[np.ndarray],
    contours2: List[np.ndarray],
    alpha: float = 0.7,
) -> float:
    """
    Combina similaridade de forma (matchShapes) e de histograma (Bhattacharyya).
    Score menor = melhor.

    Args:
        img1, img2: imagens binárias 50x50.
        contours1, contours2: contornos (usa-se o primeiro em cada lista).
        alpha: peso da forma (0..1).

    Returns:
        float: score de dissimilaridade.
    """
    shape_score = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I3, 0.0)
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return alpha * shape_score + (1 - alpha) * hist_score


# ==============================
# Carregamento de modelos
# ==============================

def load_models(model_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Carrega imagens de referência de caracteres a partir de `model_dir`.

    Convenção de nome:
      - "<LABEL>_<SUFIXO>.png|jpg|..." onde SUFIXO ∈ {"00","01","02","05"}.

    Processamento:
      - lê em tons de cinza;
      - binariza invertendo (caractere branco);
      - padroniza no canvas CHAR_SIZE.

    Returns:
        Dict[label, List[img]] com as variantes de cada rótulo.
    """
    models: Dict[str, List[np.ndarray]] = {}
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in os.listdir(model_dir):
        if not filename.lower().endswith(valid_ext):
            continue

        stem = os.path.splitext(filename)[0]   # ex: "A_01"
        parts = stem.split("_", 1)
        if len(parts) != 2 or parts[1] not in ALLOWED_MODEL_SUFFIXES:
            continue

        label = parts[0].upper()
        path = os.path.join(model_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, bin_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        standardized = standardize_char(bin_inv)
        models.setdefault(label, []).append(standardized)

    return models


# ==============================
# Heurísticas auxiliares (buracos, quadrantes, desempates)
# ==============================

def count_holes(img_bin_255: np.ndarray) -> int:
    """
    Conta 'buracos' (contornos filhos) usando RETR_CCOMP.
    Espera imagem binária 0/255 com foreground=255.

    Returns:
        int: número de buracos internos.
    """
    cnts, hierarchy = cv2.findContours(img_bin_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    for h in hierarchy[0]:
        parent = h[3]
        if parent != -1:
            holes += 1
    return holes


def make_hole_adjust_fn(char_proc_bin_255: np.ndarray):
    """
    Gera função de ajuste ao score baseado na diferença entre buracos esperados
    (LABEL_HOLES) e observados no caractere.

    Returns:
        Callable[[str], float]: delta acrescentado ao score (positivo piora).
    """
    ch = count_holes(char_proc_bin_255)
    def adjust(label: str) -> float:
        exp = LABEL_HOLES.get(label, 0)
        diff = abs(ch - exp)
        if diff == 0:
            return HOLE_MATCH_BONUS
        return HOLE_MISMATCH_PENALTY * diff
    return adjust


def _quadrant_sums(img: np.ndarray) -> Tuple[int, int, int, int]:
    """Conta pixels brancos em cada quadrante da imagem binária."""
    h, w = img.shape
    mh, mw = h // 2, w // 2
    q = lambda a: int(np.count_nonzero(a))
    return (
        q(img[0:mh, 0:mw]),     # TL
        q(img[0:mh, mw:w]),     # TR
        q(img[mh:h, 0:mw]),     # BL
        q(img[mh:h, mw:w]),     # BR
    )


def _left_right_ratio(img: np.ndarray) -> float:
    """Razão de massa branca entre metade esquerda e direita (tronco do 'R')."""
    h, w = img.shape
    left = np.count_nonzero(img[:, :w // 2])
    right = np.count_nonzero(img[:, w // 2:])
    return left / (right + 1e-6)


def _center_cross_density(img: np.ndarray) -> float:
    """Densidade de pixels em janela central (~35%) — útil para 'X'."""
    h, w = img.shape
    ch, cw = int(0.35 * h), int(0.35 * w)
    ys, xs = (h - ch) // 2, (w - cw) // 2
    center = img[ys:ys + ch, xs:xs + cw]
    return np.count_nonzero(center) / (center.size + 1e-6)


def _hole_offset_norm(img_bin_255: np.ndarray) -> Optional[float]:
    """
    Distância (em px) entre o centróide do furo e o centróide do contorno externo.
    Q tende a ter deslocamento maior (cauda).
    """
    cnts, hier = cv2.findContours(img_bin_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not cnts:
        return None
    # maior contorno externo
    outer_idx = max(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]))
    # algum filho (furo) desse contorno
    hole_idx = next((i for i, (_, _, _, parent) in enumerate(hier[0]) if parent == outer_idx), None)
    if hole_idx is None:
        return None

    def centroid(c):
        m = cv2.moments(c)
        return (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] != 0 else (None, None)

    cx_o, cy_o = centroid(cnts[outer_idx])
    cx_h, cy_h = centroid(cnts[hole_idx])
    if None in (cx_o, cy_o, cx_h, cy_h):
        return None
    return float(np.hypot(cx_h - cx_o, cy_h - cy_o))


def _o_vs_q(img: np.ndarray) -> Optional[str]:
    """
    Desempate O vs Q combinando:
      - massa em BR (cauda de Q),
      - correlação diagonal (μ11),
      - deslocamento do furo (maior em Q).
    """
    img = (img > 0).astype(np.uint8) * 255

    TL, TR, BL, BR = _quadrant_sums(img)
    total = TL + TR + BL + BR + 1e-6
    br_ratio = BR / total

    bin01 = (img > 0).astype(np.uint8)
    m = cv2.moments(bin01, binaryImage=True)
    diag_corr = m["mu11"] / (np.sqrt((m["mu20"] + 1e-6) * (m["mu02"] + 1e-6)) + 1e-12)

    hole_shift = _hole_offset_norm(img) or 0.0

    # Mesma lógica/limiares que você já utiliza
    if br_ratio >= 0.290 or (br_ratio >= 0.272 and hole_shift >= 1.0) or hole_shift >= 1.6:
        return "Q"

    if max(abs((v / total) - 0.25) for v in (TL, TR, BL, BR)) <= 0.03 and hole_shift <= 0.7 and abs(diag_corr) < 0.02:
        return "O"

    return None


def _r_vs_x(img: np.ndarray) -> Optional[str]:
    """
    Desempate R vs X:
      - 'R' tem tronco à esquerda => razão esquerda/direita alta.
      - 'X' tem cruzamento central denso e distribuição mais equilibrada.
    """
    lr = _left_right_ratio(img)
    cross = _center_cross_density(img)

    if lr >= 1.10:
        return "R"
    if cross >= 0.70 and lr < 1.05:
        return "X"
    return None


# ==============================
# Classificação
# ==============================

def best_label_by_median(
    char_proc: np.ndarray,
    char_contours: List[np.ndarray],
    models: Dict[str, List[np.ndarray]],
    adjust_fn=None,
) -> Tuple[str, float]:
    """
    Calcula o melhor rótulo pela mediana dos 3 melhores scores contra as variantes.

    Returns:
        (label, score)
    """
    best_label, best_score = "?", float("inf")
    for label, variants in models.items():
        ds = []
        for model_img in variants:
            m_proc, m_contours = preprocess_char_for_comparison(model_img)
            if m_proc is None or not m_contours:
                continue
            s = character_similarity(char_proc, m_proc, char_contours, m_contours)
            ds.append(s)
        if ds:
            score = float(np.median(sorted(ds)[:3]))
            if adjust_fn is not None:
                score += float(adjust_fn(label))
            if score < best_score:
                best_score, best_label = score, label
    return best_label, best_score


def classify_character(
    char_img: np.ndarray,
    models: Dict[str, List[np.ndarray]],
    allowed_type: Optional[str] = None
) -> str:
    """
    Classifica um caractere usando:
      - mediana dos 3 melhores scores,
      - ajuste por buracos,
      - desempates específicos (O/Q e R/X).
    Respeita `allowed_type`: 'L' (somente letras), 'D' (somente dígitos) ou None.

    Returns:
        str: rótulo previsto (ou "?" se score > F_SCORE).
    """
    # filtro por tipo
    if allowed_type == 'L':
        filtered = {k: v for k, v in models.items() if k.isalpha()}
    elif allowed_type == 'D':
        filtered = {k: v for k, v in models.items() if k.isdigit()}
    else:
        filtered = models
    if not filtered:
        return "?"

    char_proc, char_contours = preprocess_char_for_comparison(char_img)
    if char_proc is None or not char_contours:
        return "?"

    adjust_fn = make_hole_adjust_fn(char_proc)
    label, score = best_label_by_median(char_proc, char_contours, filtered, adjust_fn=adjust_fn)

    # Desempates (mantidos)
    if label in ("O", "Q"):
        decided = _o_vs_q(char_proc)
        if decided is not None:
            label = decided
    elif label in ("R", "X"):
        decided = _r_vs_x(char_proc)
        if decided is not None:
            label = decided

    return label if score <= F_SCORE else "?"


# ==============================
# Pipeline de alto nível
# ==============================

def recognize_plate(
    image_path: str,
    models_path: str,
    pattern: Optional[List[str]] = None,
    collect_artifacts: bool = True
) -> Tuple[str, Optional[PipelineArtifacts]]:
    """
    Executa o fluxo completo de reconhecimento de placa.

    Args:
        image_path: caminho da imagem da placa.
        models_path: diretório com os modelos de caracteres.
        pattern: padrão Mercosul (['L','L','L','D','L','D','D']); se None, usa padrão default.
        collect_artifacts: se True, retorna todas as imagens geradas.

    Returns:
        (recognized_text, artifacts)
    """
    img = cv2.imread(image_path)
    if img is None:
        return "", None

    eroded, gray, thresh_inv = process_plate(img)
    chars_std, boxes = segment_characters(eroded)

    # montar artefatos (inclui crops brutos e padronizados)
    artifacts = PipelineArtifacts(original_bgr=img.copy() if collect_artifacts else None,
                                  gray=gray if collect_artifacts else None,
                                  binary_inv_otsu=thresh_inv if collect_artifacts else None,
                                  eroded=eroded if collect_artifacts else None)

    if collect_artifacts:
        # guardar cada caractere com seus estágios
        for (x, y, w, h), std in zip(boxes, chars_std):
            raw = eroded[y:y + h, x:x + w]  # recorte original binário
            proc50, cnts = preprocess_char_for_comparison(std)
            artifacts.segmented.append(
                SegmentArtifact(bbox_x=x, raw_crop=raw, standardized=std, proc50=proc50, contours=cnts)
            )

    if not chars_std:
        return "", artifacts

    models = load_models(models_path)
    if not models:
        return "", artifacts

    # padrão Mercosul padrão
    if pattern is None:
        pattern = ['L', 'L', 'L', 'D', 'L', 'D', 'D']

    out_chars: List[str] = []
    for i, char_img in enumerate(chars_std):
        allowed = pattern[i] if i < len(pattern) else None
        out_chars.append(classify_character(char_img, models, allowed_type=allowed))

    recognized = "".join(out_chars)
    return recognized, artifacts


# ==============================
# Utilidades para GUI
# ==============================

def save_artifacts(artifacts: PipelineArtifacts, out_dir: str) -> None:
    """
    Salva todas as imagens intermediárias e caracteres segmentados em `out_dir`.
    Nomes previsíveis para integração com GUI.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _save(name: str, img: Optional[np.ndarray]):
        if img is None:
            return
        cv2.imwrite(os.path.join(out_dir, name), img)

    _save("00_original_bgr.png", artifacts.original_bgr)
    _save("01_gray.png", artifacts.gray)
    _save("02_binary_inv_otsu.png", artifacts.binary_inv_otsu)
    _save("03_eroded.png", artifacts.eroded)

    for idx, seg in enumerate(artifacts.segmented):
        _save(f"char_{idx:02d}_raw.png", seg.raw_crop)
        _save(f"char_{idx:02d}_std.png", seg.standardized)
        if seg.proc50 is not None:
            _save(f"char_{idx:02d}_proc50.png", seg.proc50)


def show_debug_windows(artifacts: PipelineArtifacts) -> None:
    """
    (Opcional) Exibe as etapas no OpenCV para depuração.
    Não altera a lógica do pipeline.
    """
    if artifacts.original_bgr is not None:
        cv2.imshow("00 Original (BGR)", artifacts.original_bgr)
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


# ==============================
# Execução direta (exemplo)
# ==============================

if __name__ == "__main__":
    text, arts = recognize_plate("mock/PLATE_7.png", "characters", collect_artifacts=True)
    print("Placa reconhecida:", text)

    if arts:
        # Exemplo de como salvar para a GUI consumir:
        save_artifacts(arts, out_dir="out_artifacts")
        # Se quiser visualizar:
        # show_debug_windows(arts)
