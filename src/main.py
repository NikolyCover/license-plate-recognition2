import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

CHAR_SIZE = (50, 80) # Dimensões padrão para normalização de caracteres
F_SCORE = 0.6      # limiar ajustável ao seu dataset (valores entre 0.3 e 0.7)
LABEL_HOLES = {
    # Dígitos
    "0": 0, "1": 0, "2": 0, "3": 0, "4": 0,
    "5": 0, "6": 1, "7": 0, "8": 2, "9": 0,

    # Letras (A–Z)
    "A": 1, "B": 2, "C": 0, "D": 1, "E": 0,
    "F": 0, "G": 0, "H": 0, "I": 0, "J": 0,
    "K": 0, "L": 0, "M": 0, "N": 0, "O": 1,
    "P": 1, "Q": 1, "R": 0, "S": 0, "T": 0,
    "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0,
    "Z": 0,
}

HOLE_MISMATCH_PENALTY = 0.25  # penalidade por diferença de 1 buraco
HOLE_MATCH_BONUS = -0.05 # pequeno bônus quando coincide

def process_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    Pré-processa a imagem da placa para segmentação.
    Aplica binarização inversa para obter caracteres brancos em fundo preto,
    facilitando a detecção de contornos.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Kernel de 3x3 é um bom ponto de partida para remover pequenos ruídos
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)

    return eroded

def standardize_char(
    char_img: np.ndarray, size: Tuple[int, int] = CHAR_SIZE
) -> np.ndarray:
    """
    Centraliza e normaliza um caractere para um tamanho padrão, preservando
    sua proporção original.
    """
    h, w = char_img.shape
    if h == 0 or w == 0:
        return np.zeros(size[::-1], dtype=np.uint8)

    scale = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(char_img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(size[::-1], dtype=np.uint8)
    x_offset = (size[0] - nw) // 2
    y_offset = (size[1] - nh) // 2
    canvas[y_offset : y_offset + nh, x_offset : x_offset + nw] = resized

    return canvas

def segment_characters(binary_image: np.ndarray) -> List[np.ndarray]:
    """
    Segmenta caracteres de uma imagem binária.
    Retorna uma lista de imagens de caracteres padronizados.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtra contornos baseados em altura e largura para remover ruídos
        if h > 25 and w > 10:
            char_img = binary_image[y : y + h, x : x + w]
            standardized_char = standardize_char(char_img)
            chars.append((x, standardized_char))

    chars.sort(key=lambda c: c[0])

    return [c[1] for c in chars]

def preprocess_char_for_comparison(
    char_img: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Prepara um caractere para a comparação.
    A binarização é invertida para que os contornos sejam detectados corretamente.
    """
    # A imagem já é padronizada, mas pode ser necessário uma binarização extra
    # para garantir que os contornos sejam limpos.
    _, thresh = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Redimensiona para um tamanho menor e fixo para a comparação
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
    Calcula a similaridade entre dois caracteres combinando forma e histograma.
    O 'alpha' foi ajustado para dar mais peso à forma, que é mais distintiva
    no seu dataset.
    """
    # A comparação de formas é muito eficaz para caracteres do seu dataset
    shape_score = cv2.matchShapes(
        contours1[0], contours2[0], cv2.CONTOURS_MATCH_I3, 0.0
    )

    # A comparação de histograma também contribui
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return alpha * shape_score + (1 - alpha) * hist_score

def load_models(model_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Carrega as imagens de referência (modelos) e padroniza.
    Suporta múltiplos modelos por caractere, com nomes tipo: A_01.png, 7_02.jpg.
    Retorna: {"A": [imgA_01, imgA_02, ...], "7": [img7_01, ...], ...}
    """
    models: Dict[str, List[np.ndarray]] = {}
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in os.listdir(model_dir):
        if not filename.lower().endswith(valid_ext):
            continue

        stem = os.path.splitext(filename)[0]  # ex: "A_01"
        parts = stem.split("_", 1)
        if not parts:
            continue

        allowed_suffixes = ["00", "01", "02", "05"]

        if parts[1] not in allowed_suffixes:
            continue

        label = parts[0].upper()  # suporta letras e dígitos
        # (opcional) validar sufixo de fonte: parts[1] com 2 dígitos

        filepath = os.path.join(model_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # binariza para manter "caractere branco" em "fundo preto"
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        standardized_img = standardize_char(bin_img)

        models.setdefault(label, []).append(standardized_img)

    return models

def best_label_by_median(char_proc, char_contours, models, adjust_fn=None):
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
            score = np.median(sorted(ds)[:3])  # mediana dos 3 melhores
            if adjust_fn is not None:
                score += float(adjust_fn(label))  # aplica bônus/penalidade por label
            if score < best_score:
                best_score, best_label = score, label
    return best_label, best_score

def count_holes(img_bin_255: np.ndarray) -> int:
    """
    Conta 'buracos' (counters) em uma imagem binária 0/255 com foreground branco.
    Usa RETR_CCOMP para pegar nível externo e seus filhos (furos).
    """
    cnts, hierarchy = cv2.findContours(img_bin_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    # hier[0][i] = [next, prev, first_child, parent]
    for h in hierarchy[0]:
        parent = h[3]
        if parent != -1:
            holes += 1

    print(f"Buracos encontrados: {holes}")
    return holes

def make_hole_adjust_fn(char_proc_bin_255: np.ndarray):
    """
    Cria uma função de ajuste de score por label, baseada no nº de buracos do caractere.
    Retorna delta a ser somado ao score (positivo = pior, negativo = melhor).
    """
    ch = count_holes(char_proc_bin_255)
    def adjust(label: str) -> float:
        exp = LABEL_HOLES.get(label)
        if exp is None:
            return 0.0
        diff = abs(ch - exp)
        if diff == 0:
            return HOLE_MATCH_BONUS
        return HOLE_MISMATCH_PENALTY * diff
    return adjust

## resolver O -> Q
## resolver R -> X

def _quadrant_sums(img):
    h, w = img.shape
    mh, mw = h // 2, w // 2
    q = lambda a: int(np.count_nonzero(a))
    return (
        q(img[0:mh, 0:mw]),      # TL
        q(img[0:mh, mw:w]),      # TR
        q(img[mh:h, 0:mw]),      # BL
        q(img[mh:h, mw:w]),      # BR
    )

def _left_right_ratio(img):
    h, w = img.shape
    left  = np.count_nonzero(img[:, :w//2])
    right = np.count_nonzero(img[:, w//2:])
    return left / (right + 1e-6)

def _center_cross_density(img):
    h, w = img.shape
    ch, cw = int(0.35*h), int(0.35*w)
    ys, xs = (h - ch)//2, (w - cw)//2
    center = img[ys:ys+ch, xs:xs+cw]
    return np.count_nonzero(center) / (center.size + 1e-6)

def _hole_offset_norm(img_bin_255: np.ndarray) -> float | None:
    # mede a distância (em pixels) entre o centro do furo e o centro do contorno externo
    cnts, hier = cv2.findContours(img_bin_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not cnts:
        return None
    # maior contorno externo
    outer_idx = max(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]))
    # primeiro filho (furo) desse contorno
    hole_idx = next((i for i,(nxt,prev,child,parent) in enumerate(hier[0]) if parent == outer_idx), None)
    if hole_idx is None:
        return None

    def centroid(c):
        m = cv2.moments(c)
        return (m["m10"]/m["m00"], m["m01"]/m["m00"]) if m["m00"] != 0 else (None, None)

    cx_o, cy_o = centroid(cnts[outer_idx])
    cx_h, cy_h = centroid(cnts[hole_idx])
    if None in (cx_o, cy_o, cx_h, cy_h):
        return None
    dx, dy = (cx_h - cx_o), (cy_h - cy_o)
    return float(np.hypot(dx, dy))  # norma (pixels)

def _o_vs_q(img):
    # garante foreground=255
    img = (img > 0).astype(np.uint8) * 255

    TL, TR, BL, BR = _quadrant_sums(img)
    total = TL + TR + BL + BR + 1e-6
    br_ratio = BR / total

    # assimetria diagonal leve (como já usamos)
    bin01 = (img > 0).astype(np.uint8)
    m = cv2.moments(bin01, binaryImage=True)
    diag_corr = m["mu11"] / (np.sqrt((m["mu20"] + 1e-6)*(m["mu02"] + 1e-6)) + 1e-12)

    # NOVO: deslocamento do furo
    hole_shift = _hole_offset_norm(img) or 0.0

    # --- decisão ---
    # Q: BR levemente mais pesado OU (BR um pouco acima + furo deslocado) OU (furo bem deslocado)
    if br_ratio >= 0.290 or (br_ratio >= 0.272 and hole_shift >= 1.0) or hole_shift >= 1.6:
        return "Q"

    # O: quadrantes equilibrados + pouco deslocamento e pouca correlação diagonal
    if max(abs((v/total) - 0.25) for v in (TL, TR, BL, BR)) <= 0.03 and hole_shift <= 0.7 and abs(diag_corr) < 0.02:
        return "O"

    return None

def _r_vs_x(img):
    lr = _left_right_ratio(img)       # >1 => mais massa à esquerda (tronco do R)
    cross = _center_cross_density(img)  # alto => cruzamento típico do X

    # Priorize tronco forte para R, mesmo com centro relativamente denso
    if lr >= 1.10:
        return "R"

    # X tem cruzamento MUITO denso e distribuição mais equilibrada
    if cross >= 0.70 and lr < 1.05:
        return "X"

    return None

def classify_character(char_img: np.ndarray, models: Dict[str, List[np.ndarray]], allowed_type: str | None = None) -> str:
    """
    Classifica um caractere usando a mediana das variantes como critério.
    allowed_type: 'L' para somente letras, 'D' para somente dígitos, None sem filtro.
    """
    # Filtra os modelos por tipo permitido
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

    adjust_fn = make_hole_adjust_fn(char_proc)  # usa buracos do caractere observado
    label, score = best_label_by_median(char_proc, char_contours, filtered, adjust_fn=adjust_fn)

    # desempate O vs Q
    if label in ("O", "Q"):
        decided = _o_vs_q(char_proc)
        if decided is not None:
            label = decided

    # desempate R vs X
    elif label in ("R", "X"):
        decided = _r_vs_x(char_proc)
        if decided is not None:
            label = decided

    return label if score <= F_SCORE else "?"

# ------------------------------
# Execução principal
# ------------------------------
def main(image_path: str, models_path: str):
    """
    Função principal para executar o fluxo de reconhecimento de placas.
    """
    print("Iniciando o reconhecimento de placa...")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return

    processed_plate = process_plate(img)
    characters = segment_characters(processed_plate)

    if not characters:
        print("Nenhum caractere detectado.")
        return

    models = load_models(models_path)

    if not models:
        print(f"Erro: Nenhum modelo encontrado no diretório {models_path}")
        return

    # Padrão Mercosul: ABC1D23  -> L,L,L,D,L,D,D
    pattern = ['L', 'L', 'L', 'D', 'L', 'D', 'D']

    recognized_chars = []
    for i, char in enumerate(characters):
        allowed = pattern[i] if i < len(pattern) else None  # se tiver >7, não filtra
        recognized_chars.append(classify_character(char, models, allowed_type=allowed))

    recognized_plate = "".join(recognized_chars)
    print("Placa reconhecida:", recognized_plate)

    cv2.imshow("Placa processada", processed_plate)
    for i, char in enumerate(characters):
        cv2.imshow(f"Caractere {i}", char)

    print("\nPressione qualquer tecla para fechar as janelas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("mock/PLATE_8.jpg", "characters")