from typing import Dict, Tuple, Set

# Padrão fixo para a placa (L = Letra, D = Dígito)
PLATE_PATTERN = ["L", "L", "L", "D", "L", "D", "D"]

CHAR_SIZE: Tuple[int, int] = (50, 80)
CHAR_SIZE2: Tuple[int, int] = (50, 50)
MIN_CHAR_HEIGHT = 25
MIN_CHAR_WIDTH = 10
F_SCORE: float = 0.6
MIN_HOLE_AREA = 50  

LABEL_HOLES: Dict[str, int] = {
    "0": 1,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 1,
    "7": 0,
    "8": 2,
    "9": 1,
    "A": 1,
    "B": 2,
    "C": 0,
    "D": 1,
    "E": 0,
    "F": 0,
    "G": 0,
    "H": 0,
    "I": 0,
    "J": 0,
    "K": 0,
    "L": 0,
    "M": 0,
    "N": 0,
    "O": 1,
    "P": 1,
    "Q": 1,
    "R": 1,
    "S": 0,
    "T": 0,
    "U": 0,
    "V": 0,
    "W": 0,
    "X": 0,
    "Y": 0,
    "Z": 0,
}

ALLOWED_MODEL_SUFFIXES: Set[str] = {"00", "01", "02", "05"}
