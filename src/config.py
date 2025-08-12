from typing import Dict, Tuple, Set

CHAR_SIZE: Tuple[int, int] = (50, 80)
F_SCORE: float = 0.6

LABEL_HOLES: Dict[str, int] = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 1,
    "7": 0,
    "8": 2,
    "9": 0,
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
    "R": 0,
    "S": 0,
    "T": 0,
    "U": 0,
    "V": 0,
    "W": 0,
    "X": 0,
    "Y": 0,
    "Z": 0,
}

HOLE_MISMATCH_PENALTY: float = 0.25
HOLE_MATCH_BONUS: float = -0.05

ALLOWED_MODEL_SUFFIXES: Set[str] = {"00", "01", "02", "05"}
