"""Приведение оценок к общей шкале (по нормальной валидации) и пороги."""
from __future__ import annotations
import numpy as np


def rank_with_tail(val_scores: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Эмпирическая CDF по val-норме; выше максимума val шкала продолжается логарифмически
    (монотонно), чтобы не терять порядок среди сильных аномалий."""
    v = np.sort(np.asarray(val_scores, float))
    s = np.asarray(s, float)
    r = np.searchsorted(v, s, side="right") / len(v)
    over = s > v[-1]
    if over.any():
        scale = max(v[-1] - np.median(v), 1e-12)
        r[over] = 1.0 + np.log1p((s[over] - v[-1]) / scale)
    return r


def threshold_from_val(val_scores: np.ndarray, q: float) -> float:
    return float(np.quantile(val_scores, q))
