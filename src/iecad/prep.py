"""Единая подготовка матрицы признаков: log-преобразование, scaler ТОЛЬКО по train, clip."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .windows import feature_columns, FORBIDDEN_SUBSTR

SD_FLOOR = 1e-2


def load_matrix(cfg: dict):
    feats = pd.read_parquet(cfg["paths"]["processed"] / "features.parquet")
    cols = feature_columns(cfg)
    bad = [c for c in cols if any(s in c.lower() for s in FORBIDDEN_SUBSTR)]
    assert not bad, f"запрещённые признаки: {bad}"
    X = feats[cols].to_numpy(float)
    X = np.sign(X) * np.log1p(np.abs(X))
    tr = (feats["split"] == "train").to_numpy()
    assert tr.any(), "нет обучающих окон"
    mu = X[tr].mean(0)
    sd = np.maximum(X[tr].std(0), SD_FLOOR)
    Z = np.clip((X - mu) / sd, -cfg["preprocess"]["clip_z"], cfg["preprocess"]["clip_z"])
    (cfg["paths"]["models"] / "scaler.json").write_text(
        json.dumps({"cols": cols, "mu": mu.tolist(), "sd": sd.tolist(), "fit_on": "split==train"}, indent=1))
    return feats, Z.astype(np.float32), cols
