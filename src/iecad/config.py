"""Загрузка конфигурации и путей."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "experiment_v1.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    root = os.environ.get("DATASET_ROOT") or cfg["paths"].get("dataset_root")
    cfg["paths"]["dataset_root"] = root
    for k in ("interim", "processed", "reports", "figures", "models"):
        p = Path(cfg["paths"][k])
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg


def require_dataset_root(cfg: dict) -> Path:
    r = cfg["paths"]["dataset_root"]
    if not r:
        raise SystemExit("Укажите путь к датасету: переменная DATASET_ROOT или paths.dataset_root в конфиге.")
    return Path(r)
