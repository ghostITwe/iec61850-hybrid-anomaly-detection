"""Каталог захватов датасета: сценарий, прогон, ракурс (vantage), группа оценки, файл меток."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

_ENDS_WITH_NUM = re.compile(r"\d+$")


def find_scenarios_dir(root: str | Path) -> Path:
    root = Path(root)
    if root.name == "Scenarios" and root.is_dir():
        return root
    for p in [root, *sorted(root.glob("*"))]:
        if (p / "Scenarios").is_dir():
            return p / "Scenarios"
    raise FileNotFoundError(f"Папка Scenarios не найдена внутри {root}")


def _group_of(family: str, scenario: str, cfg: dict) -> str:
    for s in cfg["normal_pool"]:
        if scenario == s or scenario.startswith(s + "/"):
            return "normal_pool"
    g = cfg["group_map"].get(family, "other")
    return "other" if g == "normal_pool" else g


def build_catalog(root: str | Path, cfg: dict) -> pd.DataFrame:
    sdir = find_scenarios_dir(root)
    rows = []
    for f in sorted(sdir.rglob("*.pcapng")):
        parts = f.relative_to(sdir).parts[:-1]
        if not parts:
            continue
        family = parts[0]
        run_id = "/".join(parts)
        # папка прогона оканчивается цифрой (baseline1, ied_..._3) -> сценарий = родитель
        scenario = "/".join(parts[:-1]) if (len(parts) > 1 and _ENDS_WITH_NUM.search(parts[-1])) else run_id
        stem = f.stem
        vantage = "vied" if "_vied_" in stem else "attacker" if "_attacker_" in stem else "single"
        labels = f.with_name(stem + "_labels.csv")
        rows.append(dict(
            capture=f"{run_id}/{stem}", run_id=run_id, scenario=scenario, family=family,
            vantage=vantage, group=_group_of(family, scenario, cfg),
            pcap=str(f), labels=str(labels) if labels.exists() else "",
            has_labels=labels.exists(), size_mb=round(f.stat().st_size / 1e6, 2),
        ))
    return pd.DataFrame(rows)


def safe_name(capture: str) -> str:
    return capture.replace("/", "__").replace(" ", "_")
