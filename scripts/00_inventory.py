"""Шаг 0. Инвентаризация датасета: все файлы, заголовки CSV, каталог захватов, версия архива.
Результат: reports/inventory.csv, reports/catalog.csv
Запуск: python scripts/00_inventory.py [--sha256]"""
import argparse, csv, hashlib, sys
from pathlib import Path
import pandas as pd
from iecad.config import load_config, require_dataset_root
from iecad.discover import build_catalog


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--sha256", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config); root = require_dataset_root(cfg)
    rows = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        r = dict(path=str(f.relative_to(root)), ext=f.suffix.lower(), size_mb=round(f.stat().st_size / 1e6, 3), header="")
        if f.suffix.lower() == ".csv":
            with open(f, newline="", encoding="utf-8", errors="replace") as fh:
                r["header"] = "|".join(next(csv.reader(fh), []))
        if a.sha256:
            r["sha256"] = sha256(f)
        rows.append(r)
    inv = pd.DataFrame(rows)
    inv.to_csv(cfg["paths"]["reports"] / "inventory.csv", index=False)
    cat = build_catalog(root, cfg)
    cat.to_csv(cfg["paths"]["reports"] / "catalog.csv", index=False)
    print(f"Корень датасета: {root}  (версия по имени папки/README: проверьте, что это v2, 2026-10-19)")
    print(f"Файлов: {len(inv)}; по расширениям:\n{inv.ext.value_counts().to_string()}")
    print(f"\nЗахватов pcapng: {len(cat)}; с метками: {int(cat.has_labels.sum())}; прогонов: {cat.run_id.nunique()}")
    print(cat.groupby(['group', 'scenario']).agg(captures=('capture', 'size'), runs=('run_id', 'nunique'),
                                                 labeled=('has_labels', 'sum')).to_string())
    bad = cat[(cat.group == "attack") & ~cat.has_labels]
    if len(bad):
        print("\nВНИМАНИЕ: атакующие захваты без файла меток:\n", bad.capture.to_string())
    print("\nПроверьте reports/catalog.csv глазами: scenario/run_id определяются по структуре папок.")


if __name__ == "__main__":
    main()
