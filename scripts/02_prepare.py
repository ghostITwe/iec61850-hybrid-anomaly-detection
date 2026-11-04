"""Шаг 2. Оконные признаки + метки + разбиение по прогонам.
Результат: data/processed/features.parquet, reports/split_manifest.csv"""
import argparse
from iecad.config import load_config, require_dataset_root
from iecad.discover import build_catalog, safe_name
from iecad.splits import assign_splits
from iecad.windows import window_features, read_labels
import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    cfg = load_config(a.config); root = require_dataset_root(cfg)
    cat = build_catalog(root, cfg)
    man = assign_splits(cat, cfg)
    man.to_csv(cfg["paths"]["reports"] / "split_manifest.csv", index=False)
    print(man.groupby(["group", "split"]).run_id.nunique().to_string())
    split_of = dict(zip(man.run_id, man.split))
    fdir = cfg["paths"]["interim"] / "frames"; parts = []
    for r in cat.itertuples():
        if split_of[r.run_id] == "unused":
            continue
        p = fdir / (safe_name(r.capture) + ".parquet")
        if not p.exists():
            print("нет parquet (запустите 01_extract):", r.capture); continue
        fr = pd.read_parquet(p)
        lab = None
        if r.group == "attack":
            if not r.has_labels:
                print("пропуск (нет меток):", r.capture); continue
            lab = read_labels(r.labels, len(fr))
        w = window_features(fr, lab, cfg)
        w.insert(0, "capture", r.capture)
        for k in ("run_id", "scenario", "family", "vantage", "group"):
            w[k] = getattr(r, k)
        w["split"] = split_of[r.run_id]
        w["uid"] = r.capture + "|" + w["window_idx"].astype(str)
        parts.append(w)
    feats = pd.concat(parts, ignore_index=True)
    feats.to_parquet(cfg["paths"]["processed"] / "features.parquet", index=False)
    print(feats.groupby(["group", "split"]).agg(windows=("uid", "size"), pos=("label", "sum"), runs=("run_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
