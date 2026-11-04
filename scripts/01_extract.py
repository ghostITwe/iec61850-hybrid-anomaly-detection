"""Шаг 1. pcapng -> parquet с покадровыми полями (tshark). Кэшируется в data/interim/frames.
Запуск: python scripts/01_extract.py [--force] [--only SUBSTR] [--limit N]"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from iecad.config import load_config, require_dataset_root
from iecad.discover import build_catalog, safe_name
from iecad.frames import extract_to_parquet


def job(args):
    pcap, out, tshark, mod = args
    t = time.time(); n = extract_to_parquet(pcap, out, tshark, mod)
    return pcap, n, time.time() - t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--only"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config); root = require_dataset_root(cfg)
    cat = build_catalog(root, cfg)
    if a.only:
        cat = cat[cat.capture.str.contains(a.only)]
    if a.limit:
        cat = cat.head(a.limit)
    outdir = cfg["paths"]["interim"] / "frames"; outdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in cat.itertuples():
        out = outdir / (safe_name(r.capture) + ".parquet")
        if a.force or not out.exists():
            todo.append((r.pcap, str(out), cfg["extract"]["tshark"], cfg["window"]["sv_modulus"]))
    print(f"К обработке: {len(todo)} из {len(cat)}")
    with ProcessPoolExecutor(max_workers=cfg["extract"]["workers"]) as ex:
        for fut in as_completed([ex.submit(job, t) for t in todo]):
            p, n, dt = fut.result(); print(f"{Path(p).name}: {n} кадров, {dt:.0f} с", flush=True)


if __name__ == "__main__":
    main()
