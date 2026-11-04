"""Шаг 1b. EDA: сетевая часть (по parquet) и физическая часть (RTDS *_rscad.csv).
Результат: reports/eda_network.csv, reports/eda_physical.csv, figures/signals_*.png"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iecad.config import load_config, require_dataset_root
from iecad.discover import build_catalog, safe_name
from iecad.frames import CLS_NAMES
from iecad.windows import read_labels


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    cfg = load_config(a.config); root = require_dataset_root(cfg)
    cat = build_catalog(root, cfg); fdir = cfg["paths"]["interim"] / "frames"
    rows = []
    for r in cat.itertuples():
        p = fdir / (safe_name(r.capture) + ".parquet")
        if not p.exists():
            continue
        fr = pd.read_parquet(p)
        dur = (fr.t_ns.iloc[-1] - fr.t_ns.iloc[0]) / 1e9
        row = dict(capture=r.capture, scenario=r.scenario, group=r.group, vantage=r.vantage,
                   n_frames=len(fr), duration_s=round(dur, 2))
        for k, nme in enumerate(CLS_NAMES):
            row[f"n_{nme}"] = int((fr.cls == k).sum())
        sv = fr[fr.cls == 1]
        row["sv_rate_fps"] = round(len(sv) / dur, 1) if dur > 0 else np.nan
        row["sv_dup_smp"] = int((sv.sv_dsmp == 0).sum()); row["sv_smp_gaps"] = int((sv.sv_dsmp > 1).sum())
        row["vlan_frames"] = int(fr.vlan.sum())
        if r.has_labels:
            lab = read_labels(r.labels, len(fr)); row["n_malicious"] = int(lab.sum()); row["n_labels"] = len(lab)
        rows.append(row)
    net = pd.DataFrame(rows); net.to_csv(cfg["paths"]["reports"] / "eda_network.csv", index=False)
    print(net.groupby("scenario")[["n_frames", "duration_s", "sv_rate_fps"]].agg(["min", "max"]).to_string())

    phys = []
    for f in sorted((root / "Scenarios").rglob("*_rscad.csv")):
        df = pd.read_csv(f)
        row = dict(file=str(f.relative_to(root)), rows=len(df), n_missing=int(df.isna().sum().sum()),
                   n_dup=int(df.duplicated().sum()))
        for c in ["I1_a", "I1_b", "I1_c", "V1_a", "V1_b", "V1_c"]:
            if c in df:
                row.update({f"{c}_min": df[c].min(), f"{c}_max": df[c].max(), f"{c}_mean": df[c].mean(), f"{c}_std": df[c].std()})
        for c in ["FLT_input", "CB1_state", "ied_trip_signal", "vied_trip_signal"]:
            if c in df:
                row[f"{c}_frac1"] = float((df[c] > 0.5).mean())
        phys.append(row)
    if phys:
        pd.DataFrame(phys).to_csv(cfg["paths"]["reports"] / "eda_physical.csv", index=False)
        base = next((f for f in sorted((root / "Scenarios" / "Baseline").rglob("*_rscad.csv"))), None)
        flt = next((f for f in sorted((root / "Scenarios" / "Fault").rglob("*_rscad.csv"))), None)
        if base is not None and flt is not None:
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
            for a_, f, ttl in ((ax[0], base, "Baseline"), (ax[1], flt, "Fault")):
                d = pd.read_csv(f); a_.plot(d["time"], d["I1_a"], lw=.6); a_.set_ylabel("I1_a, kA"); a_.set_xlabel("time, s")
                a_.set_title(f"{ttl}: {f.name} (источник: RTDS export, TU Delft v2)", fontsize=9)
            fig.tight_layout(); fig.savefig(cfg["paths"]["figures"] / "signals_baseline_vs_fault.png", dpi=150)


if __name__ == "__main__":
    main()
