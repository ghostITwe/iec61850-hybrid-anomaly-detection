"""Шаг 6. Метрики. Порог каждого метода — квантиль val-нормы. Основной тест: test_normal + attack.
Результат: reports/metrics.csv, metrics_by_scenario.csv, alert_rates_no_attack.csv, bootstrap_ap_macro.csv, bootstrap_ap_pooled.csv, metrics_by_family.csv, metrics_macro.csv, figures/pr_curves.png"""
import argparse, json, platform, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from iecad.config import load_config, ROOT
from iecad.metrics import point_metrics, cluster_bootstrap_ap, cluster_bootstrap_macro_ap
from iecad.scoring import threshold_from_val

M = {"if": "score_if", "ae": "score_ae", "hybrid": "score_hybrid"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    cfg = load_config(a.config); R = cfg["paths"]["reports"]
    df = pd.read_csv(R / "scores_all_methods.csv")
    val = df[df.split == "val"]
    thr = {k: threshold_from_val(val[c].to_numpy(), cfg["threshold"]["val_quantile"]) for k, c in M.items()}
    main_set = df[df.split.isin(["test_normal", "test_attack"])].reset_index(drop=True)
    y = main_set.label.to_numpy()

    rows = [dict(method=k, **point_metrics(y, main_set[c].to_numpy(), thr[k]), threshold=thr[k]) for k, c in M.items()]
    pd.DataFrame(rows).to_csv(R / "metrics.csv", index=False)

    sc = []
    for (scen, van), g in main_set[main_set.split == "test_attack"].groupby(["scenario", "vantage"]):
        for k, c in M.items():
            sc.append(dict(scenario=scen, vantage=van, method=k, **point_metrics(g.label.to_numpy(), g[c].to_numpy(), thr[k])))
    pd.DataFrame(sc).to_csv(R / "metrics_by_scenario.csv", index=False)

    al = []
    for (grp, spl), g in df.groupby(["group", "split"]):
        for k, c in M.items():
            al.append(dict(group=grp, split=spl, method=k, n_windows=len(g), alert_rate=float((g[c] > thr[k]).mean())))
    pd.DataFrame(al).to_csv(R / "alert_rates_no_attack.csv", index=False)

    att = df[df.split == "test_attack"].reset_index(drop=True)
    fam = pd.DataFrame([dict(family=f, method=k, **point_metrics(g.label.to_numpy(), g[c].to_numpy(), thr[k]))
                        for f, g in att.groupby("family") for k, c in M.items()])
    fam.to_csv(R / "metrics_by_family.csv", index=False)
    macro = fam.groupby("method")[["ap", "recall", "fpr"]].mean().add_prefix("macro_").reset_index()
    macro.to_csv(R / "metrics_macro.csv", index=False)
    bm = cluster_bootstrap_macro_ap(att, M, n=cfg["bootstrap"]["n"], seed=cfg["seed"])
    bm.to_csv(R / "bootstrap_ap_macro.csv", index=False)

    bt = cluster_bootstrap_ap(y, {k: main_set[c].to_numpy() for k, c in M.items()}, main_set.run_id.to_numpy(),
                              n=cfg["bootstrap"]["n"], seed=cfg["seed"])
    bt.to_csv(R / "bootstrap_ap_pooled.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, c in M.items():
        if 0 < y.sum() < len(y):
            p, r, _ = precision_recall_curve(y, main_set[c]); ax.plot(r, p, label=k.upper())
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR-кривые, общий тест (окна 1 с)"); ax.legend()
    fig.tight_layout(); fig.savefig(cfg["paths"]["figures"] / "pr_curves.png", dpi=150)

    try:
        gh = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        gh = ""
    import sklearn, pandas, numpy
    ver = dict(python=sys.version.split()[0], numpy=numpy.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__,
               platform=platform.platform(), git_commit=gh, seed=cfg["seed"], config=cfg["_config_path"], thresholds=thr)
    try:
        import torch; ver["torch"] = torch.__version__; ver["cuda"] = torch.cuda.is_available()
    except Exception:
        pass
    (R / "run_manifest.json").write_text(json.dumps(ver, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print("ОБЩИЙ ТЕСТ (вторичный):"); print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nMACRO по семействам (основной):"); print(macro.round(4).to_string(index=False)); print(bm.round(4).to_string(index=False))
    print("\nпо семействам:"); print(fam[["family","method","n","n_pos","ap","recall","fpr"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
