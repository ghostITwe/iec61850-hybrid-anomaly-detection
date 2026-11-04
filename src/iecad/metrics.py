from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def point_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp > 0 else (0.0 if tp + fn else np.nan)
    fpr = fp / (fp + tn) if fp + tn else np.nan
    out = dict(n=len(y), n_pos=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
               precision=prec, recall=rec, f1=f1, fpr=fpr)
    if 0 < y.sum() < len(y):
        out["ap"] = float(average_precision_score(y, s)); out["roc_auc"] = float(roc_auc_score(y, s))
    else:
        out["ap"] = np.nan; out["roc_auc"] = np.nan
    return out


def cluster_bootstrap_ap(y, scores: dict[str, np.ndarray], run_ids, n=500, seed=0) -> pd.DataFrame:
    """Кластерный bootstrap по прогонам: CI для AP каждого метода и для разностей гибрид − одиночный."""
    rng = np.random.default_rng(seed)
    runs = np.unique(run_ids)
    idx = {r: np.flatnonzero(run_ids == r) for r in runs}
    res = {k: [] for k in scores}
    for _ in range(n):
        pick = np.concatenate([idx[r] for r in rng.choice(runs, len(runs), replace=True)])
        yy = y[pick]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        for k, s in scores.items():
            res[k].append(average_precision_score(yy, s[pick]))
    rows = []
    for k, v in res.items():
        v = np.array(v); rows.append(dict(quantity=f"AP[{k}]", mean=v.mean(), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5), n_boot=len(v)))
    for base in [k for k in scores if k != "hybrid"]:
        d = np.array(res["hybrid"]) - np.array(res[base])
        rows.append(dict(quantity=f"AP[hybrid]-AP[{base}]", mean=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), n_boot=len(d)))
    return pd.DataFrame(rows)


def cluster_bootstrap_macro_ap(df: pd.DataFrame, score_cols: dict[str, str], n=500, seed=0) -> pd.DataFrame:
    """Основной bootstrap: AP считается внутри каждого семейства атак (негативы — окна с меткой 0 тех же прогонов),
    затем усредняется по семействам (macro-AP). Прогоны ресэмплируются ВНУТРИ семейства (стратифицированно)."""
    rng = np.random.default_rng(seed)
    fams = {}
    for fam, g in df.groupby("family"):
        idx = {r: gg.index.to_numpy() for r, gg in g.groupby("run_id")}
        fams[fam] = idx
    y = df["label"].to_numpy(); S = {k: df[c].to_numpy() for k, c in score_cols.items()}
    res = {k: [] for k in S}
    for _ in range(n):
        ap = {k: [] for k in S}; ok = True
        for fam, idx in fams.items():
            runs = list(idx)
            pick = np.concatenate([idx[r] for r in rng.choice(runs, len(runs), replace=True)])
            yy = y[pick]
            if yy.sum() == 0 or yy.sum() == len(yy):
                ok = False; break
            for k in S:
                ap[k].append(average_precision_score(yy, S[k][pick]))
        if ok:
            for k in S:
                res[k].append(np.mean(ap[k]))
    rows = []
    for k, v in res.items():
        v = np.array(v); rows.append(dict(quantity=f"macroAP[{k}]", mean=v.mean(), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5), n_boot=len(v)))
    for base in [k for k in S if k != "hybrid"]:
        d = np.array(res["hybrid"]) - np.array(res[base])
        rows.append(dict(quantity=f"macroAP[hybrid]-macroAP[{base}]", mean=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), n_boot=len(d)))
    return pd.DataFrame(rows)
