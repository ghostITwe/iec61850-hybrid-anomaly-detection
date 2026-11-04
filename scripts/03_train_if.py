"""Шаг 3. Isolation Forest: обучение на train (норма), оценки для всех окон."""
import argparse, joblib
import pandas as pd
from iecad.config import load_config
from iecad.prep import load_matrix
from iecad.models import fit_if, score_if


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    cfg = load_config(a.config)
    feats, X, cols = load_matrix(cfg)
    tr = (feats.split == "train").to_numpy()
    m = fit_if(X[tr], cfg, cfg["seed"])
    joblib.dump(m, cfg["paths"]["models"] / "isolation_forest.joblib")
    pd.DataFrame({"uid": feats.uid, "score_if": score_if(m, X)}).to_parquet(cfg["paths"]["processed"] / "scores_if.parquet", index=False)
    print(f"IF обучен на {int(tr.sum())} нормальных окнах, признаков: {len(cols)}")


if __name__ == "__main__":
    main()
