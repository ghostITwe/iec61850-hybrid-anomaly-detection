"""Шаг 5. Гибрид: среднее val-рангов оценок IF и AE (правило зафиксировано до просмотра теста).
Результат: reports/scores_all_methods.csv"""
import argparse
import pandas as pd
from iecad.config import load_config
from iecad.scoring import rank_with_tail


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    cfg = load_config(a.config); P = cfg["paths"]["processed"]
    feats = pd.read_parquet(P / "features.parquet")[["uid", "capture", "run_id", "scenario", "family", "vantage", "group", "split", "window_idx", "t_start_s", "label", "n_pos_frames"]]
    df = feats.merge(pd.read_parquet(P / "scores_if.parquet"), on="uid").merge(pd.read_parquet(P / "scores_ae.parquet"), on="uid")
    val = df.split == "val"
    df["rank_if"] = rank_with_tail(df.loc[val, "score_if"].to_numpy(), df.score_if.to_numpy())
    df["rank_ae"] = rank_with_tail(df.loc[val, "score_ae"].to_numpy(), df.score_ae.to_numpy())
    df["score_hybrid"] = (df.rank_if + df.rank_ae) / 2.0
    df.to_csv(cfg["paths"]["reports"] / "scores_all_methods.csv", index=False)
    print(f"{len(df)} окон; общий набор индексов для IF/AE/гибрида")


if __name__ == "__main__":
    main()
