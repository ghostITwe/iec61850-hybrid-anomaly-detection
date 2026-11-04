"""Шаг 4. Autoencoder: обучение на train, ранняя остановка по val-норме, оценки (MSE реконструкции)."""
import argparse
import pandas as pd, torch
from iecad.config import load_config
from iecad.prep import load_matrix
from iecad.models import fit_ae, score_ae


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    cfg = load_config(a.config)
    feats, X, cols = load_matrix(cfg)
    tr = (feats.split == "train").to_numpy(); va = (feats.split == "val").to_numpy()
    assert va.any(), "нет валидационных окон"
    model, hist, dev = fit_ae(X[tr], X[va], cfg, cfg["seed"])
    torch.save(model.state_dict(), cfg["paths"]["models"] / "autoencoder.pt")
    hist.to_csv(cfg["paths"]["reports"] / "ae_loss.csv", index=False)
    pd.DataFrame({"uid": feats.uid, "score_ae": score_ae(model, X)}).to_parquet(cfg["paths"]["processed"] / "scores_ae.parquet", index=False)
    print(f"AE ({dev}): эпох {len(hist)}, лучший val_loss {hist.val_loss.min():.4f}")


if __name__ == "__main__":
    main()
