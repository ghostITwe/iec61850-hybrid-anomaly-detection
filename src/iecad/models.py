"""Isolation Forest (scikit-learn) и полносвязный Autoencoder (PyTorch). Метки в обучении не используются."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_if(Xtr: np.ndarray, cfg: dict, seed: int) -> IsolationForest:
    c = cfg["isolation_forest"]
    return IsolationForest(n_estimators=c["n_estimators"], max_samples=c["max_samples"],
                           random_state=seed, n_jobs=-1).fit(Xtr)


def score_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)          # больше = аномальнее


def _build_ae(d: int, hidden: list[int], latent: int):
    import torch.nn as nn
    enc, prev = [], d
    for h in hidden:
        enc += [nn.Linear(prev, h), nn.GELU()]; prev = h
    enc += [nn.Linear(prev, latent)]
    dec, prev = [], latent
    for h in reversed(hidden):
        dec += [nn.Linear(prev, h), nn.GELU()]; prev = h
    dec += [nn.Linear(prev, d)]
    return nn.Sequential(*enc, *dec)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_ae(Xtr: np.ndarray, Xva: np.ndarray, cfg: dict, seed: int):
    import torch
    c = cfg["autoencoder"]
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = get_device()
    model = _build_ae(Xtr.shape[1], c["hidden"], c["latent"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    lossf = torch.nn.MSELoss()
    tr = torch.from_numpy(Xtr).to(dev); va = torch.from_numpy(Xva).to(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(1, c["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(tr), generator=g).to(dev)
        tot = 0.0
        for i in range(0, len(tr), c["batch_size"]):
            xb = tr[perm[i:i + c["batch_size"]]]
            opt.zero_grad(); loss = lossf(model(xb), xb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(va), va).item()
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_loss": vl})
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), str(dev)


def score_ae(model, X: np.ndarray) -> np.ndarray:
    import torch
    dev = next(model.parameters()).device
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            xb = torch.from_numpy(X[i:i + 8192]).to(dev)
            out.append(((model(xb) - xb) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out)
