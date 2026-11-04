"""Таблица кадров -> оконные признаки + метка окна.

Допустимые признаки: счётчики/объёмы по классам трафика, тайминги SV (µs), разрывы smpCnt,
изменения stNum/sqNum (ТОЛЬКО разности), доля кадров с 802.1Q; опционально RMS из SV (feature_set=protocol_content).
Запрещены: MAC/IP, сырые stNum/sqNum, любые значения меток.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .frames import CLS_NAMES

PROTOCOL_COLS = (
    ["n_frames", "bytes_total"]
    + [f"n_{c}" for c in CLS_NAMES] + [f"bytes_{c}" for c in CLS_NAMES]
    + ["sv_iat_mean_us", "sv_iat_std_us", "sv_iat_min_us", "sv_iat_max_us",
       "sv_n_smp_anom", "sv_n_smp_dup", "sv_vlan_frac",
       "g_n_gocb", "g_n_st_change", "g_max_abs_dst", "g_n_sq_decr", "g_min_dt_s", "g_max_dt_s", "g_vlan_frac"]
)
CONTENT_COLS = ["sv_i_rms_a", "sv_i_rms_b", "sv_i_rms_c", "sv_v_rms_a", "sv_v_rms_b", "sv_v_rms_c",
                "sv_i_absmax", "sv_v_absmax"]
FORBIDDEN_SUBSTR = ("mac", "ip_addr", "label", "stnum_raw", "sqnum_raw")


def feature_columns(cfg: dict) -> list[str]:
    cols = list(PROTOCOL_COLS)
    if cfg["window"]["feature_set"] == "protocol_content":
        cols += CONTENT_COLS
    return cols


def _z(a):  # пустой массив -> 0
    return a if a.size else np.zeros(1)


def window_features(fr: pd.DataFrame, labels: np.ndarray | None, cfg: dict) -> pd.DataFrame:
    L, S = cfg["window"]["length_s"], cfg["window"]["stride_s"]
    content = cfg["window"]["feature_set"] == "protocol_content"
    t = (fr["t_ns"].to_numpy() - fr["t_ns"].iloc[0]) / 1e9
    T = t[-1]
    starts = np.arange(0.0, max(T - L, 0.0) + 1e-9, S)
    lo = np.searchsorted(t, starts, "left")
    hi = np.searchsorted(t, starts + L, "left")
    cls, ln = fr["cls"].to_numpy(), fr["len"].to_numpy()
    vlan = fr["vlan"].to_numpy()
    iat, dsmp = fr["sv_iat_us"].to_numpy(), fr["sv_dsmp"].to_numpy()
    gc = fr["gocb"].to_numpy()
    gdst, gdsq, gdt = fr["g_dst"].to_numpy(), fr["g_dsq"].to_numpy(), fr["g_dt"].to_numpy()
    ch = fr[[f"ch{k}" for k in range(8)]].to_numpy()
    lab = None
    if labels is not None:
        fno = fr["frame_number"].to_numpy()
        lab = labels[fno - 1]

    rows = []
    for w, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            continue
        c = cls[a:b]
        cnt = np.bincount(c, minlength=6)
        byt = np.bincount(c, weights=ln[a:b], minlength=6)
        r = {"window_idx": w, "t_start_s": float(starts[w]), "n_frames": int(b - a), "bytes_total": float(byt.sum())}
        for k, name in enumerate(CLS_NAMES):
            r[f"n_{name}"] = int(cnt[k]); r[f"bytes_{name}"] = float(byt[k])
        m = c == 1
        x = iat[a:b][m]; x = x[~np.isnan(x)]
        r["sv_iat_mean_us"] = float(x.mean()) if x.size else 0.0
        r["sv_iat_std_us"] = float(x.std()) if x.size else 0.0
        r["sv_iat_min_us"] = float(x.min()) if x.size else 0.0
        r["sv_iat_max_us"] = float(x.max()) if x.size else 0.0
        d = dsmp[a:b][m]; d = d[~np.isnan(d)]
        r["sv_n_smp_anom"] = int((d != 1).sum()); r["sv_n_smp_dup"] = int((d == 0).sum())
        r["sv_vlan_frac"] = float(vlan[a:b][m].mean()) if m.any() else 0.0
        g = c == 2
        r["g_n_gocb"] = int(np.unique(gc[a:b][g & (gc[a:b] >= 0)]).size)
        ds = gdst[a:b][g]; ds = ds[~np.isnan(ds)]
        r["g_n_st_change"] = int((ds != 0).sum())
        r["g_max_abs_dst"] = float(np.abs(ds).max()) if ds.size else 0.0
        dq = gdsq[a:b][g]; dq = dq[~np.isnan(dq)]
        r["g_n_sq_decr"] = int((dq < 0).sum())
        dt = gdt[a:b][g]; dt = dt[~np.isnan(dt)]
        r["g_min_dt_s"] = float(dt.min()) if dt.size else 0.0
        r["g_max_dt_s"] = float(dt.max()) if dt.size else 0.0
        r["g_vlan_frac"] = float(vlan[a:b][g].mean()) if g.any() else 0.0
        if content:
            v = ch[a:b][m]
            if v.shape[0]:
                I = v[:, 0:3] / 1000.0      # 1 mA -> A
                V = v[:, 4:7] / 100.0       # 10 mV -> V
                for j, nm in enumerate("abc"):
                    r[f"sv_i_rms_{nm}"] = float(np.sqrt(np.nanmean(I[:, j] ** 2)))
                    r[f"sv_v_rms_{nm}"] = float(np.sqrt(np.nanmean(V[:, j] ** 2)))
                r["sv_i_absmax"] = float(np.nanmax(np.abs(I))); r["sv_v_absmax"] = float(np.nanmax(np.abs(V)))
            else:
                for nm in CONTENT_COLS:
                    r[nm] = 0.0
        if lab is not None:
            r["n_pos_frames"] = int(lab[a:b].sum()); r["label"] = int(r["n_pos_frames"] > 0)
        else:
            r["n_pos_frames"] = 0; r["label"] = 0
        rows.append(r)
    return pd.DataFrame(rows)


def read_labels(path: str, n_frames_expected: int | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    arr = np.zeros(int(df["frame_number"].max()), np.int8)
    arr[df["frame_number"].to_numpy() - 1] = df["label"].to_numpy()
    if n_frames_expected is not None and len(df) != n_frames_expected:
        raise ValueError(f"{path}: {len(df)} строк меток, но {n_frames_expected} кадров в захвате")
    return arr
