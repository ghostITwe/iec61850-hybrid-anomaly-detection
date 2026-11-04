"""pcapng -> таблица кадров (parquet) через tshark.

Имена полей проверены по `tshark -G fields` (Wireshark 4.2): sv.smpCnt, sv.meas_value,
goose.gocbRef, goose.stNum, goose.sqNum. Идентификаторы хостов (MAC/IP) НЕ извлекаются:
они анонимизированы и могли бы стать утечкой метки.
"""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ["frame.number", "frame.time_epoch", "frame.len", "vlan.id", "ip.proto", "ipv6.nxt",
          "arp.opcode", "sv.smpCnt", "sv.meas_value", "goose.gocbRef", "goose.stNum", "goose.sqNum"]
CLS = {"other": 0, "sv": 1, "goose": 2, "ipv4": 3, "ipv6": 4, "arp": 5}
CLS_NAMES = list(CLS)


def run_tshark(pcap: str | Path, tshark: str = "tshark") -> pd.DataFrame:
    cmd = [tshark, "-r", str(pcap), "-n", "-o", "sv.decode_data_as_phsmeas:TRUE", "-T", "fields", "-E", "separator=/t",
           "-E", "occurrence=a", "-E", "aggregator=;", "-E", "header=n"]
    for f in FIELDS:
        cmd += ["-e", f]
    with tempfile.NamedTemporaryFile("w+", suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name
        proc = subprocess.run(cmd, stdout=tmp, stderr=subprocess.PIPE, text=True)
    try:
        df = pd.read_csv(tmp_path, sep="\t", header=None, names=FIELDS, dtype=str,
                         quoting=3, engine="c")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=FIELDS)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if df.empty:
        raise RuntimeError(f"tshark вернул 0 кадров для {pcap}: {proc.stderr[-500:]}")
    return df


def parse_frames(df: pd.DataFrame, sv_modulus: int = 4800) -> pd.DataFrame:
    ts = df["frame.time_epoch"].str.split(".", n=1, expand=True)
    sec = ts[0].astype("int64")
    ns = ts[1].fillna("0").str.pad(9, side="right", fillchar="0").str[:9].astype("int64")
    t_ns = (sec * 1_000_000_000 + ns).to_numpy()
    fno = df["frame.number"].astype("int64").to_numpy()
    order = np.argsort(t_ns, kind="stable")          # окна требуют монотонного времени
    df = df.iloc[order].reset_index(drop=True)
    t_ns, fno = t_ns[order], fno[order]
    n = len(df)

    is_sv = df["sv.smpCnt"].notna().to_numpy()
    is_goose = df["goose.stNum"].notna().to_numpy() | df["goose.gocbRef"].notna().to_numpy()
    cls = np.zeros(n, np.int8)
    cls[df["arp.opcode"].notna().to_numpy()] = CLS["arp"]
    cls[df["ipv6.nxt"].notna().to_numpy()] = CLS["ipv6"]
    cls[df["ip.proto"].notna().to_numpy()] = CLS["ipv4"]
    cls[is_goose] = CLS["goose"]
    cls[is_sv] = CLS["sv"]

    vals = np.full((n, 8), np.nan, np.float32)
    if is_sv.any():
        sub = df.loc[is_sv, "sv.meas_value"].str.split(";", expand=True)
        sub = sub.reindex(columns=range(8)).apply(pd.to_numeric, errors="coerce")
        vals[is_sv] = sub.to_numpy(np.float32)

    smp = pd.to_numeric(df["sv.smpCnt"], errors="coerce").to_numpy(float)
    gcode = np.full(n, -1, np.int32)
    gmask = df["goose.gocbRef"].notna().to_numpy()
    if gmask.any():
        gcode[gmask] = pd.factorize(df.loc[gmask, "goose.gocbRef"])[0]
    st = pd.to_numeric(df["goose.stNum"], errors="coerce").to_numpy(float)
    sq = pd.to_numeric(df["goose.sqNum"], errors="coerce").to_numpy(float)

    sv_iat = np.full(n, np.nan, np.float32)
    sv_dsmp = np.full(n, np.nan, np.float32)
    idx = np.flatnonzero(is_sv)
    if len(idx) > 1:
        sv_iat[idx[1:]] = np.diff(t_ns[idx]) / 1e3                 # мкс
        sv_dsmp[idx[1:]] = (smp[idx[1:]] - smp[idx[:-1]]) % sv_modulus
    g_dst = np.full(n, np.nan, np.float32)
    g_dsq = np.full(n, np.nan, np.float32)
    g_dt = np.full(n, np.nan, np.float32)
    for g in np.unique(gcode[gcode >= 0]):
        gi = np.flatnonzero(gcode == g)
        if len(gi) > 1:
            g_dst[gi[1:]] = np.diff(st[gi])
            g_dsq[gi[1:]] = np.diff(sq[gi])
            g_dt[gi[1:]] = np.diff(t_ns[gi]) / 1e9                 # с

    out = pd.DataFrame({
        "frame_number": fno.astype(np.int64), "t_ns": t_ns,
        "len": df["frame.len"].astype("int32").to_numpy(), "cls": cls,
        "vlan": df["vlan.id"].notna().to_numpy(), "gocb": gcode,
        "sv_iat_us": sv_iat, "sv_dsmp": sv_dsmp, "g_dst": g_dst, "g_dsq": g_dsq, "g_dt": g_dt})
    for k in range(8):
        out[f"ch{k}"] = vals[:, k]
    return out


def extract_to_parquet(pcap: str | Path, out: str | Path, tshark: str = "tshark",
                       sv_modulus: int = 4800) -> int:
    fr = parse_frames(run_tshark(pcap, tshark), sv_modulus)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(out, index=False)
    return len(fr)
