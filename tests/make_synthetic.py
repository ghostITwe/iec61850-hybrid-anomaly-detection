"""Синтетический мини-датасет (структура как у TU Delft v2) для проверки конвейера БЕЗ реальных данных.
Содержимое НЕ имитирует реальные распределения; тест проверяет только корректность кода (утечки, метки, разбиение)."""
import struct, sys
from pathlib import Path
import numpy as np
import pandas as pd


def tlv(tag, val: bytes) -> bytes:
    n = len(val)
    return bytes([tag]) + (bytes([n]) if n < 128 else bytes([0x81, n]) if n < 256 else bytes([0x82, n >> 8, n & 255])) + val


def mac(s): return bytes(int(x, 16) for x in s.split(":"))


def sv_frame(smp, vals, vlan=False):
    asdu = tlv(0x80, b"0000MU0101") + tlv(0x82, struct.pack(">H", smp)) + tlv(0x83, struct.pack(">I", 1)) + tlv(0x85, b"\x01")
    asdu += tlv(0x87, b"".join(struct.pack(">iI", int(v), 0) for v in vals))
    pdu = tlv(0x60, tlv(0x80, b"\x01") + tlv(0xA2, tlv(0x30, asdu)))
    body = struct.pack(">HHHH", 0x4001, 8 + len(pdu), 0, 0) + pdu
    return mac("01:0c:cd:04:00:00") + mac("00:50:c2:f1:f1:f1") + b"\x88\xba" + body


def goose_frame(gocb, st, sq, vlan=False, dst="01:0c:cd:01:00:00", src="12:c5:b1:f1:f1:f2"):
    p = tlv(0x80, gocb.encode()) + tlv(0x81, struct.pack(">H", 3000)) + tlv(0x82, b"IEDA_SVSystem/LLN0$GOOSE1")
    p += tlv(0x83, b"IDC/GOOSE1") + tlv(0x84, b"\x00" * 8) + tlv(0x85, struct.pack(">I", st)) + tlv(0x86, struct.pack(">I", sq))
    p += tlv(0x87, b"\x00") + tlv(0x88, b"\x01") + tlv(0x89, b"\x00") + tlv(0x8A, b"\x01") + tlv(0xAB, tlv(0x83, b"\x00"))
    pdu = tlv(0x61, p)
    body = struct.pack(">HHHH", 0x0001, 8 + len(pdu), 0, 0) + pdu
    eth = mac(dst) + mac(src)
    return eth + (b"\x81\x00\x80\x00\x88\xb8" if vlan else b"\x88\xb8") + body


def udp_frame(rng, size=80):
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + 8 + size, 0, 0, 64, 17, 0, bytes([192, 168, 0, 20]), bytes([192, 168, 0, 21]))
    return mac("02:00:00:00:00:02") + mac("02:00:00:00:00:03") + b"\x08\x00" + ip + struct.pack(">HHHH", 5000, 53, 8 + size, 0) + bytes(size)


def tcp_syn(src, dst):
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 40, 0, 0, 64, 6, 0, bytes([192, 168, 0, src]), bytes([192, 168, 0, dst]))
    return mac("4c:5c:06:ef:5e:c9") + mac("f5:a0:8d:e8:8f:2d") + b"\x08\x00" + ip + struct.pack(">HHIIBBHHH", 40000, 22, 0, 0, 0x50, 2, 1024, 0, 0)


def write_pcapng(path, pkts):
    def blk(t, body):
        n = 12 + len(body); return struct.pack("<II", t, n) + body + struct.pack("<I", n)
    out = blk(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    out += blk(1, struct.pack("<HHI", 1, 0, 0) + struct.pack("<HH", 9, 1) + bytes([9, 0, 0, 0]) + struct.pack("<HH", 0, 0))
    for t_ns, data in pkts:
        pad = (-len(data)) % 4
        out += blk(6, struct.pack("<IIIII", 0, t_ns >> 32, t_ns & 0xFFFFFFFF, len(data), len(data)) + data + bytes(pad))
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(out)


BASE_T = 1_720_000_000 * 10**9


def make_capture(dur, seed, kind=None, fault=False):
    """kind: None | goose | svhigh | recon -> список (t_ns, bytes, label)."""
    rng = np.random.default_rng(seed); ev = []
    n = int(dur * 4800)
    for k in range(n):
        t = k / 4800 + rng.normal(0, 5e-6)
        amp = 4.0 if (fault and 3 <= t < 5) else 1.0
        ph = 2 * np.pi * 60 * k / 4800
        vals = [4e5 * amp * np.sin(ph + o) for o in (0, -2.094, 2.094)] + [0] + [5.7e6 * np.sin(ph + o) for o in (0, -2.094, 2.094)] + [0]
        ev.append((t, sv_frame(k % 4800, vals), 0))
    for k in range(int(dur)):
        ev.append((k + 0.1 + rng.normal(0, 1e-3), goose_frame("IEDA_SVSystem/LLN0$GO$gcb01", 5, k), 0))
    for k in range(int(dur / 1.5)):
        ev.append((k * 1.5 + 0.4 + rng.normal(0, 1e-3), goose_frame("vied_protHTA/LLN0$GO$EVENT_GSE_TRIP_TG", 3, k, vlan=True), 0))
    for k in range(int(dur * 10)):
        ev.append((k / 10 + rng.uniform(0, .1), udp_frame(rng), 0))
    if kind == "goose":
        for t in (3.0, 5.0):
            ev.append((t, goose_frame("IEDA_SVSystem/LLN0$GO$gcb01", 773, 0), 1))
    elif kind == "svhigh":
        for k in range(int(3 * 4800), int(5 * 4800)):
            ev.append((k / 4800 + 0.0147, sv_frame((k + 70) % 4800, [9999999] * 8), 1))
    elif kind == "recon":
        for k in range(int(dur * 0.5 * 300)):
            ev.append((2 + k / 300, tcp_syn(11, 10 + k % 3), 1))
    ev.sort(key=lambda x: x[0])
    t0 = min(e[0] for e in ev)
    return [(BASE_T + int((t - t0) * 1e9), b, l) for t, b, l in ev]


def save(root: Path, rel: str, stem: str, pk, labeled: bool):
    d = root / "Scenarios" / rel
    write_pcapng(d / f"{stem}.pcapng", [(t, b) for t, b, _ in pk])
    if labeled:
        pd.DataFrame({"frame_number": range(1, len(pk) + 1), "label": [l for *_, l in pk]}).to_csv(d / f"{stem}_labels.csv", index=False)


def build(root: Path, dur=8):
    root = Path(root); s = 0
    for i in range(1, 6):
        s += 1; save(root, f"Baseline/baseline{i}", f"baseline{i}_anon", make_capture(dur, s), False)
    for i in range(1, 3):
        s += 1; save(root, f"Fault/faults_no_reclosure/fault{i}", f"fault{i}_anon", make_capture(dur, s, fault=True), False)
    s += 1; save(root, "Resource Exhaustion/baseline_priority_a/re1", "re1_anon", make_capture(dur, s), False)
    for i in range(1, 4):
        s += 1; save(root, f"GOOSE Injection/IEDSpoofingBaseline/ied_injection_baseline{i}", f"ied_injection_baseline{i}_anon", make_capture(dur, s, "goose"), True)
    s += 1; pk = make_capture(dur, s, "recon")
    for v in ("vied", "attacker"):
        save(root, "Reconnaissance/nmap_normal", f"nmap_normal_{v}_anon", pk, True)
    s += 1; pk = make_capture(dur, s, "svhigh")
    for v in ("vied", "attacker"):
        save(root, "SV Injection/HighValues", f"HighValues_{v}_anon", pk, True)
    d = root / "Scenarios" / "Baseline" / "baseline1"
    pd.DataFrame({"time": np.arange(0, .16, .0016), "I1_a": 0.0, "I1_b": 0.0, "I1_c": 0.0, "FLT_input": 0.0, "CB1_state": 1.0,
                  "ied_trip_signal": 0.0, "vied_trip_signal": 0.0, "V1_a": 1.0, "V1_b": 1.0, "V1_c": 1.0,
                  "formatted_datetime": "2024-07-05T12:59:34.000000+00:00"}).to_csv(d / "baseline1_rscad.csv", index=False)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
    print("synthetic dataset ready:", sys.argv[1])
