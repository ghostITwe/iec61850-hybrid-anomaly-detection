"""Разбиение по НЕЗАВИСИМЫМ ПРОГОНАМ (не по соседним окнам)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def assign_splits(cat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = cfg["seed"]
    runs = cat.drop_duplicates("run_id")[["run_id", "scenario", "family", "group"]].copy()
    runs["split"] = ""
    sp = cfg["split"]
    for scen, g in runs[runs.group == "normal_pool"].groupby("scenario"):
        ids = sorted(g.run_id)
        rng = np.random.default_rng(seed)
        ids = [ids[i] for i in rng.permutation(len(ids))]
        n = len(ids)
        n_val = max(1, round(sp["val"] * n)) if n >= 3 else 0
        n_test = max(1, round(sp["test"] * n)) if n >= 3 else 0
        n_tr = n - n_val - n_test
        lab = ["train"] * n_tr + ["val"] * n_val + ["test_normal"] * n_test
        for rid, s in zip(ids, lab):
            runs.loc[runs.run_id == rid, "split"] = s
    runs.loc[runs.group == "attack", "split"] = "test_attack"
    runs.loc[runs.group.isin(["physical_fault", "host_stress"]), "split"] = "eval_only"
    runs.loc[runs.split == "", "split"] = "unused"
    return runs.reset_index(drop=True)
