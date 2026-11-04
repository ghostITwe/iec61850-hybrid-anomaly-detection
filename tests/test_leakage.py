"""Тесты на утечки: разбиение по прогонам, scaler только по train, пороги/ранги только по val, запрет признаков."""
import json
import numpy as np, pandas as pd
from iecad.config import load_config
from iecad.splits import assign_splits
from iecad.scoring import rank_with_tail, threshold_from_val
from iecad.windows import feature_columns, FORBIDDEN_SUBSTR


def _cat():
    rows = [dict(capture=f"B/b{i}/x", run_id=f"B/b{i}", scenario="Baseline", family="Baseline", group="normal_pool") for i in range(1, 6)]
    rows += [dict(capture=f"A/a{i}/x", run_id=f"A/a{i}", scenario="A/s", family="GOOSE Injection", group="attack") for i in range(1, 4)]
    rows += [dict(capture="F/f1/x", run_id="F/f1", scenario="Fault/n", family="Fault", group="physical_fault")]
    return pd.DataFrame(rows)


def test_split_by_runs_and_groups():
    cfg = load_config()
    m = assign_splits(_cat(), cfg)
    assert m.run_id.is_unique                                  # прогон попадает ровно в одну выборку
    norm = m[m.group == "normal_pool"]
    assert sorted(norm.split.value_counts().to_dict().items()) == [("test_normal", 1), ("train", 3), ("val", 1)]
    assert set(m[m.group == "attack"].split) == {"test_attack"}   # атаки никогда не в train/val
    assert set(m[m.group == "physical_fault"].split) == {"eval_only"}
    assert m.equals(assign_splits(_cat(), cfg))               # детерминированность


def test_resource_exhaustion_is_not_normal():
    cfg = load_config()
    assert all(not s.startswith("Resource") for s in cfg["normal_pool"])


def test_ranks_and_threshold_depend_only_on_val():
    rng = np.random.default_rng(0)
    val = rng.normal(size=200); a = rng.normal(size=50); b = rng.normal(size=50) + 5
    assert np.allclose(rank_with_tail(val, a), rank_with_tail(val, a))
    r_val = rank_with_tail(val, val)
    assert np.allclose(r_val, rank_with_tail(val, np.concatenate([val, b]))[:200])  # тест не влияет на шкалу
    assert threshold_from_val(val, 0.99) == threshold_from_val(val.copy(), 0.99)
    assert (np.diff(rank_with_tail(val, np.sort(np.concatenate([val, b])))) >= 0).all()  # монотонность


def test_no_identity_or_label_features():
    cfg = load_config()
    for fs in ("protocol", "protocol_content"):
        cfg["window"]["feature_set"] = fs
        cols = feature_columns(cfg)
        assert not [c for c in cols if any(s in c.lower() for s in FORBIDDEN_SUBSTR)]
        assert "label" not in cols and "n_pos_frames" not in cols


def test_scaler_fit_on_train_only(tmp_path):
    cfg = load_config()
    cfg["paths"]["processed"] = tmp_path; cfg["paths"]["models"] = tmp_path
    cols = feature_columns(cfg); n = 30
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(n, len(cols))) ** 2, columns=cols)
    df["split"] = ["train"] * 10 + ["val"] * 10 + ["test_attack"] * 10
    df.loc[20:, cols[0]] += 1e4                                  # сильный сдвиг в тесте
    df.to_parquet(tmp_path / "features.parquet")
    from iecad.prep import load_matrix
    _, Z, _ = load_matrix(cfg)
    sc = json.loads((tmp_path / "scaler.json").read_text())
    assert sc["fit_on"] == "split==train"
    tr = Z[:10]
    assert np.abs(tr.mean(0)).max() < 0.5                        # train центрирован по train
    assert np.abs(Z[20:, 0]).max() <= cfg["preprocess"]["clip_z"] + 1e-6   # сдвиг в тесте не меняет масштаб
