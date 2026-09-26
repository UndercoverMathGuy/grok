"""Trainer (core/train.py): determinism, batched == solo, float32 vs float64, provenance, NORMS guards."""
import csv
import json
import os

import numpy as np
import pytest
import torch

from core import train

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "data", "test_reports")

BASE = dict(model="product", n=2, p=7, q=2.0, width=12, lam=1e-3, lr=1.0, steps=600, seed=5, log_every=100,
            init_scale=1.0, measure_ablation=False)


def _run(tmp, name, **kw):
    cfg = dict(BASE)
    cfg.update(kw)
    out = str(tmp / name)
    s = train.train_batch([{"out_dir": out, "run_id": name, "config": cfg}], verbose=False)
    return out, s[0]


def _final(d):
    x = torch.load(os.path.join(d, "final.pt"))
    return x["U"], x["W"]


def _log(d):
    with open(os.path.join(d, "log.csv")) as f:
        return [{k: v for k, v in r.items() if k != "wall_s"} for r in csv.DictReader(f)]


def test_determinism_fixed_seed(tmp_path):
    a, _ = _run(tmp_path, "a")
    b, _ = _run(tmp_path, "b")
    Ua, Wa = _final(a)
    Ub, Wb = _final(b)
    assert torch.equal(Ua, Ub) and torch.equal(Wa, Wb)
    assert _log(a) == _log(b)
    c, _ = _run(tmp_path, "c", seed=6)
    assert not torch.equal(_final(c)[0], Ua)


def test_batched_equals_solo(tmp_path):
    variants = [dict(seed=1, lr=1.0, lam=1e-3, q=2.0), dict(seed=2, lr=0.5, lam=3e-3, q=2.5),
                dict(seed=3, lr=0.7, lam=1e-3, q=3.0, momentum=0.5)]
    runs = []
    for i, v in enumerate(variants):
        cfg = dict(BASE)
        cfg.update(v)
        runs.append({"out_dir": str(tmp_path / f"batch_{i}"), "run_id": f"batch_{i}", "config": cfg})
    train.train_batch(runs, verbose=False)
    report = {"device": "cpu", "dtype": "float32", "runs": []}
    for i, v in enumerate(variants):
        solo, _ = _run(tmp_path, f"solo_{i}", **v)
        Us, Ws = _final(solo)
        Ub, Wb = _final(str(tmp_path / f"batch_{i}"))
        du = (Us.double() - Ub.double()).abs().max().item()
        dw = (Ws.double() - Wb.double()).abs().max().item()
        scale = max(Us.abs().max().item(), Ws.abs().max().item())
        ls, lb = _log(solo), _log(str(tmp_path / f"batch_{i}"))
        gs, gb = float(ls[-1]["gamma"]), float(lb[-1]["gamma"])
        report["runs"].append({"variant": v, "bit_identical": bool(du == 0 and dw == 0),
                               "max_abs_weight_diff": max(du, dw), "rel_weight_diff": max(du, dw) / scale,
                               "gamma_solo": gs, "gamma_batched": gb, "gamma_rel_diff": abs(gs - gb) / abs(gs)})
        assert max(du, dw) / scale < 1e-5
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "batched_vs_solo_cpu.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("batched vs solo (cpu float32):", json.dumps(report))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_batched_equals_solo_cuda(tmp_path):
    variants = [dict(seed=1, lr=1.0), dict(seed=2, lr=0.5, q=2.5), dict(seed=3, lr=0.7, lam=3e-3)]
    runs = [{"out_dir": str(tmp_path / f"b{i}"), "run_id": f"b{i}", "config": {**BASE, **v, "device": "cuda"}}
            for i, v in enumerate(variants)]
    train.train_batch(runs, verbose=False)
    rep = []
    for i, v in enumerate(variants):
        solo, _ = _run(tmp_path, f"s{i}", **v, device="cuda")
        Us, Ws = _final(solo)
        Ub, Wb = _final(str(tmp_path / f"b{i}"))
        d = max((Us - Ub).abs().max().item(), (Ws - Wb).abs().max().item())
        rep.append({"variant": v, "max_abs_weight_diff": d, "bit_identical": d == 0})
        assert d / max(Us.abs().max().item(), 1e-30) < 1e-4
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "batched_vs_solo_cuda.json"), "w") as f:
        json.dump(rep, f, indent=1)


def test_float32_vs_float64_gap(tmp_path):
    """Same init (float32-rounded) trained in float32 and float64. The gap is REPORTED, not tuned away."""
    rep = []
    for kw in [dict(), dict(lam=1e-4, steps=2000, lr=2.0), dict(q=3.0, lam=1e-3, lr=1.0)]:
        a, sa = _run(tmp_path, f"f32_{len(rep)}", dtype="float32", **kw)
        b, sb = _run(tmp_path, f"f64_{len(rep)}", dtype="float64", **kw)
        la, lb = _log(a), _log(b)
        Ua, Wa = _final(a)
        Ub, Wb = _final(b)
        dw = max((Ua.double() - Ub).abs().max().item(), (Wa.double() - Wb).abs().max().item())
        ga, gb = float(la[-1]["gamma"]), float(lb[-1]["gamma"])
        rep.append({"config": kw, "gamma_f32": ga, "gamma_f64": gb, "gamma_rel_gap": abs(ga - gb) / abs(gb),
                    "ce64_f32": float(la[-1]["ce64"]), "ce64_f64": float(lb[-1]["ce64"]),
                    "max_abs_weight_diff": dw, "weight_scale": Ub.abs().max().item(),
                    "grad_rel_f32": float(la[-1]["grad_rel"]), "grad_rel_f64": float(lb[-1]["grad_rel"])})
        assert abs(ga - gb) / abs(gb) < 1e-2  # sanity bound only; the measured gap is in the report
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "fp32_vs_fp64.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("float32 vs float64:", json.dumps(rep))


def test_run_directory_provenance_and_immutability(tmp_path):
    d, s = _run(tmp_path, "prov")
    assert set(os.listdir(d)) == {"config.json", "log.csv", "final.pt", "metrics.json"}
    conf = json.load(open(os.path.join(d, "config.json")))
    for k in ("model", "n", "nu", "p", "q", "s", "width", "lam", "lam_schedule", "optimizer", "lr", "lr_schedule",
              "momentum", "steps", "init_scheme", "init_scale", "seed", "dtype", "device", "code_sha256",
              "environment", "precision_flags", "start_time", "end_time"):
        assert k in conf, k
    assert conf["precision_flags"]["torch.backends.cuda.matmul.allow_tf32"] is False
    assert conf["precision_flags"]["float32_matmul_precision"] == "highest"
    assert "core/train.py" in conf["code_sha256"]
    met = json.load(open(os.path.join(d, "metrics.json")))
    assert met["status"] in ("ok", "not_converged") and "measurement" in met
    assert "agreement" not in met  # tuning default: no prediction scoring
    with open(os.path.join(d, "log.csv")) as f:
        header = f.readline().strip().split(",")
    for k in ("step", "ce", "penalty", "total", "grad_norm", "margin", "gamma", "lr"):
        assert k in header
    assert not os.access(os.path.join(d, "final.pt"), os.W_OK)  # read-only
    with pytest.raises(FileExistsError):
        _run(tmp_path, "prov")


def test_norms_guards(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, "adam", optimizer="adam")
    with pytest.raises(ValueError):
        _run(tmp_path, "qbig", q=3.5)  # q > nu = 3
    with pytest.raises(ValueError):
        _run(tmp_path, "qsmall", q=1.5)


def test_schedules():
    cfg = dict(BASE, steps=1000, lam=1e-4, lam_schedule={"kind": "geometric", "lam0": 1e-2, "until_frac": 0.5},
               lr_schedule={"kind": "cosine", "min_frac": 0.1, "warmup": 10})
    assert abs(train.lam_at(cfg, 0) - 1e-2) < 1e-15 and abs(train.lam_at(cfg, 500) - 1e-4) < 1e-15
    assert abs(train.lam_at(cfg, 250) - 1e-3) < 1e-12 and train.lam_at(cfg, 900) == 1e-4
    assert abs(train.lr_at(cfg, 0) - 0.1) < 1e-12 and abs(train.lr_at(cfg, 1000) - 0.1) < 1e-12
    with pytest.raises(ValueError):
        train.lr_at(dict(cfg, lr_schedule={"kind": "adaptive"}), 5)


def test_init_relabel_equivariance(tmp_path):
    import math
    """E6: training from the relabelled init (a -> u a) gives the relabelled network: A'_j = A_{fold(u j)}
    and f'(u x)[u c] = f(x)[c] (float32, so up to rounding)."""
    import numpy as np
    from core import measure as MS
    from core import programme as PG
    p, n, u = 7, 2, 3
    base = {"model": "product", "n": n, "p": p, "q": 2.0, "width": 12, "lam": 1e-3, "lr": 1.0, "steps": 300,
            "seed": 4, "log_every": 100, "init_scale": math.sqrt(p), "measure_ablation": False}
    runs = [{"out_dir": str(tmp_path / "a"), "run_id": "a", "config": dict(base)},
            {"out_dir": str(tmp_path / "b"), "run_id": "b", "config": {**base, "init_relabel_u": u}}]
    train.train_batch(runs, verbose=False)
    Ua, Wa = train.load_final(str(tmp_path / "a"))
    Ub, Wb = train.load_final(str(tmp_path / "b"))
    Ta, Tb = MS.table("product", Ua, Wa), MS.table("product", Ub, Wb)
    idx = np.arange(p)
    Tb_back = Tb[np.ix_((u * idx) % p, (u * idx) % p, (u * idx) % p)]
    assert np.max(np.abs(Tb_back - Ta)) <= 1e-4 * np.max(np.abs(Ta))
    Aa, Ab = MS.spectrum(Ta)["A"], MS.spectrum(Tb)["A"]
    js = np.arange(1, (p - 1) // 2 + 1)
    assert np.allclose(Ab, Aa[PG.fold(u * js, p) - 1], rtol=1e-4, atol=1e-6 * Aa.max())
