"""Positive controls and null baselines for every scored cell (NORMS E17). Runs ON A POD (CPU).
Usage: python experiments/common/controls.py EXP [EXP ...]  -> experiments/EXP/analysis/controls.json
- positive: nu-block product network with the predicted y* (canonical optimal vertex; product), or the Morwani dense
  power network (power nu = 3, q = 3), measured by core.measure.full_measurement (ablation included) and scored by
  core.metrics.score_run against the same prediction as the trained runs;
- null (a): the step-0 random-init network of seed 0 (core.models.init_params with the frozen recipe's init), scored
  the same way;
- null (b): 1000 random allocations (support of size K* uniformly at random, shares uniform on the simplex), scored
  by core.metrics.score_amplitudes."""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("CTRL_THREADS", "8")))
from core import construct as C, measure as MS, metrics as MT, models as M


def cells_of(exp):
    P = json.load(open(os.path.join(ROOT, "experiments", exp, "predictions.json")))
    cells = P.get("cells") or [P["cell"]]
    out = []
    for c in cells:
        model = c.get("model", "product")
        y_all = c.get("y_all") or [c["y_star"]]
        y0 = c.get("y_star_canonical") or c.get("y_star")
        out.append({"model": model, "nu": c["nu"], "p": c["p"], "q": c["q"], "s": c["s"], "Gamma": c["Gamma"],
                    "y_all": y_all, "y0": y0, "K_star": c["K_star"], "label": c["label"]})
    return out


def summarize(scores):
    keys = {"tv": lambda s: s["share"]["tv"], "cos_amp": lambda s: s["amp"]["cosine"],
            "thr10_exact": lambda s: float(s["counts"]["thr_10"]["exact"]), "thr10_meas": lambda s: s["counts"]["thr_10"]["meas"],
            "r2_id_amp": lambda s: s["amp"]["r2_identity"], "jacc10": lambda s: s["jaccard_thr_10"]}
    return {k: {"median": float(np.nanmedian([f(s) for s in scores])), "mean": float(np.nanmean([f(s) for s in scores])),
                "q05": float(np.nanpercentile([f(s) for s in scores], 5)), "q95": float(np.nanpercentile([f(s) for s in scores], 95))}
            for k, f in keys.items()}


def one(c, rng):
    p, nu, q, s = c["p"], c["nu"], c["q"], c["s"]
    n = nu - 1
    pred = {"gamma": c["Gamma"], "y_all": np.asarray(c["y_all"]), "label": c["label"], "K_all": [c["K_star"]]}
    t0 = time.time()
    if c["model"] == "power":
        U, W = C.morwani_dense(p, np.asarray(c["y0"]))
    else:
        U, W = C.nu_block(p, n, np.asarray(c["y0"]))
    mpos = MS.full_measurement(c["model"], U, W, q, with_ablation=(c["model"] == "product"))
    spos = MT.score_run(mpos, pred)
    h = (p - 1) // 2
    U0, W0 = M.init_params(c["model"], n, 2 * nu * h, p, 0, math.sqrt(p), "normal")
    mnull = MS.full_measurement(c["model"], U0, W0, q, with_ablation=False)
    snull = MT.score_run(mnull, pred)
    rnd = []
    K = c["K_star"]
    for _ in range(1000):
        y = np.zeros(h)
        sup = rng.choice(h, K, replace=False)
        w = rng.dirichlet(np.ones(K))
        y[sup] = w ** s
        rnd.append(MT.score_amplitudes(y, s, pred["y_all"]))
    keep = lambda m: {k: m[k] for k in ("gamma", "margin", "A", "counts", "energy_frac", "ablation") if k in m}
    return {"cell": {k: c[k] for k in ("model", "nu", "p", "q", "s", "K_star", "label", "Gamma")},
            "positive": {"measurement": keep(mpos), "score": spos}, "null_init": {"measurement": keep(mnull), "score": snull},
            "null_random_allocation": summarize(rnd), "seconds": time.time() - t0}


def default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    return str(o)


if __name__ == "__main__":
    for exp in sys.argv[1:]:
        rng = np.random.default_rng(12345)
        res = []
        for c in cells_of(exp):
            r = one(c, rng); res.append(r)
            sp = r["positive"]["score"]
            print(exp, c["model"], c["nu"], c["p"], round(c["s"], 4), "pos: K", sp["counts"]["thr_10"]["meas"], "/", c["K_star"],
                  "tv", f"{sp['share']['tv']:.1e}", "g/IF", f"{sp['gamma_over_ceiling']['class_IF']:.6f}",
                  "| null tv", f"{r['null_init']['score']['share']['tv']:.2f}", f"{r['seconds']:.1f}s", flush=True)
        os.makedirs(os.path.join(ROOT, "experiments", exp, "analysis"), exist_ok=True)
        json.dump({"experiment": exp, "script": "experiments/common/controls.py", "cells": res},
                  open(os.path.join(ROOT, "experiments", exp, "analysis", "controls.json"), "w"), default=default)
