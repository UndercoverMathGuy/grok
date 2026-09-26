"""AUDIT: own positive controls (nu-blocks built from the paper formula, measured with audit/remeasure.py code),
own null baselines (random allocation; step-0 random init regenerated with torch.Generator, checked against log.csv
step-0 margins), and a constant-count null for the count headline."""
import json, math, os, sys, csv
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_audit import *
from remeasure import table

def measure_UW(model, U, W, q):
    n, m, p = U.shape; nu = n + 1
    F = table(model, U, W); a = np.indices((p,) * n).reshape(n, -1); y = a.sum(0) % p
    fy = F[np.arange(len(y)), y]; Fo = F.copy(); Fo[np.arange(len(y)), y] = -np.inf
    h = float((fy - Fo.max(1)).min())
    sq = (U ** 2).sum(axis=(0, 2)) + (W ** 2).sum(1); norm = float((sq ** (q / 2)).sum() ** (1 / q))
    z = (a.sum(0)[:, None] - np.arange(p)[None, :]) % p
    S = np.bincount(z.ravel(), weights=F.ravel(), minlength=p); hh = (p - 1) // 2; j = np.arange(1, hh + 1)
    A = np.abs(2 * p ** (-nu) * (S[None, :] * np.exp(-2j * np.pi * np.outer(j, np.arange(p)) / p)).sum(1))
    # per-offset worst margin, folded
    D = fy[:, None] - F  # f_y - f_c
    delta = (y[:, None] - np.arange(p)[None, :]) % p
    per = np.array([D[delta == d].min() for d in range(1, p)])
    fold = np.minimum(per[:hh], per[::-1][:hh])
    return {"margin": h, "norm": norm, "gamma": h / norm ** nu, "A": A, "per_off_folded": fold}

def nu_block(p, n, y):
    nu = n + 1; a = np.arange(p); U = []; W = []
    for j in np.flatnonzero(y > 1e-15) + 1:
        lam = (2 ** n * y[j - 1] / nu) ** (1 / nu)
        for l in range(nu):
            al = np.pi * l / nu
            U.append([lam * np.cos(2 * np.pi * j * a / p + al)] * n); W.append(lam * np.cos(2 * np.pi * j * a / p + n * al))
    U = np.transpose(np.array(U), (1, 0, 2)); return U, np.array(W)

out = {"positive": []}
idx1, _ = preds("E1"); idx7, _ = preds("E7")
for (e, key) in [("E1", ("product", 3, 23, 1.25)), ("E1", ("product", 3, 31, 1.5)), ("E1", ("product", 4, 17, 2.0)), ("E1", ("product", 5, 11, 1.25)),
                 ("E1", ("product", 5, 13, 1.5)), ("E7", ("product", 4, 37, 1.226)), ("E1", ("product", 4, 23, 1.5))]:
    cell = (idx1 if e == "E1" else idx7)[key]; p = key[2]; nu = key[1]; s = key[3]; q = nu / s
    y = np.asarray(cell.get("y_star_canonical") or cell["y_star"])
    U, W = nu_block(p, nu - 1, y)
    M = measure_UW("product", U, W, q)
    gIF = nu ** (-(nu - q) / q) * Bp(p, nu) * cell["Gamma"]
    covy = cov_matrix(p) @ y; tight = set((np.flatnonzero(covy <= covy.min() * (1 + 1e-9)) + 1).tolist())
    hf = M["per_off_folded"]; near = set((np.flatnonzero(hf <= hf.min() + 1e-9 * abs(hf.min())) + 1).tolist())
    # ablation of each support frequency: project out +-j from every weight vector
    abl_ok = []
    w = shares(M["A"], s)
    for j in np.flatnonzero(y > 0) + 1:
        c_ = np.cos(2 * np.pi * j * np.arange(p) / p); s_ = np.sin(2 * np.pi * j * np.arange(p) / p)
        Bm = np.stack([c_ / np.linalg.norm(c_), s_ / np.linalg.norm(s_)])
        proj = lambda V: V - (V @ Bm.T) @ Bm
        Ma = measure_UW("product", proj(U), proj(W), q)
        rl = 1 - Ma["margin"] / M["margin"]; abl_ok.append(bool(rl >= 1 - (1 - w[j - 1]) ** s - 1e-9))
    out["positive"].append({"exp": e, "cell": key, "gamma_over_IF": M["gamma"] / gIF, "A_max_err": float(np.abs(M["A"] - y).max()),
                            "thr10": thr(M["A"], .1), "K": cell["K_star"], "near_eq_tight": near == tight, "n_tight": len(tight), "ablation_bound_all": all(abl_ok)})
    print(out["positive"][-1])

# ---- init regeneration vs log.csv step 0
def init(n, m, p, seed, scale):
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed)); std = scale / math.sqrt(p)
    U = torch.randn((1, n, m, p), generator=g, dtype=torch.float64) * std
    W = torch.randn((1, m, p), generator=g, dtype=torch.float64) * std
    return U.float().double().numpy()[0], W.float().double().numpy()[0]
chk = []
for o in json.load(open(os.path.join(ROOT, "audit", "sample.json"))):
    d = os.path.join(EX, o["exp"], "runs", o["run_id"]); c = json.load(open(os.path.join(d, "config.json")))
    U, W = init(c["n"], c["width"], c["p"], c["seed"], c["init_scale"])
    if c.get("init_relabel_u", 1) != 1: continue
    M = measure_UW(c["model"], U, W, c["q"])
    row0 = next(csv.DictReader(open(os.path.join(d, "log.csv"))))
    chk.append((o["run_id"], M["margin"], float(row0["margin"]), abs(M["margin"] - float(row0["margin"])) / abs(float(row0["margin"]))))
out["init_regen_vs_log_step0_margin"] = chk
print("init check max rel diff", max(x[3] for x in chk))

# ---- nulls: random allocation TV + eff, random-init (seed 0) TV + eff, per cell of E1; E3 random-init flatness
rng = np.random.default_rng(777)
ra_tv, ra_eff, ri_tv, ri_eff = [], [], [], []
for key, cell in idx1.items():
    _, nu, p, s = key; h = (p - 1) // 2; K = cell["K_star"]; Y = np.asarray(cell["y_all"]); G = cell["Gamma"]
    tvs, effs = [], []
    for _ in range(300):
        y = np.zeros(h); sup = rng.choice(h, K, replace=False); y[sup] = rng.dirichlet(np.ones(K)) ** s
        tvs.append(tv(shares(y, s), shares(Y[align(y, Y, s)], s))); effs.append(prog_value(y, p, s) / G)
    ra_tv.append(np.median(tvs)); ra_eff.append(np.median(effs))
    if p ** (nu - 1) * p <= 2_000_000:
        U, W = init(nu - 1, 2 * nu * h, p, 0, math.sqrt(p)); M = measure_UW("product", U, W, nu / s)
        ri_tv.append(tv(shares(M["A"], s), shares(Y[align(M["A"], Y, s)], s))); ri_eff.append(prog_value(M["A"], p, s) / G)
out["E1_null_random_allocation_median_over_cells"] = {"tv": float(np.median(ra_tv)), "eff": float(np.median(ra_eff))}
out["E1_null_random_init_seed0_median_over_cells"] = {"tv": float(np.median(ri_tv)), "eff": float(np.median(ri_eff)), "n_cells": len(ri_tv)}
print(out["E1_null_random_allocation_median_over_cells"], out["E1_null_random_init_seed0_median_over_cells"])
e3 = []
for (model, nu, p) in [("product", 3, 23), ("product", 3, 59), ("product", 3, 113), ("product", 4, 23), ("product", 5, 11), ("product", 5, 13), ("product", 5, 17), ("power", 3, 23), ("power", 3, 59)]:
    h = (p - 1) // 2; U, W = init(nu - 1, 2 * nu * h, p, 0, math.sqrt(p)); M = measure_UW(model, U, W, nu)
    A = M["A"]; w = A / A.sum()
    e3.append((model, nu, p, thr(A, .1), h, round(float(A.min() / A.max()), 3), round(float(0.5 * np.abs(w - 1 / h).sum()), 3)))
out["E3_random_init_seed0"] = e3; print(e3)

# ---- constant-count null for the E1 count headline (post hoc audit check)
rows = []
for r in runs("E1"):
    if r["status"] != "ok": continue
    rows.append((energy_counts(r["m"]["A"])["K_2pct"], thr(r["m"]["A"], .2), idx1[key_of(r)]["K_star"], r["c"]["n"] + 1, r["c"]["p"]))
K2 = np.array([x[0] for x in rows]); T20 = np.array([x[1] for x in rows]); Ks = np.array([x[2] for x in rows])
best_const = {name: max(float(np.mean(np.abs(v - k) <= 1)) for k in range(1, 25)) for name, v in (("K_2pct", K2), ("thr_20", T20))}
perm = {name: float(np.mean([np.mean(np.abs(v - rng.permutation(Ks)) <= 1) for _ in range(2000)])) for name, v in (("K_2pct", K2), ("thr_20", T20))}
out["E1_count_nulls"] = {"within1_vs_Kstar": {"K_2pct": float(np.mean(np.abs(K2 - Ks) <= 1)), "thr_20": float(np.mean(np.abs(T20 - Ks) <= 1))},
                         "within1_best_constant": best_const, "within1_permuted_Kstar_mean": perm}
print(out["E1_count_nulls"])
json.dump(out, open(os.path.join(ROOT, "audit", "controls_audit.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else str(o)))
