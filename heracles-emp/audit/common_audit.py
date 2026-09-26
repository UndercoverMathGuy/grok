"""AUDIT helpers (own code). Reads RUNS.csv, config.json, metrics.json, predictions.json."""
import csv, glob, json, os
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "experiments")

def last_status(e):
    st = {}
    for r in csv.DictReader(open(os.path.join(EX, e, "RUNS.csv"))):
        st[r["run_id"]] = r["status"]
    return st

def runs(e):
    st = last_status(e); out = []
    for d in sorted(glob.glob(os.path.join(EX, e, "runs", "*"))):
        rid = os.path.basename(d)
        c = json.load(open(os.path.join(d, "config.json")))
        mf = os.path.join(d, "metrics.json")
        m = json.load(open(mf)) if os.path.exists(mf) else None
        out.append({"run_id": rid, "dir": d, "status": st.get(rid), "c": c, "m": (m or {}).get("measurement"), "mall": m})
    return out

def preds(e):
    P = json.load(open(os.path.join(EX, e, "predictions.json")))
    cells = P.get("cells") or [P["cell"]]
    return {(c.get("model", "product"), c["nu"], c["p"], round(c["s"], 6)): c for c in cells}, P

def key_of(r):
    c = r["c"]; return (c["model"], c["n"] + 1, c["p"], round((c["n"] + 1) / c["q"], 6))

def shares(A, s):
    a = np.clip(np.asarray(A, float), 0, None) ** (1 / s); return a / a.sum()

def bnorm(A, s):
    A = np.clip(np.asarray(A, float), 0, None); return A / (A ** (1 / s)).sum() ** s

def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())

def cov_matrix(p):
    h = (p - 1) // 2; d = np.arange(1, h + 1)
    return 1 - np.cos(2 * np.pi * np.outer(d, d) / p)  # [delta, j]

def prog_value(A, p, s):
    A = np.clip(np.asarray(A, float), 0, None)
    return float((cov_matrix(p) @ A).min() / (A ** (1 / s)).sum() ** s)

def align(A, Y, s):
    w = shares(A, s); best = None
    for i, y in enumerate(np.atleast_2d(Y)):
        k = (-cos(A, y), tv(w, shares(y, s)))
        if best is None or k < best[0]: best = (k, i)
    return best[1]

def thr(A, t):
    A = np.asarray(A, float); return int((A >= t * A.max()).sum())

def energy_counts(A):
    e = np.clip(np.asarray(A, float), 0, None) ** 2; e = e / e.sum()
    cum = np.cumsum(np.sort(e)[::-1])
    return {"K_2pct": int((e >= 0.02).sum()), "K_5pct": int((e >= 0.05).sum()),
            "N_05": int(np.argmax(cum >= 0.95 - 1e-12) + 1), "N_10": int(np.argmax(cum >= 0.90 - 1e-12) + 1)}

def sweep(meas, pred):
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    ss = ((meas - meas.mean()) ** 2).sum()
    r = np.corrcoef(meas, pred)[0, 1] if meas.std() > 0 and pred.std() > 0 else np.nan
    lr = stats.linregress(pred, meas) if pred.std() > 0 else None
    tq = stats.t.ppf(0.975, len(meas) - 2)
    return {"n": len(meas), "exact": float(np.mean(meas == pred)), "within1": float(np.mean(np.abs(meas - pred) <= 1 + 1e-12)),
            "r2_id": float(1 - ((meas - pred) ** 2).sum() / ss) if ss > 0 else np.nan, "r2_fit": float(r * r),
            "pearson": float(r), "spearman": float(stats.spearmanr(meas, pred).statistic), "kendall": float(stats.kendalltau(meas, pred).statistic),
            "slope": float(lr.slope) if lr else np.nan, "slope_ci": [float(lr.slope - tq * lr.stderr), float(lr.slope + tq * lr.stderr)] if lr else None,
            "bias": float((meas - pred).mean())}

def boot_seeds(vals_by_seed, stat, n=2000, seed=12345):
    rng = np.random.default_rng(seed); k = len(vals_by_seed)
    b = [stat(np.concatenate([np.atleast_1d(vals_by_seed[i]) for i in rng.integers(0, k, k)])) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

def Bp(p, nu):
    return 2 * (2 * p * nu) ** (-nu / 2)
