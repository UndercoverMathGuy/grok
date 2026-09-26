"""AUDIT check 4: independent float64 re-measurement of sampled runs from final.pt (own implementation,
from the paper definitions; no import of core or experiments code).
  table f(x)[c] (product: sum_i prod_t u_{it}(a_t) w_i(c); power: sum_i (sum_t u_{it}(a_t))^n w_i(c))
  margin h = min_x [f(x)[y] - max_{c != y} f(x)[c]],  y = sum a mod p
  ||theta||_{2,q} = (sum_i (sum_t ||u_it||^2 + ||w_i||^2)^{q/2})^{1/q},  gamma = h / ||theta||^nu
  clean amplitude beta_j = 2 p^-nu sum_{x,c} f(x)[c] rho^{-j (sum a - c)}, A_j = |beta_j|, j = 1..h
Usage: python audit/remeasure.py [RUN_DIR ...]  (default: audit/sample.json); writes audit/remeasure.json"""
import json, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(d):
    w = torch.load(os.path.join(d, "final.pt"), map_location="cpu")
    U = w["U"].double().numpy(); W = w["W"].double().numpy()
    if U.ndim == 4: U = U[0]
    if W.ndim == 3: W = W[0]
    return U, W  # U (n, m, p), W (m, p)

def table(model, U, W):
    n, m, p = U.shape
    if model == "product":
        G = U[0].T  # (p, m)
        for t in range(1, n):
            G = (G[:, None, :] * U[t].T[None, :, :]).reshape(-1, m)
    else:
        G = U[0].T
        for t in range(1, n):
            G = (G[:, None, :] + U[t].T[None, :, :]).reshape(-1, m)
        G = G ** n
    return G @ W  # (p^n, p), row-major a_1 slowest

def measure(d):
    c = json.load(open(os.path.join(d, "config.json")))
    U, W = load(d)
    n, m, p = U.shape; nu = n + 1; q = c["q"]
    F = table(c["model"], U, W)
    a = np.indices((p,) * n).reshape(n, -1)
    y = a.sum(0) % p
    fy = F[np.arange(len(y)), y]
    Fo = F.copy(); Fo[np.arange(len(y)), y] = -np.inf
    h = float((fy - Fo.max(1)).min())
    sq = (U ** 2).sum(axis=(0, 2)) + (W ** 2).sum(1)
    norm = float((sq ** (q / 2)).sum() ** (1 / q))
    gamma = h / norm ** nu
    z = (a.sum(0)[:, None] - np.arange(p)[None, :]) % p
    S = np.bincount(z.ravel(), weights=F.ravel(), minlength=p)
    hh = (p - 1) // 2
    j = np.arange(1, hh + 1)
    beta = 2 * p ** (-nu) * (S[None, :] * np.exp(-2j * np.pi * np.outer(j, np.arange(p)) / p)).sum(1)
    A = np.abs(beta)
    return {"margin": h, "l2q_norm": norm, "gamma": gamma, "A": A.tolist(), "p": p, "nu": nu, "q": q, "model": c["model"]}

def compare(d, r):
    m = json.load(open(os.path.join(d, "metrics.json")))["measurement"]
    A0 = np.array(m["A"]); A1 = np.array(r["A"])
    out = {k: {"audit": r[k], "metrics": m[k], "rel_diff": abs(r[k] - m[k]) / max(abs(m[k]), 1e-300)}
           for k in ("margin", "l2q_norm", "gamma")}
    out["A_max_abs_diff_rel_to_maxA"] = float(np.abs(A0 - A1).max() / A0.max())
    for t in (0.1, 0.2):
        out[f"thr_{int(t*100)}"] = {"audit": int((A1 >= t * A1.max()).sum()), "metrics": m["counts"][f"thr_{int(t*100)}"]}
    return out

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        dirs = [os.path.join(ROOT, "experiments", o["exp"], "runs", o["run_id"]) for o in json.load(open(os.path.join(ROOT, "audit", "sample.json")))]
    res = {}
    for d in dirs:
        t = time.time(); r = measure(d); cmp_ = compare(d, r)
        res[os.path.basename(d)] = {"measured": r, "compare": cmp_, "seconds": time.time() - t}
        print(os.path.basename(d), {k: (v["rel_diff"] if isinstance(v, dict) and "rel_diff" in v else v) for k, v in cmp_.items()}, f"{time.time()-t:.1f}s")
    if len(sys.argv) == 1:
        json.dump(res, open(os.path.join(ROOT, "audit", "remeasure.json"), "w"), indent=1)
