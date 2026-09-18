"""sym_factor.py -- the symmetrisation coupling factor L(N) = L_X L_Omega of trained heracles ReLU nets (THEORY_relu.md).

Translation-averaging a network N over (a,b,c) -> (a+s, b+t, c+s+t) gives N_bar with logit = clean projection of N
(all junk killed), margin(N_bar) >= margin(N), and costs
    ||X_bar||_*     = p   ( ||X P_0||_F + sqrt2 sum_{k>=1} ||X P_k||_F )       =: p   * D_X
    ||Omega_bar||_* = 1/p ( ||P_0 Omega||_F + sqrt2 sum_{k>=1} ||P_k Omega||_F ) =: 1/p * D_O
so gamma(N) <= L_X L_O gamma(N_bar) with L_X = D_X / ||X||_* >= 1 and L_O = D_O / ||Omega||_* (nu >= 4; 1 at nu 3 where
the readout is Frobenius-priced).  L = 1 iff every frequency slice is a balanced rank-2 block and no rank is shared.
Usage: uv run python heracles/sym_factor.py --cells H1 H3 --out heracles/sym_factor.json
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
ROOT = Path(__file__).resolve().parent

def projectors(p):
    th = 2 * np.pi * np.arange(p) / p
    P = [np.full((p, p), 1.0 / p)]
    for k in range(1, (p - 1) // 2 + 1):
        c, s = np.cos(k * th), np.sin(k * th)
        P.append(np.outer(c, c) * 2 / p + np.outer(s, s) * 2 / p)
    return P

def nuc(M): return float(np.linalg.svd(M, compute_uv=False).sum())
def fro(M): return float(np.linalg.norm(M))

def analyse(run, Pc):
    cfg = json.loads((run / "config.json").read_text()); w = load_file(str(run / "final.safetensors"))
    w = {k: v.astype(np.float64) for k, v in w.items()}; p, nu = cfg["p"], cfg["nu"]
    n_pre = sum(1 for k in w if k.startswith("E")); n_post = sum(1 for k in w if k.startswith("W") and k != "W")
    E = np.eye(p)
    for i in range(1, n_pre + 1): E = w[f"E{i}"] @ E
    X = np.vstack([w["U"] @ E, w["V"] @ E]); Om = w["W"]
    for i in range(n_post, 0, -1): Om = Om @ w[f"W{i}"]
    if p not in Pc: Pc[p] = projectors(p)
    Ps = Pc[p]
    DX = fro(X @ Ps[0]) + np.sqrt(2) * sum(fro(X @ P) for P in Ps[1:])
    DO = fro(Ps[0] @ Om) + np.sqrt(2) * sum(fro(P @ Om) for P in Ps[1:])
    LX = DX / nuc(X) if n_pre else 1.0          # nu 2: X Frobenius-priced, no pre-layer
    LO = DO / nuc(Om) if n_post else 1.0
    return dict(run=str(run.relative_to(ROOT / "runs")), p=p, nu=nu, wd=cfg["weight_decay"], L_X=LX, L_O=LO, L=LX * LO)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="+", default=["H1", "H3"])
    ap.add_argument("--out", default=str(ROOT / "sym_factor.json")); a = ap.parse_args()
    Pc, rows = {}, []
    for cell in a.cells:
        for f in sorted((ROOT / "runs" / cell).glob("**/final.safetensors")):
            cfg = json.loads((f.parent / "config.json").read_text())
            if not str(cfg.get("task", "add")).startswith("add"): continue
            r = analyse(f.parent, Pc); r["cell"] = cell; rows.append(r)
    groups = defaultdict(list)
    for r in rows: groups[(r["cell"], r["p"], r["nu"], r["wd"])].append(r)
    summ = []
    print("| cell | p | nu | wd | n | L_X med | L_X max | L_O med | L_O max | L med | L max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (cell, p, nu, wd), rs in sorted(groups.items()):
        LX = [r["L_X"] for r in rs]; LO = [r["L_O"] for r in rs]; L = [r["L"] for r in rs]
        s = dict(cell=cell, p=p, nu=nu, wd=wd, n=len(rs), L_X_med=float(np.median(LX)), L_X_max=max(LX),
                 L_O_med=float(np.median(LO)), L_O_max=max(LO), L_med=float(np.median(L)), L_max=max(L)); summ.append(s)
        print(f"| {cell} | {p} | {nu} | {wd:g} | {len(rs)} | {s['L_X_med']:.3f} | {s['L_X_max']:.3f} | {s['L_O_med']:.3f} | {s['L_O_max']:.3f} | {s['L_med']:.3f} | {s['L_max']:.3f} |")
    allL = [r["L"] for r in rows]
    print(f"\nall {len(rows)} nets: L median {np.median(allL):.3f}, 90th pct {np.percentile(allL, 90):.3f}, max {max(allL):.3f}")
    Path(a.out).write_text(json.dumps(dict(summary=summ, runs=rows), indent=1))
