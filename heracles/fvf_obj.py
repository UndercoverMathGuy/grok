"""fvf_obj.py -- IS THE JUNK BASIN A GLOBAL MINIMUM OF THE TRAINING OBJECTIVE?

The training objective is  obj(theta) = CE(theta) + lam * ||theta||^2 , with
  CE(theta)   = mean over all p^n rows of the cross-entropy of the p logits,
  ||theta||^2 = sum of squares of every weight in the net (plain weight decay),
  lam         = the weight-decay coefficient the run actually used.
The nets are nu-homogeneous (nu = n+1 for x^n / the gated chain, 2 for relu), so rescaling every weight by s
multiplies the logits by s^nu and the norm-squared by s^2.  For any fixed weight DIRECTION we can therefore
minimise obj exactly over the scale s by 1-D golden section:  obj*(dir) = min_s CE(s^nu z) + lam s^2 ||theta||^2.

We compute obj* for
  (a) every archived FREE-training endpoint of a cell, and
  (b) the analytic junk-free constructed K-frequency net on the coverage-optimal support S_K, K = 1..Kmax,
and compare.  Because the free runs have width m = 500 >> the constructed widths, the constructed direction is
INSIDE the free run's own parameter space (pad with zero neurons, which cost nothing), so
      min_K obj*(constructed K) < obj*(free endpoint)   PROVES the free endpoint is not the global minimiser.

No training; fp64 throughout.  Local CPU.
Usage: uv run python heracles/fvf_obj.py --cell x2_p71
"""
import argparse, itertools, json, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from safetensors.torch import load_file  # noqa: E402

KS = json.loads((ROOT / "kstar.json").read_text())


# ---------------------------------------------------------------- task + forward (fp64, chunked)
def table(p, n):
    r = torch.arange(p ** n)
    ops = torch.stack([(r // (p ** j)) % p for j in range(n)], 1)
    return ops, ops.sum(1) % p


def logit_gaps(E, W, p, n, act, chunk=20000):
    """d[row, c] = z[row, c] - z[row, y(row)] for every row (fp64).  E: (n, m, p), W: (p, m)."""
    ops, y = table(p, n)
    out = torch.empty(len(ops), p)
    for s in range(0, len(ops), chunk):
        o = ops[s:s + chunk]
        cols = [E[j][:, o[:, j]] for j in range(E.shape[0] if E.shape[0] == n else 1)]
        if E.shape[0] == 1:                     # tied
            cols = [E[0][:, o[:, j]] for j in range(n)]
        if act == "prod":
            h = cols[0]
            for j in range(1, n):
                h = h * cols[j]
        elif act == "relu":
            h = torch.relu(sum(cols))
        else:
            x = sum(cols)
            h = x ** n
        z = (W @ h).T                            # (chunk, p)
        out[s:s + chunk] = z - z.gather(1, y[s:s + chunk, None])
    return out, y


def ce_of_scale(d, y, t, nu):
    """CE when every weight is multiplied by exp(t): logits scale by exp(nu t).  d = z_c - z_y (0 at c = y)."""
    dd = d * math.exp(nu * t)
    dd = dd.scatter(1, y[:, None], -float("inf"))
    return float(torch.nn.functional.softplus(torch.logsumexp(dd, 1)).mean())


def obj_star(d, y, n2, lam, nu, lo=-4.0, hi=4.0, iters=60):
    """min over the global scale s = exp(t) of CE(s^nu z) + lam s^2 ||theta||^2.  Returns (obj, s, CE, pen)."""
    f = lambda t: ce_of_scale(d, y, t, nu) + lam * math.exp(2 * t) * n2
    gr = (math.sqrt(5) - 1) / 2
    c, dd = hi - gr * (hi - lo), lo + gr * (hi - lo); fc, fd = f(c), f(dd)
    for _ in range(iters):
        if fc < fd:
            hi, dd, fd = dd, c, fc; c = hi - gr * (hi - lo); fc = f(c)
        else:
            lo, c, fc = c, dd, fd; dd = lo + gr * (hi - lo); fd = f(dd)
    t = (lo + hi) / 2; s = math.exp(t)
    ce = ce_of_scale(d, y, t, nu); pen = lam * s ** 2 * n2
    return ce + pen, s, ce, pen


def margin_of(d, y):
    dd = d.scatter(1, y[:, None], -float("inf"))
    return float((-dd.max(1).values).min())


# ---------------------------------------------------------------- constructed junk-free nets
def build_x2(p, S):
    """Morwani's x^2 net, junk-free 6-neuron cyclic family per frequency (initaudit_ceiling.py)."""
    a = torch.arange(p) * 2 * math.pi / p
    ph = [(2 * math.pi * l / 6, 2 * math.pi * 4 * l / 6) for l in range(6)]
    E = torch.zeros(2, len(S) * 6, p); W = torch.zeros(p, len(S) * 6); i = 0
    for k in S:
        for p1, p2 in ph:
            E[0, i] = torch.cos(k * a + p1); E[1, i] = torch.cos(k * a + p2)
            W[:, i] = torch.cos(k * a + p1 + p2); i += 1
    return E, W


def build_chain(p, n, S, tied=False):
    """Gated chain, (n+1)-neuron common-phase junk-free population per frequency (PROOF_AUDIT F1)."""
    a = torch.arange(p) * 2 * math.pi / p
    E = torch.zeros(1 if tied else n, len(S) * (n + 1), p); W = torch.zeros(p, len(S) * (n + 1)); i = 0
    for k in S:
        for l in range(n + 1):
            ph = math.pi * l / (n + 1)
            for j in range(E.shape[0]):
                E[j, i] = torch.cos(k * a + ph)
            W[:, i] = torch.cos(k * a + n * ph); i += 1
    return E, W


CELLS = {
    # name: (p, n, act, lam, free-run glob, builder)
    "x2_p71": dict(p=71, n=2, act="pow", lam=1e-4, tied=False,
                   free="FINAL_RUNS/REP/n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000",
                   free2="FINAL_RUNS/REP/n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init1_m500_st40000", build="x2"),
    "chain_n2_p71": dict(p=71, n=2, act="prod", lam=1e-5, tied=False,
                         free="FINAL_RUNS/REP/n2_prod_p71_l2q2_lam1e-05_lrmax12.8_init4_m500_st40000_clip1_wdw20000_esc0.8_lr212.8",
                         build="chain"),
    "chain_n2_p71_lam1e-6": dict(p=71, n=2, act="prod", lam=1e-6, tied=False,
                                 free="FINAL_RUNS/REP/n2_prod_p71_l2q2_lam1e-06_lrmax12.8_init4_m500_st60000_from1e-05_clip1",
                                 build="chain"),
    "x2_p71_lam1e-5": dict(p=71, n=2, act="pow", lam=1e-5, tied=False,
                           free="FINAL_RUNS/REP/n2_pow_p71_l2q2_lam1e-05_lrmax12.8_init1_m500_st60000_from0.0001",
                           build="x2"),
    "chain_n3_p71": dict(p=71, n=3, act="prod", lam=1e-5, tied=False,
                         free="FINAL_RUNS/NU4/n3_prod_p71_l2_lam1e-05_lrmax12.8_init14_m500_st120000_clip1_wdw20000_esc0.8_lr23.2",
                         build="chain"),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="x2_p71", choices=list(CELLS))
    ap.add_argument("--Kmax", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=6, help="max free seeds to evaluate")
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "FVF"))
    a = ap.parse_args()
    C = CELLS[a.cell]; p, n, act, lam = C["p"], C["n"], C["act"], C["lam"]; nu = n + 1
    print(f"== cell {a.cell}: p {p} n {n} act {act} nu {nu} lam {lam:g}", flush=True)

    res = dict(cell=a.cell, p=p, n=n, act=act, nu=nu, lam=lam, constructed={}, free=[])
    sup = KS[str(p)]["S"]
    for K in range(1, a.Kmax + 1):
        S = sup[str(K)]
        E, W = build_x2(p, S) if C["build"] == "x2" else build_chain(p, n, S, C["tied"])
        n2 = float(E.square().sum() + W.square().sum())
        d, y = logit_gaps(E, W, p, n, act)
        o, s, ce, pen = obj_star(d, y, n2, lam, nu)
        mg = margin_of(d, y) * s ** nu; nn = n2 * s ** 2
        gam = mg / nn ** (nu / 2)
        res["constructed"][K] = dict(S=S, obj=o, ce=ce, pen=pen, scale=s, margin=mg, norm2=nn, gamma=gam, m=int(E.shape[1]))
        print(f"  constructed K={K:2d} m={E.shape[1]:4d}  obj* {o:.6f}  (CE {ce:.6f} + pen {pen:.6f})  "
              f"||th||^2 {nn:8.1f}  margin {mg:7.3f}  gamma {gam:.4e}", flush=True)

    for tag in ("free", "free2"):
        if tag not in C:
            continue
        for sd in sorted((ROOT / C[tag]).glob("seed*"))[:a.seeds]:
            st = load_file(str(sd / "final.safetensors"))
            E, W = st["E"].double(), st["W"].double()
            mm = json.loads((sd / "metrics.json").read_text())
            n2 = float(E.square().sum() + W.square().sum())
            d, y = logit_gaps(E, W, p, n, act)
            o, s, ce, pen = obj_star(d, y, n2, lam, nu)
            mg = margin_of(d, y) * s ** nu; nn = n2 * s ** 2
            row = dict(run=C[tag].split("/")[-1], seed=sd.name, obj=o, ce=ce, pen=pen, scale=s, margin=mg, norm2=nn,
                       gamma=mg / nn ** (nu / 2), clean=mm["clean_frac"], K_2pct=mm["K_2pct"], N_05=mm["N_05"], m=mm["m"])
            res["free"].append(row)
            print(f"  FREE {sd.name} ({C[tag].split('/')[-1][:34]})  obj* {o:.6f}  (CE {ce:.6f} + pen {pen:.6f})  "
                  f"scale {s:.4f}  margin {mg:7.3f}  gamma {row['gamma']:.4e}  clean {row['clean']:.3f}  K2 {row['K_2pct']}", flush=True)

    bK = min(res["constructed"], key=lambda K: res["constructed"][K]["obj"])
    bo = res["constructed"][bK]["obj"]
    fo = [r["obj"] for r in res["free"]]
    print(f"\n  best constructed: K = {bK}, obj* = {bo:.6f}")
    if fo:
        print(f"  free endpoints  : obj* median {np.median(fo):.6f}, min {min(fo):.6f}")
        print(f"  VERDICT: constructed {'BEATS' if bo < min(fo) else 'does NOT beat'} every free endpoint "
              f"(gap to best free = {min(fo) - bo:+.6f})")
    res["best_constructed_K"] = bK
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"obj_{a.cell}.json").write_text(json.dumps(res, indent=1))
