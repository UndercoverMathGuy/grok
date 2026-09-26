"""ladder_archived.py -- the archived free-trained endpoints against the exact constructions.

Claim to check (Part 1 item 6): a trained endpoint has a STRICTLY HIGHER training objective than the
constructed K*-net zero-padded to the same width.  Since zero neurons cost nothing and the trained
runs use m = 500 >> (n+1)K, the construction lives inside the trained run's own parameter space, so a
lower objective proves the endpoint is a LOCAL, non-global minimum of CE + lam P_q.

Everything is exact fp64 over the FULL p^n table for the trained nets (their tables are not clean, so
the reduction of ladder_core does not apply to them) and over the p-1 gap vector for the constructions.

Usage: uv run python heracles/ladder_archived.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
from safetensors.torch import load_file  # noqa: E402

from ladder_core import (OUT, clean_spectrum, gaps_from_spectrum, gmat, margin_of_gaps,  # noqa: E402
                         obj_star_gaps, val_of)

# archived cells, exactly as fvf_obj.py's CELLS names them (same directories, same lam)
CELLS = {
    "chain_n2_p71_q2_lam1e-4": dict(p=71, n=2, act="prod", q=2.0, lam=1e-4, family="chain",
        runs=["FINAL_RUNS/REP/n2_prod_p71_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000_clip1_wdw20000_esc0.8_lr212.8"]),
    "chain_n2_p71_q2_lam1e-5": dict(p=71, n=2, act="prod", q=2.0, lam=1e-5, family="chain",
        runs=["FINAL_RUNS/REP/n2_prod_p71_l2q2_lam1e-05_lrmax12.8_init4_m500_st40000_clip1_wdw20000_esc0.8_lr212.8"]),
    "chain_n2_p71_q3_lam1e-5": dict(p=71, n=2, act="prod", q=3.0, lam=1e-5, family="chain",
        runs=["FINAL_RUNS/REP/n2_prod_p71_l2q3_lam1e-05_lrmax12.8_init4_m500_st40000_clip1_wdw20000_esc0.8_lr212.8"]),
    "chain_n3_p71_q2_lam1e-5": dict(p=71, n=3, act="prod", q=2.0, lam=1e-5, family="chain",
        runs=["FINAL_RUNS/NU4/n3_prod_p71_l2_lam1e-05_lrmax12.8_init14_m500_st120000_clip1_wdw20000_esc0.8_lr23.2"]),
    "chain_n3_p71_q4_lam1e-6": dict(p=71, n=3, act="prod", q=4.0, lam=1e-6, family="chain",
        runs=["FINAL_RUNS/NU4/n3_prod_p71_l2q4_lam1e-06_lrmax12.8_init14_m500_st120000_clip1_wdw20000_esc0.8_lr23.2"]),
    "x2_p71_q2_lam1e-4": dict(p=71, n=2, act="pow", q=2.0, lam=1e-4, family="xn",
        runs=["FINAL_RUNS/REP/n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000",
              "FINAL_RUNS/REP/n2_pow_p71_l2q2_lam0.0001_lrmax12.8_init1_m500_st40000"]),
    "x2_p113_q2_lam1e-4": dict(p=113, n=2, act="pow", q=2.0, lam=1e-4, family="xn",
        runs=["FINAL_RUNS/REP/n2_pow_p113_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000"]),
}


def table(p, n):
    r = torch.arange(p ** n)
    ops = torch.stack([(r // (p ** j)) % p for j in range(n)], 1)
    return ops, ops.sum(1) % p


def logit_gaps(E, W, p, n, act, chunk=20000):
    ops, y = table(p, n)
    out = torch.empty(len(ops), p)
    for s in range(0, len(ops), chunk):
        o = ops[s:s + chunk]
        cols = [E[0 if E.shape[0] == 1 else j][:, o[:, j]] for j in range(n)]
        if act == "prod":
            h = cols[0]
            for j in range(1, n):
                h = h * cols[j]
        else:
            h = sum(cols) ** n
        z = (W @ h).T
        out[s:s + chunk] = z - z.gather(1, y[s:s + chunk, None])
    return out, y


def ce_scaled(d, y, t, nu):
    dd = d * math.exp(nu * t)
    dd = dd.scatter(1, y[:, None], -float("inf"))
    return float(torch.nn.functional.softplus(torch.logsumexp(dd, 1)).mean())


def obj_star_table(d, y, Pq, lam, nu, q, lo=-6.0, hi=6.0, iters=90):
    f = lambda t: ce_scaled(d, y, t, nu) + lam * math.exp(q * t) * Pq
    gr = (math.sqrt(5) - 1) / 2
    c, dd = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(dd)
    for _ in range(iters):
        if fc < fd:
            hi, dd, fd = dd, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, dd, fd
            dd = lo + gr * (hi - lo)
            fd = f(dd)
    t = (lo + hi) / 2
    return f(t), math.exp(t), ce_scaled(d, y, t, nu)


def main():
    fl = json.loads((OUT / "json" / "free_ladders_polished.json").read_text())
    res = []
    lines = ["| cell | p | nu | q | lam | K | constructed obj* | best free obj* | free - constructed | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, C in CELLS.items():
        p, n, q, lam, act, fam = C["p"], C["n"], C["q"], C["lam"], C["act"], C["family"]
        nu = n + 1
        s = nu / q
        lad = fl[f"{p}_{s:.6f}"]
        g = gmat(p)
        con = {}
        for K in sorted(lad, key=int):
            S = [k for k, b in zip(lad[K]["S"], lad[K]["b"]) if b > 1e-4]
            b = np.array([b for b in lad[K]["b"] if b > 1e-4])
            b = b / b.sum()
            A, Pq = clean_spectrum(fam, p, n, S, b, q)
            gp = gaps_from_spectrum(p, S, A)
            o = obj_star_gaps(gp, Pq, lam, nu, q)
            con[int(K)] = dict(obj=o["obj"], ce=o["ce"], K_active=len(S), margin=margin_of_gaps(gp))
        # the DENSE FLAT constructed net (all (p-1)/2 frequencies, equal shares) is also inside the
        # trained net's parameter space whenever m >= N*nf; at q = nu it is the programme's true
        # optimum, which the K <= 12 ladder truncates away.
        nf = (p - 1) // 2
        Sd = list(range(1, nf + 1))
        bd = np.full(nf, 1.0 / nf)
        Ad, Pqd = clean_spectrum(fam, p, n, Sd, bd, q)
        gpd = gaps_from_spectrum(p, Sd, Ad)
        od = obj_star_gaps(gpd, Pqd, lam, nu, q)
        con["dense"] = dict(obj=od["obj"], ce=od["ce"], K_active=nf, margin=margin_of_gaps(gpd))
        bK = min(con, key=lambda K: con[K]["obj"])
        frees = []
        for run in C["runs"]:
            for sd in sorted((ROOT / run).glob("seed*")):
                f = sd / "final.safetensors"
                if not f.exists():
                    continue
                stt = load_file(str(f))
                E, W = stt["E"].double(), stt["W"].double()
                n2 = (E ** 2).sum((0, 2)) * (n if E.shape[0] == 1 else 1) + (W ** 2).sum(0)
                Pq = float((n2 ** (q / 2)).sum())
                d, y = logit_gaps(E, W, p, n, act)
                o, sc, ce = obj_star_table(d, y, Pq, lam, nu, q)
                frees.append(dict(run=run.split("/")[-1], seed=sd.name, obj=o, ce=ce, scale=sc, Pq=Pq))
        if not frees:
            print(f"  {name}: no archived seeds found, skipped", flush=True)
            continue
        bf = min(f["obj"] for f in frees)
        row = dict(cell=name, p=p, n=n, nu=nu, q=q, lam=lam, best_K=bK, K_active=con[bK]["K_active"],
                   con_obj=con[bK]["obj"], free_obj_min=bf,
                   free_obj_median=float(np.median([f["obj"] for f in frees])),
                   n_seeds=len(frees), constructed=con, free=frees)
        res.append(row)
        v = "constructed BEATS every free endpoint" if con[bK]["obj"] < bf else "free endpoint is lower"
        lines.append(f"| {name} | {p} | {nu} | {q:g} | {lam:g} | {bK} ({con[bK]['K_active']} active) | "
                     f"{con[bK]['obj']:.6f} | {bf:.6f} ({len(frees)} seeds) | {bf - con[bK]['obj']:+.6f} | {v} |")
        print(lines[-1], flush=True)
    (OUT / "json" / "archived.json").write_text(json.dumps(res, indent=1))
    (OUT / "archived.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "archived.md")


if __name__ == "__main__":
    main()
