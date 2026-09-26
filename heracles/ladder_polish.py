"""ladder_polish.py -- repair and polish the free-amplitude ladders, then rebuild the cells.

Two fixes to `ladder_exact.py`'s first pass, both needed because the support search is heuristic:

  (R) ACTIVE-SUPPORT REPAIR.  A solve on a K-support may leave some shares at zero; that solution is
      then really a k-frequency net with k = n_active < K, and it is a valid (and sometimes better)
      candidate for rung k.  We collect every solution found at every rung, key it by its number of
      active shares, and give rung K the best value over all solutions with n_active <= K.  This
      also makes the free ladder monotone in K, which it must be (a K-support contains every smaller
      support), so any decrease was search noise.

  (P) SWAP POLISH.  From each rung's winner, try replacing one frequency at a time by every other
      frequency, re-solving the shares, and keep any improvement; repeat to convergence.

Then re-derives cells.json with ladder_exact.cell_rows, and flags COLLAPSED cells (where the optimal
global scale is so small that the objective is within 1% of log p, i.e. weight decay kills the net
rather than shaping it -- the regime RESULTS_PROD saw at lam 1e-5, q = 4).

Usage: uv run python heracles/ladder_polish.py
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import OUT, block_spec, gmat, solve_support, val_of  # noqa: E402
from ladder_exact import FAMILIES, LAMS, QS, cell_rows, s_values  # noqa: E402
from ladder_core import B_const  # noqa: E402


def jdefault(o):
    """numpy -> plain python for json.dump."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def polish_rung(g, S, b, s, max_pass=3):
    """Swap local search on the FREE-amplitude value."""
    nf = g.shape[0]
    v, S, b = val_of(g, S, b, s), list(S), [float(x) for x in b]
    v2, b2 = solve_support(g, S, s, restarts=8)
    if v2 > v:
        v, b = v2, [float(x) for x in b2]
    for _ in range(max_pass):
        improved = False
        for i in range(len(S)):
            for knew in range(1, nf + 1):
                if knew in S:
                    continue
                S2 = sorted(S[:i] + S[i + 1:] + [knew])
                v2, b2 = solve_support(g, S2, s, restarts=2)
                if v2 > v + 1e-12:
                    v3, b3 = solve_support(g, S2, s, restarts=8)
                    if v3 > v:
                        v, S, b, improved = v3, S2, [float(x) for x in b3], True
                        break
            if improved:
                break
        if not improved:
            break
    return v, S, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxpass", type=int, default=1)
    ap.add_argument("--kpolish", type=int, default=8, help="polish rungs K <= this (repair applies to all)")
    a = ap.parse_args()
    fl_path = OUT / "json" / "free_ladders.json"
    fl = json.loads(fl_path.read_text())
    covs = {p: json.loads((OUT / "json" / f"cov_p{p}.json").read_text())
            for p in sorted({int(k.split("_")[0]) for k in fl})}
    out = {}
    for key, lad in fl.items():
        p, s = int(key.split("_")[0]), float(key.split("_")[1])
        g = gmat(p)
        t0 = time.time()
        # (P) polish every rung
        pol = {}
        for K in sorted(lad, key=int):
            if int(K) <= a.kpolish:
                v, S, b = polish_rung(g, lad[K]["S"], lad[K]["b"], s, max_pass=a.maxpass)
            else:
                v, S, b = lad[K]["value"], lad[K]["S"], [float(x) for x in lad[K]["b"]]
            pol[int(K)] = dict(value=v, S=S, b=b, n_active=int((np.asarray(b) > 1e-4).sum()))
        # (R) active-support repair + monotone closure
        pool = []
        for K, r in pol.items():
            act = [(k, bb) for k, bb in zip(r["S"], r["b"]) if bb > 1e-4]
            Sa = [k for k, _ in act]
            ba = np.array([bb for _, bb in act])
            ba = ba / ba.sum()
            pool.append((len(Sa), val_of(g, Sa, ba, s), Sa, ba.tolist()))
            pool.append((len(r["S"]), r["value"], r["S"], r["b"]))
        rep = {}
        for K in sorted(pol):
            cand = [c for c in pool if c[0] <= K]
            best = max(cand, key=lambda c: c[1]) if cand else None
            if best is not None and best[1] > pol[K]["value"] + 1e-15:
                Sx, bx = list(best[2]), list(best[3])
                while len(Sx) < K:                      # pad with a zero-share frequency
                    for k in range(1, g.shape[0] + 1):
                        if k not in Sx:
                            Sx.append(k)
                            bx.append(0.0)
                            break
                rep[K] = dict(value=best[1], S=Sx, b=bx, n_active=best[0])
            else:
                rep[K] = pol[K]
        out[key] = {str(K): rep[K] for K in sorted(rep)}
        top = max(rep, key=lambda K: rep[K]["value"])
        print(f"  p {p} s {s:.4f}: Gamma {rep[top]['value']:.8f} (was {max(r['value'] for r in lad.values()):.8f})"
              f"  K* {rep[top]['n_active']}  S {sorted(k for k, bb in zip(rep[top]['S'], rep[top]['b']) if bb > 1e-4)}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(OUT / "json" / "free_ladders_polished.json", "w"), indent=1, default=jdefault)

    # rebuild the cells
    kmax = max(len(v) for v in out.values())
    allres = {}
    for family, n in FAMILIES:
        for p in sorted(covs):
            for q in QS[n]:
                s = (n + 1) / q
                fl2 = {int(K): v for K, v in out[f"{p}_{s:.6f}"].items()}
                rows, dense = cell_rows(family, p, n, q, kmax, covs[p], fl2, LAMS)
                key = f"{family}_n{n}_p{p}_q{q:.4g}"
                gam_eq = [r["eq"]["gamma"] for r in rows]
                gam_fr = [r["free"]["gamma"] for r in rows]
                lam_main = "0.0001" if (abs(q - 2) < 1e-9 and n == 2) else ("1e-05" if abs(q - 2) < 1e-9 else "1e-06")
                collapsed = {lk: bool(min(r["eq"]["ce"][lk] for r in rows) > 0.99 * math.log(p))
                             for lk in rows[0]["eq"]["ce"]}
                allres[key] = dict(family=family, p=p, n=n, nu=n + 1, q=q, s=s,
                                   N=block_spec(family, n)[0], mu=block_spec(family, n)[3], B=B_const(p, n),
                                   rows=rows, dense=dense, lam_main=lam_main, collapsed=collapsed,
                                   argmax_gamma_eq=int(np.argmax(gam_eq)) + 1,
                                   K_star_free=int(rows[int(np.argmax(gam_fr))]["free"]["n_active"]),
                                   argmin_obj={lk: int(np.argmin([r["eq"]["obj"][lk] for r in rows])) + 1
                                               for lk in rows[0]["eq"]["obj"]},
                                   argmin_obj_free={lk: int(np.argmin([r["free"]["obj"][lk] for r in rows])) + 1
                                                    for lk in rows[0]["free"]["obj"]},
                                   dense_ratio_eq=dense["gamma"] / max(gam_eq),
                                   dense_ratio_free=dense["gamma"] / max(gam_fr))
                print(f"{key:24s} s {s:.3f} K* {allres[key]['K_star_free']:2d} argmax_eq {allres[key]['argmax_gamma_eq']:2d} "
                      f"argmin obj*({lam_main}) {allres[key]['argmin_obj'][lam_main]:2d} "
                      f"{'COLLAPSED' if collapsed[lam_main] else ''} dense/best {allres[key]['dense_ratio_eq']:.4f}",
                      flush=True)
    json.dump(allres, open(OUT / "json" / "cells.json", "w"), indent=1, default=jdefault)
    print("rewrote", OUT / "json" / "cells.json")


if __name__ == "__main__":
    main()
