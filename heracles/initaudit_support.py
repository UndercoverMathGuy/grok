"""initaudit_support.py -- is the measured frequency set a GOOD support, and is the net near the ceiling for it?
  S_run  = the top-K_2pct frequencies of the clean logit spectrum of a run.
  M_S    = min over D != 0 of sum_{k in S} (1 - cos(2 pi k D / p))  -- the 'coverage' the theory maximises.
  M_K^*  = the best achievable coverage for that many frequencies (heracles/kstar.json).
  cov    = M_S / M_K^*   -- 1.0 means the run picked a coverage-optimal support of its own size.
  gamma_con(K) = exact constructed normalised margin on the optimal K-support (FINAL_RUNS/INITAUDIT/ceiling_pXX.json).
  att    = gamma_run / gamma_con(K_2pct_run)  -- how close the trained net is to the best circuit of its own size.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
KS = json.loads((ROOT / "kstar.json").read_text())
CEIL = {p: json.loads((ROOT / "FINAL_RUNS" / "INITAUDIT" / f"ceiling_p{p}.json").read_text()) for p in (59, 71)}
cells = defaultdict(list)
for f in sorted((ROOT / "FINAL_RUNS" / "REP").glob("n2_pow_p*_l2q2_lam0.0001_*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0 or r["acc"] < 0.999 or r["p"] not in (59, 71): continue
    cells[(r["p"], r["init_scale"])].append(r)

def Mset(p, S):
    D = np.arange(1, p)
    return float(sum(1 - np.cos(2 * np.pi * k * D / p) for k in S).min())

print("== H3e: quality of the measured support and distance to the constructed ceiling ==")
print("   p init | K2 med | cov = M_S/M_K* med [range] | att = gamma_run/gamma_con(K_run) med [range] | gamma/gamma_con(K*)")
for k in sorted(cells):
    p, init = k; rs = cells[k]
    covs, atts = [], []
    for r in rs:
        amp = np.array(r["amp"]); en = amp**2 / (amp**2).sum()
        K = max(1, min(12, r["K_2pct"]))
        S = [int(i) + 1 for i in np.argsort(-en)[:K]]
        covs.append(Mset(p, S) / KS[str(p)]["M_K"][str(K)])
        atts.append(r["gamma"] / CEIL[p][str(K)])
    kstar = "6" if p == 59 else "5"
    print(f" {p:>4}{init:>5g} | {st.median([r['K_2pct'] for r in rs]):6} | {st.median(covs):.3f} [{min(covs):.3f}-{max(covs):.3f}] | "
          f"{st.median(atts):.3f} [{min(atts):.3f}-{max(atts):.3f}] | {st.median([r['gamma'] for r in rs])/CEIL[p][kstar]:.3f}")
