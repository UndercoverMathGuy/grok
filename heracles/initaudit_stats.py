"""initaudit_stats.py -- H1: do the different count instruments agree on the SIGN of the init effect, and is it
significant?  Mann-Whitney U (exact-normal approx) on init-low vs init-high within each prime.
Instruments (all on the CLEAN logit profile g(D), the part of the logits depending only on D = a+b-c):
  K_2pct / K_5pct = #{k : clean energy share >= 2% / 5%}
  N_.05           = smallest #frequencies holding 95% of the clean energy (a CUMULATIVE count)
  PR              = participation ratio 1/sum_k en_k^2, en = clean energy shares (threshold-free effective count)
  expH            = exp(Shannon entropy of en) (threshold-free effective count)
  A_1pct          = #{k : b2_k >= 1%}, b2 = share of TOTAL (not clean) centred logit energy -- absolute threshold
"""
import json, math, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
cells = defaultdict(list)
for f in sorted((ROOT / "FINAL_RUNS" / "REP").glob("n2_pow_*_l2q2_lam0.0001_*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0 or r["acc"] < 0.999: continue
    amp = np.array(r["amp"]); en = amp**2 / (amp**2).sum()
    r["PR"] = 1.0 / (en**2).sum()
    r["expH"] = float(np.exp(-(en * np.log(en + 1e-300)).sum()))
    r["A_1pct"] = int((np.array(r["b2"]) >= 0.01).sum())
    cells[(r["p"], r["init_scale"])].append(r)

def mwu(x, y):
    """two-sided Mann-Whitney U with normal approximation and tie correction; returns (U/nm effect size, p)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n, m = len(x), len(y)
    a = np.concatenate([x, y]); rk = np.argsort(np.argsort(a)).astype(float) + 1
    # average ranks for ties
    for v in np.unique(a):
        idx = a == v; rk[idx] = rk[idx].mean()
    U = rk[:n].sum() - n * (n + 1) / 2
    mu = n * m / 2
    _, cnt = np.unique(a, return_counts=True)
    tie = (cnt**3 - cnt).sum()
    sd = math.sqrt(n * m / 12 * ((n + m + 1) - tie / ((n + m) * (n + m - 1))))
    z = (U - mu) / sd if sd > 0 else 0.0
    return U / (n * m), 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

PAIRS = [(59, 1, 4), (71, 1, 4), (71, 1, 2), (71, 2, 4), (89, 2, 4), (97, 2, 4), (113, 3, 4)]
KEYS = ["K_2pct", "K_5pct", "N_05", "PR", "expH", "A_1pct", "clean_frac", "gamma", "margin", "norm2"]
print("== H1/H3: init-low vs init-high, median (low -> high), AUC = P(high > low), two-sided Mann-Whitney p ==")
for p, lo, hi in PAIRS:
    A, B = cells[(p, float(lo))], cells[(p, float(hi))]
    if not A or not B: continue
    print(f"\n  p = {p}: init {lo} (n={len(A)}) vs init {hi} (n={len(B)})")
    for k in KEYS:
        x = [r[k] for r in A]; y = [r[k] for r in B]
        auc, pv = mwu(x, y)
        star = "***" if pv < 1e-3 else "**" if pv < 0.01 else "*" if pv < 0.05 else "   "
        arrow = "UP  " if st.median(y) > st.median(x) else ("DOWN" if st.median(y) < st.median(x) else "FLAT")
        f = "{:.3f}" if k in ("clean_frac", "PR", "expH") else ("{:.3e}" if k in ("gamma",) else "{:.2f}")
        print(f"    {k:11s} {f.format(st.median(x)):>10s} -> {f.format(st.median(y)):>10s}  {arrow}  AUC {auc:.3f}  p {pv:.1e} {star}")
