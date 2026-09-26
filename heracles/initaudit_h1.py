"""initaudit_h1.py -- H1 instrument-artefact tests on the saved clean spectra (amp, b2 in metrics.json).
Definitions (all per run):
  amp[k]  = |FFT of the clean logit profile g(D)|/p at frequency k, k = 1..n_f. g(D) is the part of the logits that
            depends only on D = a+b-c, i.e. the "clean" (symmetric) part of the function the net computes.
  en[k]   = amp[k]^2 / sum_k amp[k]^2  -- share of CLEAN energy at frequency k (this is what K_2pct/K_5pct threshold).
  b2[k]   = Levine spectral weight = share of TOTAL centred logit energy at frequency k. sum_k b2[k] = clean fraction.
  K_t     = #{k : en[k] >= t}                      (relative-to-clean count, the paper's instrument)
  Kabs_t  = #{k : b2[k] >= t}                      (absolute count: share of TOTAL logit energy -- immune to the
                                                    clean-fraction confound)
  PR      = participation ratio (sum en)^2/sum en^2 = 1/sum en^2   -- effective number of frequencies, threshold-free
  expH    = exp(-sum en log en)                     -- perplexity of the clean spectrum, threshold-free
  PR_abs, expH_abs = same two, computed on b2 renormalised? no: on b2/sum(b2) == en for the clean part... (identical)
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
REP = ROOT / "FINAL_RUNS" / "REP"
cells = defaultdict(list)
for f in sorted(REP.glob("n2_pow_*_l2q2_*/seed*/metrics.json")):
    r = json.loads(f.read_text())
    if r["q"] != 2.0 or r["lam"] != 1e-4 or "_from" in f.parent.parent.name or r["acc"] < 0.999: continue
    cells[(r["p"], r["init_scale"])].append(r)
def med(xs): return st.median(xs)
THR = [0.01,0.02,0.03,0.05,0.07,0.10]
print("== relative-to-clean threshold sweep: median K_t over fit seeds ==")
print("  p init  n | " + " ".join(f"K_{t:.0%}".rjust(6) for t in THR) + " |    PR   expH | clean")
for k in sorted(cells):
    rs = cells[k]; row=[]
    prs=[]; hs=[]
    for r in rs:
        amp=np.array(r["amp"]); en=amp**2/ (amp**2).sum()
        prs.append(1.0/ (en**2).sum()); hs.append(float(np.exp(-(en*np.log(en+1e-300)).sum())))
    for t in THR:
        row.append(med([int((np.array(r["amp"])**2/(np.array(r["amp"])**2).sum() >= t).sum()) for r in rs]))
    print(f"{k[0]:>4}{k[1]:>4g}{len(rs):>3} | " + " ".join(f"{v:6.1f}" for v in row) +
          f" | {med(prs):5.2f} {med(hs):6.2f} | {med([r['clean_frac'] for r in rs]):.3f}")
print()
print("== ABSOLUTE threshold sweep on b2 (share of TOTAL logit energy): median #{k: b2[k]>=t} ==")
ATHR=[0.005,0.01,0.02,0.03,0.05]
print("  p init  n | " + " ".join(f"A_{t:.1%}".rjust(7) for t in ATHR))
for k in sorted(cells):
    rs=cells[k]
    print(f"{k[0]:>4}{k[1]:>4g}{len(rs):>3} | " + " ".join(f"{med([int((np.array(r['b2'])>=t).sum()) for r in rs]):7.1f}" for t in ATHR))
print()
print("== sorted clean-energy spectra en[k] (top 14), p=71, seeds 100-104 ==")
for init in (1.0,2.0,4.0):
    for r in sorted(cells[(71,init)], key=lambda r:r["seed"])[:5]:
        amp=np.array(r["amp"]); en=np.sort(amp**2/(amp**2).sum())[::-1]
        print(f" init{init:g} seed{r['seed']} K2={r['K_2pct']:2d} K5={r['K_5pct']:2d} clean={r['clean_frac']:.2f} gam={r['gamma']:.2e} | " +
              " ".join(f"{v:.3f}" for v in en[:14]))
