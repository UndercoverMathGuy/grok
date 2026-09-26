"""pax_xi.py -- the escape number Xi and what it explains.  Analysis only (uses the tau* runs from pax_job1.sh).

Symbols (plain language)
  s0 = ||theta_0||      length of the whole weight vector at step 0 (sqrt of the sum of squares of every weight)
  lam                   the weight-decay coefficient in  mean-CE + lam * ||theta||^2
  tau = sum_t lr_t * ||theta_t||^(nu-2)   PROGRESS TIME: the clock on which the *direction* of the weight vector
                        moves.  With nu = 3 it is  sum_t lr_t * ||theta_t||.
  tau*(p)               progress time the flow needs to escape the CE plateau (measured at lam = 0, where the norm
                        is constant so tau = lr * s0 * t).
  tau_inf = s0/(2 lam)  ALL the progress time a run can ever buy, because ds/dtau = -2 lam exactly (see PRIME_AXIS §2)
  Xi = tau_inf/tau*     ESCAPE NUMBER.  Escape iff Xi > 1; fraction of the initial norm still there at escape = 1 - 1/Xi.
"""
import json, math, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; P = ROOT / "FINAL_RUNS" / "PAX"; R = ROOT / "FINAL_RUNS" / "REP"
M = 500

def tau_runs():
    out = {}
    for d in sorted(P.glob("pax_tau_*")):
        ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
        if not ms or ms[0]["t_ce80"] is None:
            print("no escape:", d.name); continue
        out[d.name] = dict(p=ms[0]["p"], init=ms[0]["init_scale"], lr=ms[0]["lr_const"],
                           t=float(np.mean([m["t_ce80"] for m in ms])),
                           tau=float(np.mean([m["tau_final"] for m in ms])), n=len(ms))
    return out

T = tau_runs()
print("== tau* measurements (lambda = 0) ==")
print(f"{'cell':28s} {'p':>4s} {'init':>4s} {'lr':>5s} {'t_esc':>6s} {'tau*':>10s} {'tau*/p^2.5':>10s}")
for k, v in sorted(T.items(), key=lambda kv: (kv[1]["p"], kv[1]["init"], kv[1]["lr"])):
    print(f"{k:28s} {v['p']:4d} {v['init']:4.0f} {v['lr']:5.1f} {v['t']:6.0f} {v['tau']:10.4g} {v['tau']/v['p']**2.5:10.3f}")

TAU = {v["p"]: v["tau"] for v in T.values() if v["init"] == 4 and v["lr"] == 12.8}
ps = sorted(TAU)
A = np.polyfit([math.log(p) for p in ps], [math.log(TAU[p]) for p in ps], 1)
print(f"\nfit  tau* = exp({A[1]:.3f}) * p^{A[0]:.3f}   (init 4, lr 12.8, m {M})")
def tau_of(p):
    return TAU[p] if p in TAU else math.exp(A[1]) * p ** A[0]

# ---- Xi for every archived x^2 cell, against what actually happened -------------------------------------
CE = json.load(open(P / "ceiling.json"))
print("\n== escape number Xi = s0 / (2 lam tau*(p))  vs the archived outcome ==")
print(f"{'p':>4s} {'init':>4s} {'s0':>7s} {'tau*':>9s} {'Xi':>5s} {'1-1/Xi':>7s} | {'fit':>5s} {'acc':>6s} "
      f"{'margin':>7s} {'gamma/gcon':>10s} {'clean':>5s} {'N_.05':>5s} {'K_5pct':>6s}")
rows = []
for d in sorted(R.glob("n2_pow_p*_l2q2_lam0.0001_lrmax12.8_init*_m500_st40000")):
    p = int(re.search(r"_p(\d+)_", d.name).group(1)); init = float(re.search(r"_init(\d+)", d.name).group(1))
    s0 = init * math.sqrt((M + p) / 3.0)
    t = tau_of(p); xi = s0 / (2e-4 * t)
    ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
    fit = [m for m in ms if m["acc"] >= 0.999]
    gc = max(float(v) for v in CE[str(p)]["gcon"].values())
    med = (lambda k: float(np.median([m[k] for m in fit]))) if fit else (lambda k: float("nan"))
    q = med("gamma") / gc if fit else float("nan")
    rows.append((p, init, xi, q, len(fit), len(ms)))
    print(f"{p:4d} {init:4.0f} {s0:7.1f} {t:9.3g} {xi:5.2f} {max(0,1-1/xi):7.3f} | {len(fit):2d}/{len(ms):<2d} "
          f"{float(np.median([m['acc'] for m in ms])):6.4f} {med('margin'):7.3f} {q:10.3f} {med('clean_frac'):5.3f} "
          f"{med('N_05'):5.1f} {med('K_5pct'):6.1f}")

ok = [(x, q) for _, _, x, q, nf, _ in rows if nf > 0 and np.isfinite(q)]
ok.sort()
from scipy.stats import spearmanr
r, pv = spearmanr([a for a, _ in ok], [b for _, b in ok])
print(f"\nquality gamma/gcon vs Xi over the {len(ok)} cells that fit: Spearman rho = {r:.3f} (p = {pv:.2g})")
print("  Xi:      " + " ".join(f"{a:5.2f}" for a, _ in ok))
print("  gam/gc:  " + " ".join(f"{b:5.3f}" for _, b in ok))
json.dump(dict(tau=T, fit=[float(A[0]), float(A[1])],
               xi=[dict(p=p, init=i, xi=x, q=q, n_fit=nf, n=n) for p, i, x, q, nf, n in rows]),
          open(P / "xi.json", "w"), indent=1)
