"""pax_axis.py -- the prime-axis table: archived recipe vs the fixed recipe, at each lambda rung.
Columns are the three pre-registered count instruments plus the quality gamma/gamma_con(K*)."""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; P = ROOT / "FINAL_RUNS" / "PAX"; R = ROOT / "FINAL_RUNS" / "REP"
CE = json.load(open(P / "ceiling.json")); XI = {(x["p"], x["init"]): x["xi"] for x in json.load(open(P / "xi.json"))["xi"]}
PRIMES = [59, 71, 89, 97, 113, 127, 157, 211, 307, 509]

def row(d, p):
    ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
    if not ms:
        return None
    gc = max(float(v) for v in CE[str(p)]["gcon"].values()) if str(p) in CE else float("nan")
    fit = [m for m in ms if m["acc"] >= 0.999]; pos = [m for m in fit if m["margin"] > 0]
    F = fit or ms; q = lambda k: float(np.median([m[k] for m in F]))
    return dict(n=len(ms), fit=len(fit), pos=len(pos), acc=q("acc"), margin=q("margin"), gamma=q("gamma"),
                qual=q("gamma") / gc, norm2=q("norm2"), clean=q("clean_frac"),
                K5=q("K_5pct"), K2=q("K_2pct"), N05=q("N_05"),
                K5r=(min(m["K_5pct"] for m in F), max(m["K_5pct"] for m in F)),
                esc=(int(np.median([m.get("t_ce80") for m in ms if m.get("t_ce80")])) if any(m.get("t_ce80") for m in ms) else None))

groups = [("archived REP (ramp, penalty on, 40k, init 4)",
           lambda p: R / f"n2_pow_p{p}_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000"),
          ("FIX lam 1e-4 (warm-up, lr 12.8, 25k)", lambda p: P / "pod" / f"paxB_p{p}_lam1e-4"),
          ("FIX lam 1e-5 (warm-up, lr 12.8, 25k)", lambda p: P / "pod" / f"paxC_p{p}_lam1e-5"),
          ("FIX lam 1e-5, p509", lambda p: P / "pod" / f"paxE_p{p}_lam1e-5")]
for name, f in groups:
    any_ = False
    for p in PRIMES:
        d = f(p)
        if not d.exists():
            continue
        r = row(d, p)
        if r is None:
            continue
        if not any_:
            print(f"\n== {name} ==")
            print(f"{'p':>4s} {'n_f':>4s} {'n':>3s} {'fit':>5s} {'m>0':>5s} {'esc':>6s} {'margin':>7s} "
                  f"{'quality':>7s} {'clean':>5s} {'K_5pct':>6s} {'K_2pct':>6s} {'N_.05':>6s} {'m/ln(p-1)':>9s}")
            any_ = True
        import math
        print(f"{p:4d} {(p-1)//2:4d} {r['n']:3d} {r['fit']:2d}/{r['n']:<2d} {r['pos']:2d}/{r['n']:<2d} "
              f"{(r['esc'] if r['esc'] else -1):6d} {r['margin']:7.3f} {r['qual']:7.3f} {r['clean']:5.3f} "
              f"{r['K5']:6.1f} {r['K2']:6.1f} {r['N05']:6.1f} {r['margin']/math.log(p-1):9.2f}")
for extra in sorted(P.glob("pod/paxD*")) + sorted(P.glob("pod/paxCTL*")):
    p = int(re.search(r"_p(\d+)", extra.name).group(1)); r = row(extra, p)
    if r:
        print(f"{extra.name:28s} p{p} n{r['n']} fit {r['fit']} m>0 {r['pos']} margin {r['margin']:.3f} "
              f"quality {r['qual']:.3f} clean {r['clean']:.3f} K5 {r['K5']:.1f} K2 {r['K2']:.1f} N.05 {r['N05']:.1f}")
