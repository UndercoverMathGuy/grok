"""pax_ceiling.py -- the exact constructed gamma ceiling gamma_con(K, p) for the x^2 net at every prime on the
prime axis, and the closed form for it.  Analysis only.

gamma = (min margin over the p^2 table) / ||theta||^3   (nu = 3, scale free).
The junk-free 6-neuron-per-frequency construction (initaudit_ceiling.build_x2) with unit amplitude gives
   z(c; a, b) = 3 * sum_{k in S} cos(2 pi k (a + b - c) / p)          (verified numerically below)
so  min margin = 3 * M_S,   M_S = min_{D != 0} sum_{k in S} (1 - cos(2 pi k D / p)),
and ||theta||^2 = 6 neurons * 3 tables * (p/2) per frequency = 9 p K.  Hence the CLOSED FORM

   gamma_con(K, p) = 3 * M_K(p) / (9 p K)^{3/2}        ~   M_K / K^{3/2}  *  p^{-3/2} * (1/18)

-> the ceiling falls like p^{-3/2} at fixed K.  That single factor is what the escape and the quality story run on.
"""
import json, math, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from audit_common import rows, logits, margin          # noqa: E402
from kstar import best_subset                          # noqa: E402
from initaudit_ceiling import build_x2                 # noqa: E402
torch.set_default_dtype(torch.float64)
OUT = ROOT / "FINAL_RUNS" / "PAX"; OUT.mkdir(parents=True, exist_ok=True)
PRIMES = [59, 71, 89, 97, 113, 127, 157, 211, 307]
KMAX = 12

# ---- 1. verify the closed form on the exact construction at two primes -------------------------------------
KS = json.loads((ROOT / "kstar.json").read_text())
for p in (59, 71):
    ops, y = rows(p, 2)
    for K in (3, 5, 8):
        S = KS[str(p)]["S"][str(K)]
        E, W = build_x2(p, S)
        mg = float(margin(logits(E, W.T, ops, "pow"), y).min())
        n2 = float(E.square().sum() + W.square().sum())
        M = float(min((1 - np.cos(2 * np.pi * np.array(S) * D / p)).sum() for D in range(1, p)))
        print(f"check p{p} K{K}: margin {mg:.5f} vs 3*M_S {3 * M:.5f} | ||th||^2 {n2:.1f} vs 9pK {9 * p * K}"
              f" | gamma {mg / n2 ** 1.5:.4e} vs closed {3 * M / (9 * p * K) ** 1.5:.4e}")

# ---- 2. ladders and ceilings at every prime ---------------------------------------------------------------
rng = np.random.default_rng(0)
lad = {}
for p in PRIMES:
    Ms, Ss = {}, {}
    for K in range(1, KMAX + 1):
        M, S, how = best_subset(p, K, rng, restarts=120)
        Ms[K] = M; Ss[K] = S
    gcon = {K: 3 * Ms[K] / (9 * p * K) ** 1.5 for K in Ms}
    Kst = max(gcon, key=gcon.get)
    lad[p] = dict(nf=(p - 1) // 2, M=Ms, S=Ss, gcon=gcon, Kstar=Kst, gcon_max=gcon[Kst])
    print(f"p {p:4d} nf {(p-1)//2:4d}  K*_con {Kst}  gamma_con(K*) {gcon[Kst]:.4e}  "
          f"p^1.5*gcon {gcon[Kst] * p ** 1.5:.4f}  ladder " + " ".join(f"{K}:{gcon[K]/gcon[Kst]:.2f}" for K in range(3, 11)))
json.dump({str(p): dict(nf=v["nf"], Kstar=v["Kstar"], gcon={str(k): g for k, g in v["gcon"].items()},
                        S={str(k): s for k, s in v["S"].items()}, M={str(k): m for k, m in v["M"].items()})
           for p, v in lad.items()}, open(OUT / "ceiling.json", "w"), indent=1)

# ---- 3. how far are the trained nets from it? -------------------------------------------------------------
R = ROOT / "FINAL_RUNS" / "REP"
print(f"\n{'act':5s} {'p':>4s} {'fit':>5s} {'gamma_med':>10s} {'gamma_con(K*)':>13s} {'ratio':>6s} "
      f"{'margin':>7s} {'||th||^2':>9s} {'clean':>5s} {'N_.05':>6s} {'K_5pct':>6s}")
tab = []
for act, tagf in (("pow", "n2_pow_p{p}_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000"),
                  ("prod", "n2_prod_p{p}_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000_clip1_wdw20000_esc0.8_lr212.8")):
    for p in PRIMES:
        d = R / tagf.format(p=p)
        if not d.exists():
            continue
        ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
        fit = [m for m in ms if m.get("acc", 0) >= 0.999]
        if not fit:
            print(f"{act:5s} {p:4d} {0:2d}/{len(ms):<2d}  -- no fit seed --"); continue
        g = float(np.median([m["gamma"] for m in fit])); gc = lad[p]["gcon_max"]
        r = dict(act=act, p=p, n_fit=len(fit), n=len(ms), gamma=g, gcon=gc, ratio=g / gc,
                 margin=float(np.median([m["margin"] for m in fit])), norm2=float(np.median([m["norm2"] for m in fit])),
                 clean=float(np.median([m["clean_frac"] for m in fit])), N05=float(np.median([m["N_05"] for m in fit])),
                 K5=float(np.median([m["K_5pct"] for m in fit])))
        tab.append(r)
        print(f"{act:5s} {p:4d} {len(fit):2d}/{len(ms):<2d} {g:10.4e} {gc:13.4e} {g/gc:6.3f} {r['margin']:7.3f} "
              f"{r['norm2']:9.0f} {r['clean']:5.3f} {r['N05']:6.1f} {r['K5']:6.1f}")
json.dump(tab, open(OUT / "quality_vs_p.json", "w"), indent=1)
