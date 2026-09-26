"""pax_report.py -- score the pax_job2 warm-up cells against the archived REP baseline at the same prime.
quality = gamma / gamma_con(K*)  with gamma_con from pax_ceiling.py (exact constructed junk-free ceiling)."""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; P = ROOT / "FINAL_RUNS" / "PAX"; R = ROOT / "FINAL_RUNS" / "REP"
CE = json.load(open(P / "ceiling.json"))

def show(label, d, p):
    ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
    if not ms:
        return
    gc = max(float(v) for v in CE[str(p)]["gcon"].values())
    fit = [m for m in ms if m["acc"] >= 0.999]
    pos = [m for m in ms if m["acc"] >= 0.999 and m["margin"] > 0]
    F = fit or ms
    med = lambda k: float(np.median([m[k] for m in F]))
    esc = [m.get("t_ce80") or m.get("escape_step") for m in ms]
    print(f"{label:34s} p{p:4d} n{len(ms)}  fit {len(fit)}/{len(ms)} pos-margin {len(pos)}/{len(ms)}  "
          f"esc {min(e for e in esc if e) if any(esc) else '-':>6} | acc {med('acc'):.5f} margin {med('margin'):7.3f} "
          f"gamma {med('gamma'):.4e} **q {med('gamma')/gc:.3f}** ||th||^2 {med('norm2'):6.0f} clean {med('clean_frac'):.3f} "
          f"K_5pct {med('K_5pct'):4.1f} K_2pct {med('K_2pct'):4.1f} N_.05 {med('N_05'):5.1f} "
          f"nfreq_w {med('n_freq_hist_1pct'):4.1f}")

for p in (113, 157, 211):
    print(f"--- p = {p}  (K* {CE[str(p)]['Kstar']}, n_f {CE[str(p)]['nf']}, gamma_con(K*) "
          f"{max(float(v) for v in CE[str(p)]['gcon'].values()):.4e}) ---")
    a = R / f"n2_pow_p{p}_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000"
    if a.exists():
        show("ARCHIVED x2 ramp+decay 40k", a, p)
    b = R / f"n2_prod_p{p}_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000_clip1_wdw20000_esc0.8_lr212.8"
    if b.exists():
        show("ARCHIVED bilinear warm-up 40k", b, p)
    for d in sorted(P.glob(f"pax_fix_p{p}_*")):
        show("PAX " + d.name, d, p)
    print()
