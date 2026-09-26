"""pax_esclr.py -- the SECOND variable: the learning rate at the moment of escape.
The archived recipe ramps lr (0.05 doubled at steps 1000..10000, cap 12.8), so a cell that escapes at step 6000
escapes at lr 3.2 and one that escapes at step 8000 escapes at lr 12.8.  In the archive lr_esc is collinear with
the escape number Xi; the pax warm-up runs break that collinearity (Xi = infinity, lr_esc = 12.8)."""
import json, math, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; R = ROOT / "FINAL_RUNS" / "REP"; P = ROOT / "FINAL_RUNS" / "PAX"
CE = json.load(open(P / "ceiling.json")); XI = {(x["p"], x["init"]): x["xi"] for x in json.load(open(P / "xi.json"))["xi"]}
rows = []
for d in sorted(R.glob("n2_pow_p*_l2q2_lam0.0001_lrmax12.8_init4_m500_st40000")) + \
         sorted(R.glob("n2_pow_p*_l2q2_lam0.0001_lrmax12.8_init[123]_m500_st40000")):
    p = int(re.search(r"_p(\d+)_", d.name).group(1)); init = float(re.search(r"_init(\d+)", d.name).group(1))
    ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
    fit = [m for m in ms if m["acc"] >= 0.999]
    if not fit:
        continue
    esc = int(np.median([next((l["step"] for l in m["log"] if l["acc"] >= 0.99), 10 ** 9) for m in fit]))
    lr_esc = 0.05 * 2 ** min(10, esc // 1000)
    gc = max(float(v) for v in CE[str(p)]["gcon"].values())
    q = float(np.median([m["gamma"] for m in fit])) / gc
    rows.append(dict(p=p, init=init, xi=XI[(p, init)], esc=esc, lr_esc=lr_esc, q=q,
                     clean=float(np.median([m["clean_frac"] for m in fit])),
                     N05=float(np.median([m["N_05"] for m in fit])), K5=float(np.median([m["K_5pct"] for m in fit]))))
rows.sort(key=lambda r: (r["lr_esc"], r["p"]))
print(f"{'p':>4s} {'init':>4s} {'Xi':>6s} {'esc step':>8s} {'lr@esc':>6s} {'quality':>7s} {'clean':>5s} {'N_.05':>5s} {'K_5pct':>6s}")
for r in rows:
    print(f"{r['p']:4d} {r['init']:4.0f} {r['xi']:6.2f} {r['esc']:8d} {r['lr_esc']:6.1f} {r['q']:7.3f} "
          f"{r['clean']:5.3f} {r['N05']:5.1f} {r['K5']:6.1f}")
for L in sorted({r["lr_esc"] for r in rows}):
    g = [r["q"] for r in rows if r["lr_esc"] == L]
    print(f"lr@escape {L:5.1f}: n {len(g):2d}  quality {np.median(g):.3f} [{min(g):.3f}-{max(g):.3f}]")
try:
    from scipy.stats import mannwhitneyu
    a = [r["q"] for r in rows if r["lr_esc"] <= 3.2]; b = [r["q"] for r in rows if r["lr_esc"] > 3.2]
    print(f"Mann-Whitney lr<=3.2 (n {len(a)}) vs lr 12.8 (n {len(b)}): U p = {mannwhitneyu(a, b).pvalue:.4f}")
    print("  NOTE: in the archive lr@escape is collinear with Xi (every lr 3.2 cell has Xi >= 3.37,"
          " every lr 12.8 cell has Xi <= 2.86), so this split alone cannot separate the two.")
except ImportError:
    pass
for d in sorted(P.glob("pax_*_p*_i4_wu*")):
    ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
    if not ms:
        continue
    p = ms[0]["p"]; gc = max(float(v) for v in CE[str(p)]["gcon"].values())
    print(f"PAX {d.name:32s} Xi inf  esc {ms[0]['t_ce80']:5d}  lr@esc {ms[0]['lr_const']:5.1f}  "
          f"quality {np.median([m['gamma'] for m in ms]) / gc:.3f}  clean {np.median([m['clean_frac'] for m in ms]):.3f} "
          f" margin {np.median([m['margin'] for m in ms]):.3f}  N_.05 {np.median([m['N_05'] for m in ms]):.1f}"
          f"  K_5pct {np.median([m['K_5pct'] for m in ms]):.1f}")
