"""pax_score.py -- score any pax_run cell directory against the exact constructed ceiling.
  uv run python heracles/pax_score.py FINAL_RUNS/PAX/paxA_*    (quality = gamma/gamma_con(K*))"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
CEIL = json.load(open(ROOT / "FINAL_RUNS" / "PAX" / "ceiling.json"))
print(f"{'cell':34s} {'p':>4s} {'n':>3s} {'fit':>5s} {'m>0':>5s} {'esc':>6s} {'acc':>7s} {'margin':>8s} "
      f"{'gamma':>10s} {'qual':>5s} {'||th||2':>8s} {'clean':>5s} {'K5':>4s} {'K2':>4s} {'N.05':>5s}")
for a in sys.argv[1:]:
    d = Path(a)
    ms = [json.loads((s / "metrics.json").read_text()) for s in sorted(d.glob("seed*"))]
    if not ms:
        continue
    p = ms[0]["p"]; gc = max(float(v) for v in CEIL[str(p)]["gcon"].values())
    fit = [m for m in ms if m["acc"] >= 0.999]
    pos = [m for m in fit if m["margin"] > 0]
    F = fit or ms
    q = lambda k: float(np.median([m[k] for m in F]))
    rng = lambda k: f"[{min(m[k] for m in F):.0f}-{max(m[k] for m in F):.0f}]"
    esc = [m["t_ce80"] for m in ms if m["t_ce80"]]
    print(f"{d.name:34s} {p:4d} {len(ms):3d} {len(fit):2d}/{len(ms):<2d} {len(pos):2d}/{len(ms):<2d} "
          f"{(int(np.median(esc)) if esc else -1):6d} {q('acc'):7.5f} {q('margin'):8.3f} {q('gamma'):10.3e} "
          f"{q('gamma')/gc:5.3f} {q('norm2'):8.0f} {q('clean_frac'):5.3f} {q('K_5pct'):4.1f} {q('K_2pct'):4.1f} "
          f"{q('N_05'):5.1f}  K5{rng('K_5pct')} K2{rng('K_2pct')} N{rng('N_05')}")
