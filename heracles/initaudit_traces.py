"""initaudit_traces.py -- print the count / gamma trajectories from the local INITAUDIT runs."""
import json, sys, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent
IA = ROOT / "FINAL_RUNS" / "INITAUDIT"
for cell in sorted(IA.glob("*/")):
    fs = sorted(cell.glob("seed*/metrics.json"))
    if not fs: continue
    rs = [json.loads(f.read_text()) for f in fs]
    print(f"\n== {cell.name} (resume={rs[0].get('resume') and Path(rs[0]['resume']).name} init={rs[0]['init_scale']:g}) ==")
    steps = [t["step"] for t in rs[0]["trace"]]
    print("   step  | " + " | ".join(f"seed{r['seed']}: K2 K5 N05  gamma     n2    clean" for r in rs[:2]))
    for i, s in enumerate(steps):
        print(f" {s:6d}  | " + " | ".join(
            f"{r['trace'][i]['K_2pct']:3d}{r['trace'][i]['K_5pct']:3d}{r['trace'][i]['N_05']:4d} "
            f"{r['trace'][i]['gamma']:.3e} {r['trace'][i]['norm2']:7.1f} {r['trace'][i]['clean_frac']:.3f}" for r in rs))
    print("  medians over seeds: " + "  ".join(
        f"step {s}: K2 {st.median([r['trace'][i]['K_2pct'] for r in rs]):.1f} "
        f"gamma {st.median([r['trace'][i]['gamma'] for r in rs]):.3e}" for i, s in enumerate(steps)))
