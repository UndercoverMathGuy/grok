"""Phase-2 scoreboard: does selection pressure beat the flat-start lottery?

Arms (all orthogonal W_E, p=113, cell 2034, same 4 init seeds):
  phase1   flat start, no pressure          runs/orthWE/p-113/seed2034
  noise    + SAM-lite sigma=0.3 -> 0 @6k    runs/phase2-noise/p-113/seed2034
  tilt     + tilted ERM t=5                 runs/phase2-tilt/p-113/seed2034

Per arm: committee, lp_relM, percentile vs random K-subsets; paired deltas
vs phase1 on shared init seeds. Baseline (normal W_E) pooled numbers are
printed for reference.

Run: uv run python scripts/phase2_analysis.py
"""

from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from orthwe_analysis import discover, analyze_runs, jaccard

RUNS = Path("runs")

ARMS = [
    ("phase1", RUNS / "orthWE" / "p-113"),
    ("noise", RUNS / "phase2-noise" / "p-113"),
    ("tilt", RUNS / "phase2-tilt" / "p-113"),
    ("noise2", RUNS / "phase2-noise2" / "p-113"),  # sigma 0.2 sustained to 16k
]


def main():
    rng = np.random.default_rng(0)
    rows = {}
    for name, root in ARMS:
        runs = [r for r in discover(root) if r[0] == 2034]
        if runs:
            rows[name] = analyze_runs(runs, f"{name} ({len(runs)} runs, cell 2034)", rng)

    p1 = {r["iseed"]: r for r in rows.get("phase1", [])}
    for arm in ("noise", "tilt", "noise2"):
        if arm not in rows:
            continue
        print(f"\n=== paired: {arm} vs phase1 (same init seed) ===")
        deltas = []
        for r in rows[arm]:
            b = p1.get(r["iseed"])
            if b is None:
                continue
            deltas.append(r["pct"] - b["pct"])
            print(f"  i{r['iseed']}: {arm} {set(r['comm'])} pct {r['pct']:.1f} "
                  f"vs phase1 {set(b['comm'])} pct {b['pct']:.1f}  "
                  f"(J {jaccard(r['comm'], b['comm']):.2f}, "
                  f"delta pct {r['pct'] - b['pct']:+.1f})")
        if deltas:
            print(f"  mean delta percentile: {np.mean(deltas):+.1f}")

    print("\n=== pooled percentiles ===")
    for name in rows:
        pcts = [r["pct"] for r in rows[name]]
        accs = [r["acc"] for r in rows[name]]
        print(f"{name:>7}: mean {np.mean(pcts):5.1f}  min {np.min(pcts):5.1f}  "
              f"acc mean {np.mean(accs):.3f}  (n={len(pcts)})")
    print("baseline normal W_E reference (all cells): mean 79.8 (n=4)")


if __name__ == "__main__":
    main()
