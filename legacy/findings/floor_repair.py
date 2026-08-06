"""Greedy-plus-floor-repair as an identity-from-menu decision rule.
Rule(theta): among all subsets of the e3000 top-8 menu with lp_relM >= theta,
pick the smallest K; tie-break by highest total e3000 amplitude.
Compare vs final committee; baseline = amplitude top-K at the TRUE K."""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs
from margin_analysis import lp_relM

def jac(A, B):
    A, B = set(A), set(B)
    return len(A & B) / len(A | B)

runs = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    idx = int(np.argmin(np.abs(z["epochs"] - 3000)))
    amp = np.abs(z["coeffs"][idx])
    menu = (np.argsort(amp)[::-1][:8] + 1).tolist()
    runs.append(dict(name=str(d.relative_to(ROOT/"runs")), p=cfg.p, menu=menu,
                     amp={k: amp[k-1] for k in menu},
                     final=sorted(committee_from_coeffs(coeffs))))

# precompute lp_relM for all subsets K=2..8 of each menu
for r in runs:
    scores = {}
    for K in range(2, 9):
        for S in combinations(r["menu"], K):
            scores[S] = lp_relM(list(S), r["p"])[0]
    r["scores"] = scores

for theta in (0.25, 0.274, 0.30):
    exact = 0; jacs = []; base = []; kmatch = 0
    for r in runs:
        feas = [(len(S), -sum(r["amp"][k] for k in S), S)
                for S, m in r["scores"].items() if m >= theta]
        if not feas:
            pred = tuple(r["menu"])
        else:
            pred = sorted(min(feas)[2])
        Ktrue = len(r["final"])
        jacs.append(jac(pred, r["final"]))
        base.append(jac(sorted(r["menu"][:Ktrue]), r["final"]))
        exact += list(pred) == r["final"]
        kmatch += len(pred) == Ktrue
    print(f"theta={theta}: exact {exact}/{len(runs)}  K-match {kmatch}/{len(runs)}  "
          f"mean Jaccard {np.mean(jacs):.3f}  (amp-topK@trueK baseline "
          f"{np.mean(base):.3f})")

# detail at theta=0.274
theta = 0.274
print("\ndetail (theta=0.274):")
for r in runs:
    feas = [(len(S), -sum(r["amp"][k] for k in S), S)
            for S, m in r["scores"].items() if m >= theta]
    pred = sorted(min(feas)[2]) if feas else r["menu"]
    flag = "EXACT" if list(pred) == r["final"] else ""
    print(f"  {r['name']:<34} pred {pred}  final {r['final']}  {flag}")
