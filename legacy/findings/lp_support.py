"""Proxy 1: is the final committee the SUPPORT of the max-min-margin LP over
the audition menu?  Menus: top-8 by |coeff| at memorization end and at e3000.
Baseline: top-K-by-amplitude from the same menu (K = final committee size)."""
import sys
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

rows = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p
    final = committee_from_coeffs(coeffs)
    for label, pick in (("mem", None), ("e3000", 3000)):
        if pick is None:
            idx = int(np.argmax(z["train_acc"] >= 0.995))
            if z["train_acc"][idx] < 0.995:
                continue
        else:
            idx = int(np.argmin(np.abs(z["epochs"] - pick)))
        menu = (np.argsort(np.abs(z["coeffs"][idx]))[::-1][:8] + 1).tolist()
        relm, alloc = lp_relM(menu, p)
        for thr in (0.01, 0.05):
            supp = sorted(np.array(menu)[alloc > thr * alloc.sum()].tolist())
            rows.append(dict(run=str(d.relative_to(ROOT/"runs")), p=p,
                             label=f"{label}/thr{thr}", menu=menu, supp=supp,
                             final=sorted(final),
                             contain=set(final) <= set(menu)))

# per-condition summary
from collections import defaultdict
by = defaultdict(list)
for r in rows:
    by[r["label"]].append(r)
for label, rs in sorted(by.items()):
    exact = sum(r["supp"] == r["final"] for r in rs)
    jacs = [jac(r["supp"], r["final"]) for r in rs]
    # baseline: top-K amplitude from same menu
    base = []
    for r in rs:
        K = len(r["final"])
        base.append(jac(r["menu"][:K], r["final"]))
    cont = sum(r["contain"] for r in rs)
    print(f"{label}: exact {exact}/{len(rs)}  mean Jaccard {np.mean(jacs):.3f} "
          f"(amp-topK baseline {np.mean(base):.3f})  "
          f"[final within menu: {cont}/{len(rs)}]")

print("\nper-run detail (mem/thr0.01):")
for r in by["mem/thr0.01"]:
    flag = "EXACT" if r["supp"] == r["final"] else ""
    print(f"  {r['run']:<34} menu-top8 {r['menu']}")
    print(f"    LP supp {r['supp']}  final {r['final']}  {flag}")
