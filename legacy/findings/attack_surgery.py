"""Verify surgery claims (T25-T31) from raw spectra + committee-detector
ambiguity + menu-closure triviality.

1. For each surgery arm: final committee (detector), target-freq trajectory
   peaks, grok epoch, and drift check (committee at 8k/10k vs 12k).
2. Control vs original og_seed0/seed27058: committee + grok epoch.
3. Detector ambiguity across the 31 census runs: largest vs second-largest
   log-gap; runs where the cut is fragile.
4. Menu closure: containment of final committee in top-6/top-8 at
   e2000/e3000/e4000 — how much slack does "top-8 at e3000" have?
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

print("=== 1/2. surgery arms ===")
for arm in ("control", "collision", "boost_strong", "suppress", "boost_subtle"):
    d = ROOT / "runs" / "surgery" / arm
    z = np.load(d / "spectra.npz")
    ep, c, ta = z["epochs"], np.abs(z["coeffs"]), z["test_acc"]
    gi = np.argmax(ta >= 0.99)
    grok = int(ep[gi]) if ta[gi] >= 0.99 else None
    comm_final = committee_from_coeffs(z["coeffs"][-1])
    drift = {}
    for e_chk in (8000, 10000):
        i = int(np.argmin(np.abs(ep - e_chk)))
        drift[e_chk] = committee_from_coeffs(z["coeffs"][i])
    # trajectories of the interesting freqs
    interesting = {"control": [14, 49, 52], "collision": [12, 14, 49, 52],
                   "boost_strong": [7, 14, 49, 52], "suppress": [49, 14, 52],
                   "boost_subtle": [7, 14, 49, 52]}[arm]
    peaks = {k: float(c[:, k - 1].max()) for k in interesting}
    finals = {k: float(c[-1, k - 1]) for k in interesting}
    print(f"\n{arm}: grok {grok}, final committee {comm_final}")
    print(f"  committee @8k {drift[8000]} @10k {drift[10000]} @12k {comm_final}")
    print(f"  peaks {peaks}")
    print(f"  final coeffs {finals}")
    print(f"  final test acc {float(ta[-1]):.4f}, last epoch {int(ep[-1])}")

# original base run
d = ROOT / "runs" / "og_seed0" / "seed27058"
z = np.load(d / "spectra.npz")
ep, ta = z["epochs"], z["test_acc"]
gi = np.argmax(ta >= 0.99)
print(f"\noriginal seed27058: grok {int(ep[gi])}, "
      f"final committee {committee_from_coeffs(z['coeffs'][-1])}, "
      f"last epoch {int(ep[-1])}")

print("\n=== 3. detector ambiguity (31 census runs) ===")
frag = 0
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99:
        continue
    a = np.abs(coeffs)
    order = np.argsort(a)[::-1]
    logs = np.log(a[order] + 1e-12)
    gaps = logs[:-1] - logs[1:]
    g = gaps[:12]
    top2 = np.sort(g)[::-1][:2]
    ratio = top2[0] / top2[1]
    cut = int(np.argmax(g)) + 1
    alt = int(np.argsort(g)[::-1][1]) + 1
    tag = ""
    if ratio < 1.5:
        frag += 1
        tag = f"  <-- FRAGILE (2nd gap would give K={alt})"
    print(f"  {str(d.relative_to(ROOT/'runs')):<34} K={cut}  gap ratio {ratio:5.2f}{tag}")
print(f"fragile cuts (ratio < 1.5): {frag}")

print("\n=== 4. menu-closure slack ===")
cont = {(n, e): 0 for n in (5, 6, 8) for e in (2000, 3000, 4000)}
tot = 0
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    tot += 1
    final = set(committee_from_coeffs(coeffs))
    for e in (2000, 3000, 4000):
        i = int(np.argmin(np.abs(z["epochs"] - e)))
        for n in (5, 6, 8):
            top = set((np.argsort(np.abs(z["coeffs"][i]))[::-1][:n] + 1).tolist())
            cont[(n, e)] += final <= top
for e in (2000, 3000, 4000):
    print(f"  e{e}: final within top-5: {cont[(5,e)]}/{tot}, "
          f"top-6: {cont[(6,e)]}/{tot}, top-8: {cont[(8,e)]}/{tot}")
