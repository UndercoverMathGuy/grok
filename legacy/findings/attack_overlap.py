"""Adversarial re-analysis of T8/S1 (within-mask overlap 0.688 vs 0.300).

Problems with the original test (overlap_check.py):
 1. permutes PAIR values, not run labels — pairs sharing a run are dependent,
    so the permutation unit is wrong;
 2. each permutation iteration draws TWO independent permutations, so the two
    groups double-count elements (not a partition);
 3. within-mask pairs are dominated by one mask (dseed 0 has 17 of the p=113
    runs -> 136 of the within pairs).

Correct test: permute the mask label ACROSS RUNS (preserving mask
multiplicities), recompute the within-across difference each time.
Also: per-mask within-overlap means, and the same test excluding dseed 0.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

rows = []
for d, cfg in discover():
    coeffs, acc, _ = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or cfg.p != 113:
        continue
    rows.append((cfg.data_seed, frozenset(committee_from_coeffs(coeffs))))

labels = np.array([r[0] for r in rows])
comms = [r[1] for r in rows]
n = len(rows)
print(f"{n} p=113 runs; mask counts: "
      f"{dict(zip(*np.unique(labels, return_counts=True)))}")

ov = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        ov[i, j] = len(comms[i] & comms[j])

def stat(lab):
    w, a = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (w if lab[i] == lab[j] else a).append(ov[i, j])
    return np.mean(w), np.mean(a), len(w), len(a)

w0, a0, nw, na = stat(labels)
print(f"within {w0:.3f} (n={nw} pairs), across {a0:.3f} (n={na} pairs), "
      f"diff {w0 - a0:.3f}")

rng = np.random.default_rng(0)
null = np.array([(lambda s: s[0] - s[1])(stat(rng.permutation(labels)))
                 for _ in range(20000)])
print(f"RUN-LEVEL permutation: p = {(null >= w0 - a0).mean():.5f} "
      f"(null {null.mean():.3f} +/- {null.std():.3f})")

# per-mask within means
print("\nper-mask within-overlap means:")
for m in np.unique(labels):
    idx = np.flatnonzero(labels == m)
    if len(idx) < 2:
        continue
    vals = [ov[i, j] for k, i in enumerate(idx) for j in idx[k+1:]]
    print(f"  dseed {m}: {np.mean(vals):.3f} (n={len(vals)} pairs, {len(idx)} runs)")

# excluding the dominant mask (dseed 0)
keep = labels != 0
lab2 = labels[keep]; idx2 = np.flatnonzero(keep)
ov2 = ov[np.ix_(idx2, idx2)]
n2 = len(idx2)
def stat2(lab):
    w, a = [], []
    for i in range(n2):
        for j in range(i + 1, n2):
            (w if lab[i] == lab[j] else a).append(ov2[i, j])
    return np.mean(w), np.mean(a)
w1, a1 = stat2(lab2)
null2 = np.array([(lambda s: s[0] - s[1])(stat2(rng.permutation(lab2)))
                  for _ in range(20000)])
print(f"\nexcluding dseed 0 ({n2} runs): within {w1:.3f}, across {a1:.3f}, "
      f"diff {w1-a1:.3f}, run-level perm p = {(null2 >= w1 - a1).mean():.5f}")
