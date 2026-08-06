"""Within-mask vs across-mask committee overlap at p=113 (the anchor fact).

Reproduces: within-mask mean shared members 0.688 (144 pairs) vs across-mask
0.300 (207 pairs) vs uniform K=4 null 0.279; permutation p < 2e-4.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

rows = []
for d, cfg in discover():
    coeffs, acc, _ = final_coeffs_and_acc(d, cfg)
    if acc < 0.99:
        continue
    rows.append(((cfg.p, cfg.data_seed), set(committee_from_coeffs(coeffs))))

within, across = [], []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        (p1, d1), c1 = rows[i]
        (p2, d2), c2 = rows[j]
        if p1 != p2 or p1 != 113:
            continue
        (within if d1 == d2 else across).append(len(c1 & c2))

rng = np.random.default_rng(0)
null = [len(set(rng.choice(56, 4, replace=False) + 1)
            & set(rng.choice(56, 4, replace=False) + 1)) for _ in range(20000)]
print(f"within-mask : mean {np.mean(within):.3f} (n={len(within)} pairs)")
print(f"across-mask : mean {np.mean(across):.3f} (n={len(across)} pairs)")
print(f"uniform null (K=4): {np.mean(null):.3f}")
w, a = np.array(within), np.array(across)
comb = np.concatenate([w, a]); nw = len(w)
obs = w.mean() - a.mean()
perm = [comb[rng.permutation(len(comb))][:nw].mean()
        - comb[rng.permutation(len(comb))][nw:].mean() for _ in range(5000)]
print(f"within-across diff {obs:.3f}, permutation p = "
      f"{(np.array(perm) >= obs).mean():.4f}")
