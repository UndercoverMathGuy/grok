"""Across-dynamics twin test (claim-2 re-review): the well-posed version of
the orthWE necessity question, enabled by the eff/phase2 families.

Cohort: orth-init runs in cell (p=113, data_seed=2034) — the same 4 flat
inits (seeds 11285, 33428, 4242, 777) trained under up to ~10 different
DYNAMICS variants (plain, SAM-noise x2, tilted ERM, CVaR, wd sweeps, ...).
The init energy tilt is identically zero in every one, so any within-seed
committee agreement across dynamics is carried by the init draw's GEOMETRY.

Statistic: mean pairwise Jaccard of final committees between runs sharing
the init seed (different dynamics) vs runs with different init seeds.
Permutation null: shuffle init-seed labels within each family (10k iters).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
from mask_lottery import committee_from_coeffs

ORTH = {"orthWE", "phase2-noise", "phase2-noise2", "phase2-tilt",
        "eff-A", "eff-B", "eff-C", "eff-D", "eff-E", "eff-G", "combined"}

runs = []
for cj in sorted((ROOT / "runs").rglob("config.json")):
    d = cj.parent
    fam = str(d.relative_to(ROOT / "runs")).split("/")[0]
    if fam not in ORTH or not (d / "spectra.npz").exists():
        continue
    c = json.loads(cj.read_text())
    if c["p"] != 113 or c["data_seed"] != 2034:
        continue
    z = np.load(d / "spectra.npz")
    if float(z["test_acc"][-1]) < 0.99:
        continue
    comm = frozenset(committee_from_coeffs(z["coeffs"][-1]))
    runs.append((fam, c["init_seed"], comm))
    print(f"{fam:<16} i{c['init_seed']:<7} {sorted(comm)}")

def jac(a, b):
    return len(a & b) / len(a | b)

def stat(labels):
    w, a = [], []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            if runs[i][0] == runs[j][0]:
                continue  # same family never compared (no self-pairs)
            (w if labels[i] == labels[j] else a).append(
                jac(runs[i][2], runs[j][2]))
    return np.mean(w), np.mean(a), len(w), len(a)

labels = [r[1] for r in runs]
w, a, nw, na = stat(labels)
print(f"\nn runs = {len(runs)}, families = {len(set(r[0] for r in runs))}")
print(f"within-init-seed (across dynamics): mean J = {w:.3f}  (n pairs {nw})")
print(f"across-init-seed:                   mean J = {a:.3f}  (n pairs {na})")

rng = np.random.default_rng(0)
diffs = []
fams = [r[0] for r in runs]
idx_by_fam = {}
for i, f in enumerate(fams):
    idx_by_fam.setdefault(f, []).append(i)
obs = w - a
cnt = 0
for _ in range(10000):
    lab = list(labels)
    for f, idxs in idx_by_fam.items():
        vals = [lab[i] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            lab[i] = v
    w_, a_, _, _ = stat(lab)
    cnt += (w_ - a_) >= obs
print(f"observed diff {obs:+.3f}, permutation p = {cnt / 10000:.4f}")
