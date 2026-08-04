"""CLAIM 2 — the init (not training noise) chooses, shown properly.

Three analyses over every compatible run (no training, spectra only):

  A. Across-dynamics twins: among orth-flat runs sharing a (p, data_seed)
     cell, the SAME init seed trained under different dynamics families
     (plain / noise / tilt / CVaR / wd sweeps ...) vs different init seeds.
     Permutation test (init-seed labels shuffled within family, 10k iters).
     Reported twice: per FAMILY (ten variants; inflated by near-duplicate
     tilt-5 recipes) and per RECIPE GROUP (families collapsed into
     qualitatively distinct dynamics; only cross-group pairs count — the
     honest version).
  B. Cross-flattening same-seed table: for init seeds present at several
     flattening levels (normal / orth-flat / double-flat) in the same cell,
     the committee overlap between levels — expected at stranger baseline
     (each flattening re-rolls the lottery).
  C. Paired normal-vs-orth twins (the original orthWE observation, kept as
     description, not proof).
"""
import numpy as np

from common import discover, jaccard

runs = list(discover())

# family -> qualitatively-distinct recipe group. Six of the ten original
# families are tilt-5 variants (one knob nudged each) and some give
# literally identical committees for the same init — cross-family pairs
# inside that cluster are near-replicates, not independent dynamics draws.
GROUPS = {"orthWE": "plain", "orthfresh": "plain",
          "phase2-tilt": "tilt", "eff-A": "tilt", "eff-B": "tilt",
          "eff-D": "tilt", "eff-E": "tilt", "combined": "tilt",
          "phase2-noise": "noise", "phase2-noise2": "noise",
          "eff-G": "cvar",
          "dyn-wd25": "wd", "dyn-wd04": "wd",
          "dyn-lr3": "lr", "dyn-lrlo": "lr"}

# --- A: across-dynamics twins ----------------------------------------------
print("=== A. across-dynamics twins (orth-flat, per cell) ===")
by_cell = {}
for r in runs:
    if r["cohort"] == "orth-flat":
        by_cell.setdefault((r["cfg"].p, r["cfg"].data_seed), []).append(r)
rng = np.random.default_rng(0)
unknown = {r["fam"] for rs in by_cell.values() for r in rs} - set(GROUPS)
if unknown:
    print(f"WARNING: families not in GROUPS, each treated as its own recipe "
          f"group (may inflate cross-group pairs): {sorted(unknown)}")
for cell, rs in sorted(by_cell.items()):
    fams = {}
    for r in rs:
        fams.setdefault(r["fam"], []).append(r)
    if len(fams) < 2:
        print(f"cell {cell}: only {len(fams)} dynamics family — skipped")
        continue
    items = [(r["fam"], r["cfg"].init_seed, frozenset(r["committee"]),
              GROUPS.get(r["fam"], r["fam"]))
             for r in rs]
    n = len(items)
    # pairwise Jaccards once per cell; permutations only relabel.
    J = np.array([[jaccard(items[i][2], items[j][2]) for j in range(n)]
                  for i in range(n)])
    seeds = np.array([x[1] for x in items])
    idx_by_fam = {}
    for i, x in enumerate(items):
        idx_by_fam.setdefault(x[0], []).append(i)
    idx_by_fam = {f_: np.array(v) for f_, v in idx_by_fam.items()}

    for lvl, keyf in (("family", lambda x: x[0]), ("group", lambda x: x[3])):
        keys = [keyf(x) for x in items]
        iu, ju = np.triu_indices(n, 1)
        cross = np.array([keys[i] != keys[j] for i, j in zip(iu, ju)])
        pi, pj, jv = iu[cross], ju[cross], J[iu[cross], ju[cross]]

        def stat(lab):
            same = lab[pi] == lab[pj]
            w_, a_ = jv[same], jv[~same]
            return (w_.mean() if w_.size else np.nan,
                    a_.mean() if a_.size else np.nan, w_.size, a_.size)

        w, a, nw, na = stat(seeds)
        nunit = len(set(keys))
        if nw == 0 or na == 0:
            print(f"cell {cell} [{lvl}: {nunit}] n={n} runs | "
                  f"within-seed pairs {nw}, across-seed pairs {na} — "
                  f"permutation test skipped (undefined)")
            continue
        obs = w - a
        cnt = valid = 0
        for _ in range(10000):
            lab = seeds.copy()
            for idxs in idx_by_fam.values():
                lab[idxs] = rng.permutation(lab[idxs])
            w_, a_, nw_, na_ = stat(lab)
            if nw_ == 0 or na_ == 0:
                continue      # statistic undefined for this relabeling
            valid += 1
            cnt += (w_ - a_) >= obs
        if valid == 0:
            print(f"cell {cell} [{lvl}: {nunit}] n={n} runs | no valid "
                  f"permutations — test skipped")
            continue
        print(f"cell {cell} [{lvl}: {nunit}] n={n} runs | "
              f"within-seed J {w:.3f} ({nw} pairs) vs across-seed {a:.3f} "
              f"({na} pairs) | perm p = {cnt/valid:.4f} "
              f"({valid} valid perms)")

# --- B: cross-flattening same-seed table ------------------------------------
print("\n=== B. same seed across flattening levels (cell p=113 ds=2034) ===")
levels = {"normal": {}, "orth-flat": {}, "double-flat": {}}
for r in runs:
    if (r["cfg"].p, r["cfg"].data_seed) != (113, 2034):
        continue
    lv = ("normal" if r["cohort"] == "natural-normal" else
          r["cohort"] if r["cohort"] in ("orth-flat", "double-flat") else None)
    if lv == "orth-flat" and r["fam"] != "orthWE":
        continue          # one representative per level: plain dynamics only
    if lv:
        seed = r["cfg"].init_seed
        if seed in levels[lv]:
            print(f"WARNING: duplicate {lv} run for init_seed {seed} "
                  f"({r['rel']}) — keeping the first discovered")
            continue
        levels[lv][seed] = sorted(r["committee"])
pairs = []
for seed in sorted(set().union(*[set(v) for v in levels.values()])):
    row = {lv: levels[lv].get(seed) for lv in levels}
    present = [lv for lv in levels if row[lv]]
    if len(present) < 2:
        continue
    line = f"seed {seed}: " + "  ".join(f"{lv}={row[lv]}" for lv in present)
    js = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            js.append(jaccard(row[present[i]], row[present[j]]))
            pairs.append(js[-1])
    print(line + f"   J {['%.2f' % x for x in js]}")
if pairs:
    print(f"mean same-seed cross-level J = {np.mean(pairs):.3f} "
          f"(stranger baseline in this cell ~0.11)")

# --- C: paired normal-vs-orth (descriptive) ---------------------------------
print("\n=== C. paired normal vs orth twins (descriptive) ===")
norm = {(r["cfg"].p, r["cfg"].data_seed, r["cfg"].init_seed): r
        for r in runs if r["cohort"] == "natural-normal"}
for r in runs:
    if r["cohort"] == "orth-flat" and r["fam"] == "orthWE":
        b = norm.get((r["cfg"].p, r["cfg"].data_seed, r["cfg"].init_seed))
        if b:
            print(f"  {r['rel']}: orth {sorted(r['committee'])} vs normal "
                  f"{sorted(b['committee'])}  J "
                  f"{jaccard(r['committee'], b['committee']):.2f}")
print("""
Backs SEMIFINAL claim 2: identity is chosen by the init (A: within-seed
overlap ~3x baseline across ten dynamics), each flattening level re-rolls
the choice completely (B: same-seed cross-level overlap ~ stranger
baseline), and the raw orthWE Jaccards (C) are kept as description only.""")
