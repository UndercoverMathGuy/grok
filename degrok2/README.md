# De-grok 2 — how far the two clocks pervade

Adopted 2026-08-16. The v1 de-grok cohort (degrok/) established the
two-clock dissociation on mod-add p=113 across frac_train 0.3-0.9 with a
p=127 anchor — one task family, one operation, one width, four fractions.
That is thin support for a "property of training" claim. This cohort
widens every axis at once:

- **finer/wider data grid** (frac 0.2-0.9, 7 points, incl. fresh
  replication of the v1 endpoints on new seeds),
- **a second operation** (subtraction, a-b mod p — new `fn_name="sub"`
  branch in the trainer; same Fourier instrument applies),
- **more moduli** (composite p=96 across fractions; p=127 extended
  beyond frac 0.3),
- **width** (d_mlp 128 and 2048 at the frac extremes),
- **a non-abelian task** (S5 across frac 0.6-0.9, irrep instrument).

Everything is scored per family so a failure in one family scopes the
claim rather than killing it — pre-declared below.

Known confounds, inherited from v1 and designed around identically:
mask sets the winner (committee identity across fracs is exploratory,
never scored); the AUC instrument saturates late (t_ident lives in the
scored early window); S5's wd-coupling is a KNOWN boundary (tasks/), so
no wd-invariance prediction is made for S5 — only the data-clock
dissociation is scored there.

## Design

FRESH seeds: META_SEED=20260818; 2 data masks from [40000, 50000)
(disjoint from every previously spent range) x 4 init seeds. d_model 128,
1 layer, 4 heads, d_mlp 512 unless stated. Epoch budgets: 30k for
frac < 0.3, 15k for frac 0.3, 10k otherwise; S5 100k. Spectra every 5
(S5: 25); checkpoints every 1000 (epoch 0 carries the tilt readout).

| arm | task | p | d_mlp | fracs | wd | runs |
|-----|------|---|-------|-------|----|------|
| A fine grid | add | 113 | 512 | .2 .25 .3 .4 .6 .8 .9 | 1, 3 | 112 |
| B operation | sub | 113 | 512 | .3 .6 .9 | 1, 3 | 48 |
| C modulus | add | 96 | 512 | .3 .6 .9 | 1, 3 | 48 |
| C modulus | add | 127 | 512 | .6 .9 | 1, 3 | 32 |
| D width | add | 113 | 128, 2048 | .3 .9 | 3 | 32 |
| E non-abelian | s5 | 120 | 512 | .6 .7 .8 .9 | 1, 2 | 32 |

304 runs. One lockstep batch per (task, p, d_mlp, frac). S5 uses 4 seed
pairs (2 masks x 2 inits) and runs last.

Driver: degrok2/train_degrok2.py (idempotent). Analyzer:
degrok2/analyze_degrok2.py (local; numpy+safetensors).

## Metrics (identical to degrok v1)

t_ident (logit-spectrum AUC >= 0.9 vs own final committee, sustained 2
snapshots), t_gen (test acc >= 0.99 sustained 2), tilt_auc0 (epoch-0 W_E
frequency-energy AUC vs own final committee; not computed for S5 — the
rank-normalization instrument for irreps stays in tasks/ as exploratory).
Cell = (task, p, d_mlp, frac, wd); medians per cell.

## Pre-registered predictions (committed before training)

Spread = max/min of cell medians over fracs, within a family at fixed wd,
using only cells where >= half the runs generalize (cells that never
reach test 0.99 in budget are reported as the DATA FLOOR and excluded
pairwise — pre-declared, same rule as v1).

- **P-H1 (fine grid):** add p113 d512 wd1 — t_gen spread >= 20x while
  t_ident spread <= 3x. KILL if t_ident spread >= 5x.
- **P-H2 (fresh-seed wd replication):** at every add-p113-d512 frac,
  |median t_ident(wd3) - t_ident(wd1)| <= 25 epochs, while
  t_gen(wd1)/t_gen(wd3) >= 2 at frac 0.3.
- **P-H3 (operation):** sub p113 — t_gen spread >= 5x, t_ident spread
  < 3x, and wd-invariance (|diff| <= 25) at every sub frac.
- **P-H4 (modulus):** p96 passes the P-H3 criteria over its fracs;
  p127 wd-invariance (|diff| <= 25) extends to fracs 0.6 and 0.9.
- **P-H5 (width):** at each of d_mlp 128 and 2048 (wd3):
  t_gen(0.3)/t_gen(0.9) >= 5 while t_ident(0.3)/t_ident(0.9) <= 3.
- **P-H6 (non-abelian, conditional):** scored only if >= 2 S5 fracs have
  >= half their runs generalize at some wd; then within that wd, t_gen
  spread >= 3x while t_ident spread <= 3x. S5 wd effects are REPORTED,
  not scored (known boundary from tasks/).
- **P-H7 (readability pervades):** mean tilt_auc0 >= 0.60 in every
  add/sub cell (all p, all fracs, both wd).
- **Family-scoping rule, pre-declared:** a family where t_gen spread
  >= 5x AND t_ident spread >= 5x fails the dissociation FOR THAT FAMILY;
  the claim becomes family-scoped and the failure is reported first-class.
  The universal phrasing ("property of training") requires P-H1, P-H3,
  P-H4, P-H5 all passing; P-H6 extends it beyond abelian groups.

No re-rolling of seeds, fractions, budgets, thresholds, or detectors
after seeing results.

## On the pod

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse \
    https://github.com/UndercoverMathGuy/grok.git && cd grok \
    && git sparse-checkout set degrok2 dial cloud
pip install --break-system-packages safetensors
(setsid nohup python3 -u degrok2/train_degrok2.py > degrok2.log 2>&1 < /dev/null &)
```

Smoke (CPU, ~2 min): `python3 degrok2/train_degrok2.py --smoke`
