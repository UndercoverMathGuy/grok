# Probe ceiling: how much of the final committee is readable from epoch 0?

Question: the closed-form ticket `T_k = sum_h ||W_O^h W_V^h W_E|_k||^2` reaches
AUC 0.758 on the natural v2 runs. Is that near the information ceiling of the
init weights, or far below it?

Scripts: `scripts/pivot/probe_features.py`, `probe_models.py`,
`probe_ceiling.py`, `probe_prior_control.py`, `probe_noise_ceiling.py`.
Numbers: `notes/pivot/probe_ceiling.json` (+ `probe_noise_ceiling.json`,
`probe_prior_control.json`). Analysis-only; nothing was trained.

---

## 0. Two corrections to the premise

**The natural cohort is 8 runs, not 96.** `discover()` returns 96 kept v2 runs,
but only the `p-113` family (8 runs) is `natural-normal`. The rest are
54 `orth-flat` (orthogonalized W_E: `orthWE`, `dyn-lr3/lrlo/wd04/wd25`,
`eff-G`, `phase2-tilt`), 26 `surgical`, 8 `double-flat`. Every checkpoint was
materialized (no LFS pointers); nothing was skipped.

**All 96 runs descend from 8 independent init draws.** Distinct
`(data_seed, init_seed)` pairs: 8, with 8-21 runs each. `orth-flat` variants
and `surgical` arms are deterministic edits of the *same* epoch-0 tensors, so
grouping CV by run — or even by `common.cluster_key`, which splits them —
leaks the init across folds. **Primary CV here is leave-one-init-out, 8 folds.**
That is the real evidential n, and it caps everything below.

## 1. Baseline reproduces exactly

Per-run AUC of raw `T_k` against the unified detector's final committees:

| cohort | n | T_k AUC | Jaccard@K | exact-set |
|---|---|---|---|---|
| natural-normal | 8 | **0.7575** (README: 0.758) | 0.137 | 0/8 |
| surgical | 26 | 0.7372 | 0.225 | 0/26 |
| orth-flat | 54 | 0.6223 | 0.094 | 0/54 |
| double-flat | 8 | 0.5568 | 0.061 | 0/8 |
| **pooled** | 96 | **0.6592** | 0.130 | 0/96 |

## 2. The floor is not 0.5 — it is ~0.66

Score every frequency by *how often it is a committee member in the other
runs* (leave-one-init-out). This uses **no weights at all**:

| model | pooled | natural | surgical | orth-flat |
|---|---|---|---|---|
| popularity prior (no init) | **0.6604** | 0.6929 | 0.6024 | 0.6790 |
| T_k | 0.6558 | 0.7616 | 0.7332 | 0.6167 |

Committee membership is far from uniform over k (k=52 is a member in 32% of
runs; 14 of 56 frequencies are never members; max/mean = 4.5). Permuting
committees across runs and refitting the full probe lands at 0.558 for the
same reason.

So **pooled T_k adds nothing over frequency popularity** (-0.004, p=0.9). On
the natural cohort T_k does read the init: +0.069 over the prior, though at
n=8 that is p=0.148.

## 3. Learned probes, leave-one-init-out

123 features per (run, frequency) from the init checkpoint only, covering
every path the closed form drops — W_U energy, QK path (incl. the `=`-position
attention logit), MLP path (`W_in OV W_E`, the skip read, the full linearized
loop `W_U^T W_out W_in OV W_E` projected back to output frequency k, a
per-neuron read/write matched filter), principal angles between `W_E|_k` and
`W_U|_k`, projections onto the OV circuit's top transmitted subspaces — each
in three transforms (abs-log, within-run z, within-run rank).

| model | pooled | natural | surgical | orth-flat | Jac | exact |
|---|---|---|---|---|---|---|
| T_k (closed form) | 0.6592 | 0.7575 | 0.7372 | 0.6223 | 0.130 | 0 |
| popularity prior | 0.6604 | 0.6929 | 0.6024 | 0.6790 | 0.081 | 0 |
| logreg, 123 feats | 0.6933 | 0.7593 | 0.7928 | 0.6456 | 0.161 | 0 |
| GBT, 123 feats | 0.6701 | 0.7205 | 0.7508 | 0.6266 | 0.119 | 0 |
| MLP (2x hidden) | 0.6754 | 0.7309 | 0.7854 | 0.6133 | 0.110 | 0 |
| prior + T_k | 0.7143 | 0.7661 | 0.6960 | 0.7200 | 0.118 | 0 |
| **prior + 123 feats** | **0.7458** | **0.7909** | 0.7879 | 0.7279 | 0.124 | 0 |

Paired over runs (Wilcoxon):

- `prior+ALL` vs `T_k`: **+0.087 pooled** (p=1.6e-11); **+0.033 natural**, n=8, ns.
- `prior+ALL` vs `prior+T_k`: **+0.031 pooled** (p=0.003); +0.025 natural, ns.
- `logreg(123)` vs `T_k`: +0.034 pooled (p=0.019, 51W/41L); +0.002 natural, ns.

Nonlinear models *lose* to logistic regression — with 6 training init groups
they overfit. **Exact-set recovery is 0/96 for every model, including T_k.**
Jaccard improves only 0.130 -> 0.161.

### Which group carried the gain

Nested (pooled / natural, logreg):

| features | pooled | natural |
|---|---|---|
| E (embedding only) | 0.6419 | 0.7299 |
| E+V (add OV / T_k family) | 0.6928 | 0.7304 |
| E+V+U (add unembedding) | **0.7037** | **0.7588** |
| E+V+U+M (add MLP path) | 0.7046 | 0.7479 |
| E+V+U+M+Q (add QK) | **0.7082** | 0.7404 |
| +P (phases), = ALL | 0.6933 | 0.7593 |

Leave-one-group-out from ALL: dropping V costs -0.018 (largest); dropping P
*gains* +0.017 (the phase-angle block is noise at this n). The single best
univariate feature is **`T_k x ||W_U|_k||^2` at 0.689**, above T_k's 0.659 —
i.e. the one real closed-form upgrade found here is multiplying arrival
loudness by the unembedding's per-frequency energy. Next: `V_top_sv` 0.675
(top singular value of the stacked OV-transmitted freq-k block, rather than
its trace) and `M_match` 0.671 (MLP read/write matched filter).

Transform ablation is flat (abs-log 0.688, z 0.680, rank 0.679, z+rank 0.691),
so nothing hinges on the relative-vs-absolute encoding.

## 4. The actual ceiling

Two runs sharing an init but differing in dynamics do **not** land on the same
committee. That disagreement bounds any init-only readout. Averaging a run's
siblings' committee indicators estimates `P(k in committee | init)` directly —
the Bayes-optimal init-only score — and the sibling runs even cheat by having
been trained:

| oracle | n | AUC | Jaccard | exact-set |
|---|---|---|---|---|
| **dynamics-only siblings** (orthWE / dyn-lr* / dyn-wd* / eff-G; 4.8 siblings each) | 46 | **0.8838 ± 0.161** | 0.566 | **0.283** |
| same-cohort siblings, orth-flat | 54 | 0.8798 | 0.492 | 0.222 |
| same-cohort siblings, surgical | 26 | 0.9312 | 0.461 | 0.000 |
| natural via normal-W_E siblings (leaky, see below) | 3 | 0.9896 | 0.644 | 0.333 |
| different-init pairs (chance) | 3921 | 0.5177 | 0.061 | — |
| different init, **same train mask** | 1813 | 0.5418 | 0.089 | — |

The `dynamics-only` row is the trustworthy one: those families differ from
their init-mates *only* by optimizer hyperparameters, so no committee
knowledge went into their design. Surgical arms and `phase2-tilt` were built
knowing the base run's committee, so oracles containing them (including the
0.9896 natural row, n=3) leak the label and are upper-biased.

Same-mask/different-init pairs sit at 0.542 vs 0.497 for different-mask, so
the shared training mask contributes only ~+0.045 — the oracle's power is init
geometry, not the data split.

## 5. Is the gap a sample-size problem or a feature problem?

- **Leave-one-RUN-out** (leaks the init across folds, deliberately optimistic):
  logreg 0.7338 pooled / 0.7784 natural, only +0.04 over leave-one-init-out.
  So the 8-group limit costs ~0.04, not the ~0.19 to the ceiling.
- **In-sample fit** (train == test, no CV at all): logreg on all 123 features
  reaches only **0.8202 pooled / 0.7696 natural**, with Jaccard 0.328 and
  exact-set 3%. A linear readout of this feature bank *cannot express* the
  committees even when handed the answers. (GBT in-sample hits 0.974 — it
  memorizes; uninformative.)

The features, not the folds, are the binding constraint.

## 6. Verdict

**0.76 is not the information ceiling, but the headroom is not where I
looked.** Three separate statements:

1. **Pooled across cohorts, T_k is weaker than it looks.** Its 0.659 is
   indistinguishable from a no-weights frequency-popularity prior. Any future
   AUC claim on mixed cohorts must be reported as an increment over that
   prior, not over 0.5.
2. **On the natural cohort, T_k is close to what my probes can do.** 123
   features, three model classes, and a popularity prior buy +0.033 AUC over
   0.7575 and it is not significant at n=8. The one genuine closed-form
   improvement is `T_k x ||W_U|_k||^2` (univariate 0.689 vs 0.659 pooled).
3. **A leak-free oracle reaches 0.884 with 28% exact-set recovery**, versus
   0/96 exact-set for every probe and for T_k. So substantially more of the
   committee *is* determined by the init than any of this reads.

The most likely reason the gap survived a 123-feature bank: **every model here
scores frequencies independently, and the committee is a set-level object.**
The veto grammar (0/70 distinct-trio finals; eviction of the weakest member)
means "k is in because j is out" — a constraint no per-k probe can represent.
That is exactly the signature observed: probes gain a little AUC (a ranking
statistic) and zero exact-set matches, while the sibling oracle, which
implicitly sees whole sets, gets 28% of the sets exactly right.

### What I would do next

1. **Set-level readout.** Score candidate *sets* (feasible under the additive
   veto) rather than individual frequencies — e.g. beam search over sets
   scored by `sum_k T_k*||W_U|_k||^2` minus a veto penalty. Cheap, and it
   attacks the failure mode the numbers point at.
2. **Buy the natural ceiling.** There are zero natural repeats of a single
   init in v2, so the natural-cohort ceiling is currently unmeasurable
   (the 0.9896 estimate is n=3 and label-leaky). 6 optimizer-hyperparameter
   variants on each of the 8 natural inits would settle it, and is the same
   recipe already used for `dyn-*`.
3. **Drop the phase block**, keep `T_k x U-energy` and `V_top_sv` as the
   closed-form upgrade candidates.

### Honest failures

- The stated premise "~96 natural runs" was wrong; the natural cohort is 8.
- n=8 independent inits: every natural-cohort comparison is underpowered, and
  GBT/MLP overfit rather than help.
- The phase/geometry feature block (principal angles, OV-subspace
  projections) is a net negative and did not replicate any signal.
- No leak-free ceiling could be measured on natural runs at all.
