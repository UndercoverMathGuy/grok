# nu-dial cohort — is the Fourier committee size set by homogeneity degree?

Pre-registered 2026-08-22, before any run. Theory leg: notes
`sparse-is-l2-max-margin` (memory) + scratch `kstar.py`; uniqueness audit
`notes/uniqueness_sparse_margin.md`.

## Claim under test

Morwani et al. (2311.07568, Thm 7) prove the max-margin solution of a+b mod p
is Fourier-DENSE **under the L_{2,nu} norm**. Under plain l2 weight decay the
per-frequency budget is an l_{2/nu} quasi-ball, and within the
single-frequency-circuit family the optimal committee size is

    K*(nu) = argmax_K K^{-nu/2} M_K,   M_K ~= K - c,  c ~ 2.1 at p=113 (c in [1.6,2.4] over p=59..211, K<=12; numerical, not a constant)   =>   K* ~= c nu/(nu-2); c grows ~log p, see notes/kstar_pscale.json (K* is NOT p-independent)

    nu = 2 : dense ((p-1)/2)      nu = 3 : 6 (flat plateau 5-11)
    nu = 4 : 4 (sharp)            nu = 5 : 4          nu = 6 : 3

independent of p and of width. The 1-layer transformer's effective degree,
MEASURED by the scaling test nu_eff = d log f / d log t on the 148 grokked
runs_dial checkpoints (notes/nu_eff_dial.json, added before any nudial result
was read), is 6.6 (correct logit) / 6.1 (margin), IQR 6.5-6.7, width-flat —
the attention OV path dominates, not the nu = 4 MLP path. At nu = 6 the theory
predicts K* = 3 with K = 4 a near-tie (6% behind); the transformer's observed
3:74 / 4:80 split is the number to reproduce WITHOUT a transformer, i.e. the
relu-6 and quad-5 cells (P-N2 is scored at nu = 4 as registered; the nu = 6
comparison is reported alongside).

## Design

MLPs on one-hot (a, b), p = 113, frac_train 0.5, full-batch, MLX-semantics
AdamW (lr 1e-3, betas .9/.98, eps 1e-8, warmup 10), 10k epochs, spectra every
100, init N(0, 1/fan_in). Degree is dialled by inserting LINEAR layers only:

| family | nu | layout (pre, post) | net |
|---|---|---|---|
| relu | 2 | (0,0) | U,V -> ReLU -> W |
| relu | 3 | (1,0) | E -> U,V -> ReLU -> W |
| relu | 4 | (1,1) | E -> U,V -> ReLU -> W1 -> W  (transformer analogue) |
| relu | 5 | (2,1) | E1 -> E2 -> U,V -> ReLU -> W1 -> W |
| relu | 6 | (2,2) | E1 -> E2 -> U,V -> ReLU -> W1 -> W2 -> W |
| quad | 3 | (0,0) | U,V -> x^2 -> W  (Morwani / Tian) |
| quad | 4 | (0,1) | U,V -> x^2 -> W1 -> W |
| quad | 5 | (0,2) | U,V -> x^2 -> W1 -> W2 -> W |

E width d = 128, readout width r = 128, hidden m in {128, 512, 2048}.

**Cohort A (nu-dial):** 8 (family, nu) x 3 widths x wd {0.3, 1, 3} x 32 seeds (16 at m = 2048)
= 1920 runs, one 96-wide lockstep batch per (family, nu, m) (24-wide chunks at m = 2048).

**Cohort B (norm swap):** quad nu = 3, m = 512, Morwani's own regulariser
CE + lam * sum_j ||(u_j, v_j, w_j)||^3, lam in {1e-4, 1e-3, 1e-2} x 32 seeds
= 96 runs. Control = cohort A's quad3_m512 cells (plain wd).

**Cohort C (margin brute force, no training data split):** directly maximise
the softmin normalised margin min_i g_i(theta)/||theta||^nu from 32 random
inits (beta ramp 1 -> 200), for (relu 2), (quad 3), (relu 4) at p in {23, 113}
= 192 runs. Tests the optimum itself, independent of CE dynamics.

Seeds: META_SEED 20260822, 32 init seeds from [200000, 300000); data mask
seed = init seed mod 1000. Batches are 100-wide by default (`--batch`).

## Instruments (all on FINAL-epoch logits over the full p^2 grid)

- `K_gap`: `committee_from_coeffs` — largest log-gap in the top-12 Fourier
  amplitudes of cos(k(a+b-c)), 2% floor. The transformer cohorts' instrument.
  Saturates at ~13 by construction: NOT used to certify density.
- `K_eps`: # frequencies with >= 0.5% of Fourier energy (MAIS instrument).
- `k_eff`: participation ratio 1/sum(share^2) — continuous, unbounded.
- Grokked = final test acc >= 0.99. Cells scored on grokked runs only;
  cells with < half grokked are reported as a DATA FLOOR, not scored.

## Pre-registered predictions

- **P-N1 (monotone dial):** across cohort-A cells, Spearman(k_eff median, nu)
  <= -0.8 and Spearman(K_gap median, nu) <= -0.8. KILL if either > -0.5.
- **P-N2 (the transformer number):** at nu = 4 (both families pooled),
  modal K_gap = 4 and >= 60% of grokked runs have K_gap in {3, 4, 5}.
  KILL if modal K_gap >= 7 or <= 2.
- **P-N3 (dense at nu = 2):** >= 80% of grokked relu-2 runs at m >= 512 have
  K_eps >= 0.8 (p-1)/2. KILL if < 40%.
- **P-N4 (width):** k_eff at m = 2048 / k_eff at m = 128 is >= 2 for relu-2
  and <= 1.3 for every nu >= 4 cell. (nu = 3 reported, not scored — the
  plateau is flat and two papers see dense there.)
- **P-N5 (norm swap):** under L_{2,3}, K_eps median >= 0.8 (p-1)/2 for at
  least one lam, while the plain-wd quad3_m512 control has K_gap median
  <= 8. KILL if L_{2,3} is as sparse as plain wd at every lam.
- **P-N6 (optimum, no dynamics):** cohort C medians: relu-2 K_eps >= 0.8
  (p-1)/2 (dense), relu-4 K_gap in {3, 4, 5}, and relu-4 normalised margin
  strictly exceeds that of a dense construction evaluated in the same net
  (post-hoc construction, reported).

Pass = P-N1, P-N2, P-N3, P-N5 all pass. P-N4 / P-N6 are secondary.

## Running

    python3 nudial/train_nudial.py --smoke                 # CPU, ~1 min
    python3 nudial/train_nudial.py --cohort A              # pod
    python3 nudial/train_nudial.py --cohort B
    python3 nudial/train_nudial.py --cohort C
    python3 nudial/analyze_nudial.py                       # -> notes/nudial_summary.json

Idempotent: runs with spectra.npz are skipped. RUNS_DIR overrides the
output root (default runs_nudial/).

## Round 2 (pre-registered 2026-08-22 after round-1 scoring, notes/nudial_results.md)

**Cohort D (optimum, fixed optimiser):** margin mode now keeps theta on the
unit sphere (renormalised every step) and softmins a scale-free copy of the
margins, so beta means the same thing at every nu. relu 2/4/5/6 + quad 3/4,
m=512, p=113, 32 inits. Reports the true l2 optimum's K per family/nu.
- **P-N7:** D's relu-4/5/6 optima have positive margin and K_eps within +-2
  of the trained A-cells' K_eps (trained relu nets sit AT their optimum);
  D's quad-3 optimum is SPARSE (K_eps <= 12) while trained quad-3 is dense
  (dynamics trap). KILL of the "quad offset = dynamics" reading if the quad-3
  optimum is itself dense (K_eps >= 45).

**Cohort E (where the degree sits, fixed nu):** quad (1,0) [shared embedding,
nu=5] vs quad (0,2) [readout stack, nu=5]; relu (2,0) vs (0,2) vs (1,1)
[all nu=4]. m=512, wd {0.3,1,3} x 32 seeds = 96/cell.
- **P-N9:** k_eff differs by >= 2x between placements at fixed nu (sharing /
  placement is a real variable the nu-only formula lacks); direction
  prediction: a shared pre-activation layer (E) gives the SMALLER K.
  KILL of the sharing hypothesis if all placements agree within 1.3x.

**Cohort F (init scale):** relu-4 and quad-3, m=512, wd=1, init_scale
{0.1, 0.3, 1, 3} x 24 seeds.
- **P-N8:** quad-3 k_eff falls with smaller init scale (the dense symmetric
  point is reached only from large symmetric seeds); relu-4 k_eff is flat
  within 1.5x across scales. KILL if quad-3 is dense (k_eff >= 35) at every
  scale that groks.

Run: `python3 nudial/train_nudial.py --cohort all2`.

## Round 3 (pre-registered 2026-08-22 after round-2 scoring): the +1

Evidence so far: the extra trained member is a real generalising circuit
(lesion cuts train and test margin equally), present in the pre-grok top-K in
88-90% of runs (random non-member 1%), final K tracks memorisation breadth,
transformers at frac 0.9 sit at the optimum K=3, AdamW box inactive.
Hypothesis: **the +1 is a memorisation-era circuit that survives cleanup
because it is a valid circuit** (nothing for wd to decay, no gradient to
concentrate at the CE floor).

**Cohort G:** relu-4 and relu-6, m=512, frac {0.3, 0.5, 0.7, 0.9, 1.0} x wd
{0.3, 1, 3} x 32 seeds.
- **P-N10:** mean K_gap decreases monotonically with frac and equals the
  D optimum (4 for relu-4, 3 for relu-6) at frac 1.0 (mode). KILL if K at
  frac 1.0 is still optimum+1 (mode 5 / 4).
**Cohort H:** D-style optimum search on the frac-0.5 SUBSET (relu-4, relu-6,
32 inits).
- **P-N11:** the subset optimum has the same mode as the full-data optimum
  (4 / 3) — i.e. the +1 is not a property of the subset's margin landscape.
  KILL if the subset optimum mode is 5 / 4.
