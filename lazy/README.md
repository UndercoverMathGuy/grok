# Lazy — the linearized twin (DRAFT prereg; theory leg)

STATUS: DRAFT 2026-08-18 — declared before training; nothing trained yet.
(Distinct from twin/ — that is the fork/swap/dial/hold edit suite. This
cohort trains the TANGENT MODEL.)

## The question

Everything in the program says selection is an init-time amplification
race (epoch-0 readability, compile-at-init works / edit-mid-race fails,
the bilinear-race mechanism), while generalization requires building
features a random init does not contain. If that picture is right, the
two processes live at different orders of the Taylor expansion around
the init:

> **Selection is decided in the lazy (kernel) regime; learning is what
> the lazy regime cannot do.**

Test it literally: train the LINEARIZATION of the network at its init —
f_lin(theta, x) = f(theta0, x) + J_{theta0}(x) (theta - theta0) — with
the same data, mask, optimizer arithmetic (MLX-semantics AdamW, same
warmup, wd applied to theta exactly as in the real trainer), and the
same spectral instrument. The tangent model can only amplify structure
the init already has; it cannot build new features.

Predictions: the twin runs the same audition and crowns (approximately)
the same committee as its real counterpart — but never generalizes.

## Design

The 8 natural p=113 bases (2 data masks x 4 init seeds, mainline
recipe). Three arms, all from the SAME epoch-0 checkpoints:

- **real** (8): ordinary training, in-cohort (same device/batching), 20k
  epochs — the reference committees and t_gen.
- **lin-wd1** (8): linearized twin, weight decay 1.0 (matched to real),
  10k epochs.
- **lin-wd0** (8): linearized twin, wd 0 (exploratory: pure kernel
  regression, no ridge pull).

Spectra every 5 epochs, identical Fourier instrument. Gradients of the
linear model are exact: dL/dtheta = J^T dL/df via one forward-mode jvp
(logits) + one vjp (gradient) per step — no approximation anywhere; the
smoke asserts that at theta = theta0 the twin's gradient equals the real
model's gradient to float tolerance (they are mathematically identical
at that point).

## Pre-registered predictions

Static baseline for P-J1: top-K epoch-0 embedding-energy set (K = real
run's final committee size) — the no-dynamics readout, J_static.

- **P-J1 (same audition):** J(lin-wd1 final committee, real final
  committee) > J_static in >= 6/8 pairs, and median J >= 0.5.
  KILL: median J <= median J_static — running the lazy race adds
  nothing over the static tilt; selection is decided at order >= 2.
  Pre-declared branch: a kill here is the "order-2 (bilinear/AGF)
  selection" result, not a dead end — it fixes WHICH order suffices.
- **P-J2 (no rich learning):** 0/8 lin-wd1 runs reach test acc >= 0.99
  within 10k epochs, while >= 6/8 reach train acc >= 0.9 (the tangent
  model memorizes but does not grok). If any lin run generalizes, the
  lazy/rich split is FALSIFIED for this task and reported first-class.
- **P-J3 (the race is a race):** the lin twin's own committee stabilizes
  early — spectrum AUC vs its own final committee >= 0.9 sustained
  within the first 500 epochs in >= 6/8 runs (same shape as the real
  t_ident, i.e., the audition dynamics live inside the tangent model).
- Exploratory (never scored): lin-wd0 arm; t_ident(lin) vs
  t_ident(real) timing; per-frequency energy trajectories lin vs real
  over the first 100 epochs (how long the real run stays on its tangent).

No re-rolling of seeds, arms, budgets, thresholds, or detectors after
results.

## Files

- lazy/train_lazy.py — real + linearized arms (batched, torch.func
  jvp/vjp; `--smoke` asserts delta=0 exactness, gradient identity at
  theta0, and early real-vs-lin tracking).
- lazy/analyze_lazy.py — P-J1..P-J3 -> notes/lazy_summary.json (numpy
  only; runs on the pod).

## Order-2 (quadratic) twin — declared 2026-08-18, after the P-J1 kill

P-J1 KILLED as registered: the tangent model does not reproduce the real
committee (median J 0.09 vs static 0.11; wd0 arm 0.18). Selection is not
order-1. The pre-declared branch now runs: the QUADRATIC model

    f_quad(theta) = f(theta0) + J delta + 1/2 delta^T H delta

(the second directional term computed exactly by a nested jvp; gradient
exactly J^T g + H[delta,.]^T g via the cached vjp plus a per-step vjp of
the J-delta map). Same bases, same optimizer arithmetic, same instrument.
This is the bilinear-co-growth regime (AGF's equations): W_E-downstream
product terms are order 2, and every mechanism result (P-T1 revision,
transplant, swap) points here.

Arms: quad-wd1 x8, quad-wd0 x8, 10k epochs, spectra every 5. References:
the already-trained real_wd1 committees (shipped with the cohort).

- **P-Q1 (order 2 suffices for selection):** J(quad-wd1, real) median
  >= 0.5 AND beats the static baseline in >= 6/8 AND median exceeds the
  lin-wd1 median (0.09). KILL: median <= static median (0.11) —
  selection needs order >= 3; the expansion-order framing survives with
  the flag one order higher, reported as such.
- **P-Q2 (order 2 does not generalize):** 0/8 quad arms reach test acc
  >= 0.99. Pre-declared branch: if quad DOES generalize, the split
  becomes order-1 vs order-2 and "learning at order infinity" is
  retracted — reported first-class either way, since quadratic models
  are known to capture some feature learning.
- Exploratory: quad-wd0 (fit + committee), quad test-acc plateau vs the
  lin plateau (~0.27), timing of the quad audition.

Smoke adds two exactness gates before any cohort run: (i) at delta=0 the
quad gradient equals the real gradient; (ii) the order-2 machinery is
validated on a SMOOTH (GeLU) surrogate — truncation residual O(a^3)
(halving delta shrinks it ~8x; measured 8.02) and the nested-jvp Hessian
term matches central finite differences to <1e-9. Amendment, declared
before the cohort ran: the originally promised O(a^3) gate on the ReLU
net itself is mathematically unattainable (piecewise-linear nets have
kinks; measured mean residual ~O(a^1.5), and FD of J(P)d differentiates
across jump discontinuities). The ReLU quadratic twin is therefore the
second-order Taylor model with the almost-everywhere Hessian — a
well-defined model class, computed by machinery proven exact on the
smooth surrogate.

## RESULT (2026-08-18, fresh in-cohort real/lin/quad on one device, L40S)

- **P-J1 KILL as registered** (order-1): lin-wd1 median J 0.07 vs static
  0.11, beats static 4/8. Replicated the 4090 run on a second device.
- **P-Q1 KILL as registered** (order-2): quad-wd1 median J 0.087 vs
  static 0.11, beats static 3/8 — the quadratic model ALSO fails to
  reproduce the real committee.
- **P-J2 GRAY / P-Q2 PASS:** lin-wd1 never fits train (ridge-blocked;
  the wd0 arm fits 8/8 and never generalizes, test ~0.27-0.29,
  exploratory). quad-wd1 fits train PERFECTLY (1.000, 8/8, despite
  wd 1) and never generalizes (max test 0.22).
- **P-J3 PASS:** twin auditions settle early (7/8 within 500 ep).

Verdict: **selection is not a Taylor-truncation-order phenomenon** —
neither the tangent nor the quadratic model crowns the real committee,
so "selection decided at order 2 (bilinear/AGF sufficiency)" is
refuted here alongside order 1. The doubly-confirmed half: **no
truncated model generalizes** even when it memorizes perfectly —
the learning process genuinely requires the full nonlinearity.

Interpretation caveat (recorded, cuts both ways): the truncated twins
never leave the memorization regime, so their final "committees" are
memorization spectra; the real committee is only realized through
incipient feature learning. On this reading the negative result says
selection-as-realized-committee is inseparable from the feature-learning
process itself — consistent with the drift finding that in the grokking
regime selection completes only at grok. The expansion-order program is
CLOSED (negative, reportable); order-3 is not pursued.

## Twin distillation — locating the barrier (DRAFT prereg 2026-08-18; NOT trained)

The kills above show the twins do not FIND the grokking solution when
trained on the task. Open loophole ("didn't try hard enough"): maybe the
solution IS expressible as f_quad for some delta and task-SGD just never
reaches it. Distillation closes it: train each twin to match a grokked
teacher's logits with FULL supervision (all p^2 inputs — the input space
is finite, teacher logits precomputed as a table; protocol from
grok/distill.py, mode=mse on row-centered answer logits, the "=" logit
never a target). This removes the task, the scarcity, and the plateau
from the question entirely: pure can-the-class-express/reach-it.

Positive control already exists: legacy/runs/naive_distill — FULL-net
students distill to teacher at test loss ~2-5e-7 (distill_42_into_37_mse,
distill_72_into_81_mse); ce_from_onehot_t80_e390_lr1e4 reaches 8.6e-6.
Caveat carried honestly: legacy students were full nets, mostly
warm-started from trained checkpoints — hence the fresh from-theta0
control arm below.

Design (same 8 natural bases and epoch-0 checkpoints as above):
- Arms x 8 bases: **full-ctrl** (ordinary network trained from theta0 on
  the distill loss — the from-random-init positive control), **lin**,
  **quad**. Primary arms wd 0 (a representation/reachability test gets
  no ridge pull); wd 1.0 companions exploratory only.
- Teachers, both per base: (T-real) the base's own in-cohort grokked
  real_wd1 final logits; (T-syn) synthetic idealized-Fourier logits
  scale*sum_k cos(w_k(a+b-c)) on that run's final committee (zero
  teacher noise — the algorithm itself as target).
- 10k epochs, full-batch, same MLX-semantics optimizer arithmetic,
  same Fourier instrument, spectra every 5.
- Readouts: MSE-to-teacher curve; student test accuracy on the natural
  held-out split (argmax vs true labels); final committee J vs teacher
  committee.

Pre-registered:
- **Gate G-D:** full-ctrl reaches test acc >= 0.99 in >= 7/8 bases for
  each teacher type. If the gate fails, the protocol (not the twins) is
  indicted and NO verdicts below are licensed.
- **P-D1 (locate the quad barrier):** REPRESENTATIONAL verdict if quad
  reaches test acc >= 0.99 in 0/8 AND its best MSE plateaus >= 100x the
  full-ctrl median — the solution is not in the order-2 class around
  this init (matches the degree->=3 block-product argument).
  DYNAMICAL verdict if quad reaches >= 0.99 in >= 6/8 — expressible and
  reachable under dense supervision, so the task-training failure is a
  search/dynamics barrier. Anything between = GRAY, reported as such.
  Both verdicts are first-class results; neither is a kill.
- **P-D2 (lin, secondary):** same readout for the tangent model. Prior
  recorded now: lin may well interpolate (kernel regression on p^2
  points); a lin-fits/quad-fails split would put the barrier exactly at
  order 2->3 in reverse and is the cleanest possible figure.
- **P-D3 (content transfer, conditional):** scored only for arms that
  fit (acc >= 0.99): committee J vs teacher committee >= 0.8 in >= 6/8
  — dense supervision at low order does transfer content. Never scored
  for non-fitting arms.
- Pre-declared branch: a T-syn vs T-real split verdict (fits one, not
  the other) means the trained network's implementation carries
  structure beyond the textbook circuit — reported first-class.

No re-rolling of seeds, arms, budgets, thresholds, or detectors after
results. Implementation: extend lazy/train_lazy.py with a distill mode
(torch only — the mlx grok/distill.py is protocol reference, not a
dependency).
