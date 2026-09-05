# Ensemble cascade test: is the committee count init-agnostic? (2026-09-04)

Pre-committed decision rule (user, same day): build a reduced GD+wd cleanup
cascade with NO per-run seeding and NO constant fitted to K; if the halting
K is insensitive to the seed ensemble, the derivation leads the paper; if K
tracks the seed, the count is init-borne and the plateau+takeover framing
leads. Model: nudial/cascade.py.

## Model

nf=(p-1)/2 circuits with balanced factors, A_k = r_k^nu; coverage
g_k(D) = 1 - cos(2 pi k D/p); per-example margin m(D) = s + sum_k A_k g_k(D);
CE softmax over the p-1 wrong offsets; Adam in the eps regime = GD at lr/eps
+ decoupled wd:  r <- r(1 - lr wd) + kappa G_k r^{nu-1},  G_k = sum_D p_D g_k(D),
kappa = (lr/eps) C^{2/nu} with C the circuit path's margin-per-norm constant.
kappa MEASURED from the final equilibrium of the base runs (kappa G_k
r_k^{nu-2} = lr wd for every surviving circuit; median 1.2e4, per-circuit
agreement within ~1.5x; the ansatz CE floor 2-5e-8 vs measured 9e-8).

Scaffold (memorisation) two ways:
 (i) one degree-nu object with uniform coverage 1/c_s, c_s measured from the
     memorised-state floor (c_s ~ 2e4, noisy): collapses in a burst within
     1-2k epochs, recruiting everyone above threshold -> K mode 5-6 for
     every seed shape/scale that groks, every c_s over 3 decades, every
     kappa over 2 decades, wd 0.3-3, nu_s 2-6. Wrong: real train margin
     stays pinned at 16 nats (q05-q95 = 15-17) through the whole takeover
     (test margin -19 -> +12 over epochs 2000-5000).
 (ii) PINNED scaffold: s = max(0, m0 - min_D sum A g), i.e. memorisation is
     repaired continuously and hands load over gradually; no c_s; the drive
     is always at floor scale e^{-m0}, which cancels kappa ~ 1/eps -> the
     model is eps-invariant by construction (matches the sweep).

## Result (pinned model, p=113, nu=6, lr 1e-3, wd 1, kappa 1.2e4, m0 16)

Halting K is set by the SEED SCALE (mean seed amplitude in nats; the
physical variable is the initial worst-case coverage relative to m0):

| seed scale | min coverage at t0 | K mode (exp / lognorm2 / pareto) |
|-----------|-------------------:|---------------------------------|
| 0.001 | 0.03-0.05 | 2 / 2 / 1 |
| 0.003 | 0.09-0.14 | 3 / 2 / 3 |
| 0.01  | 0.3-0.5   | 4 / 3 / 3-4 |
| 0.03  | 0.8-1.5   | 5 / 4 / 4 |
| 0.1   | 2.7-4.8   | 5 / 5 / 5 |

and by the seed SHAPE at fixed scale (dominant leader -> fewer members,
flat top -> more). Insensitive to kappa (3e3-5e4), wd (0.3-3), m0 within the
grokking range; nu trend right (nu 4: 7, nu 5: 6, nu 6: 5, nu 7: 5 at scale
0.3); p 97 ~ 113 < 211.
=> The count is INIT-BORNE. Pre-committed outcome: framing pivot.

## What the model does get right (keep)

- Grokking with the measured CE floor (4e-8 vs 9e-8), extinction at small
  seeds (init-scale / grokking-boundary regime), wd- and eps-invariance,
  the degree trend, and "halts above its own static optimum" (its optimum
  is K*=3; dynamics lands 4-6). The dynamics-vs-statics gap is real and
  reproducible; the size of the gap is a seed property.
- Takeover in the model is faster (s -> 0 by ~1-3k epochs) than real
  (3-5k); leaders in the real net grow to 20-40 nats vs ~10-20 in the model.

## The seed, measured at the ensemble level (8 natural runs, ckpts)

At cleanup start (epochs 1000-3000) EVERY frequency carries ~0.13 nats
(median over 56; mean 0.2-0.25) and the min coverage is ~9-10 nats of the
16-nat train margin. That floor is the memoriser's own Fourier content: a
one-hot memoriser of N_tr pairs at margin m projects 2 m N_tr / p^3 = 0.085
nats onto EVERY cos(w(a+b-c)) mode (measured 0.13). It is not a circuit
seed — it does not generalise (test margin -19) — and the ansatz cannot
separate memoriser content from circuit seeds by projection. The circuit
seed is the excess above that floor: 3-8 frequencies at 0.5-8 nats, whose
profile already sorts the outcome (dominant leader 4.2/1.3/1.0 -> K=3;
flat top 0.74/0.70/0.64/0.56 -> K=4; 0.66/0.60/0.47/0.43/0.39 -> K=5).

## Bottom line

The count is not derivable from (p, nu, wd, lr) alone; it is the halting
point of an init-seeded cascade, and the seed profile above the memoriser
floor is the deciding input. The reduced model is a correct mechanism, not
a formula for 4. Paper framing: K* optimum + plateau (proved/measured),
count chosen at takeover from an init-determined ordering (readable at
init, AUC 0.91), eps sweep locating the decision in the eps-invariant GD
phase, cascade model as the mechanism figure. No integer is promised.
