# The count under a constant l_q geometry

Everything in `heracles` is scoped to optimizers whose implicit geometry is one fixed norm for the
whole run, with the explicit penalty in the SAME norm. This note states the prediction for a general
exponent q and lists which optimizers qualify.

## 0. Symbols

- p: prime modulus; task a + b mod p. nf = (p-1)/2 frequencies (k and -k identified).
- theta: all weights. f(a, b; theta): the p logits.
- nu: homogeneity degree. Scaling every weight by t > 0 scales the logits by t^nu. For the heracles
  ReLU MLP, nu = number of weight matrices on the input-to-logit path (2 + pre + post linear layers).
- g(theta) = min over (a, b) of [correct logit - largest wrong logit]: the min margin over the grid.
- ||theta||_q = (sum_i |theta_i|^q)^(1/q). q = 2 is weight decay's norm; q = inf is max |theta_i|.
- gamma_q(theta) = g(theta) / ||theta||_q^nu: the q-normalised margin (scale-free by homogeneity).
- A committee S: a set of K frequencies; A_k >= 0 the amplitude of frequency k's clean circuit
  (logit contribution A_k cos(2 pi k (a + b - c) / p)).
- M_K(p) = max over |S| = K of min over offset D != 0 of sum_{k in S} (1 - cos 2 pi k D / p):
  the best worst-case coverage a K-committee can give at unit amplitude.

## 1. The chain of results

(i) Wei et al. 2019, Thm 4.1 (any norm, any r > 0): the GLOBAL minimiser of
    CE + lambda ||theta||^r has normalised margin -> max_theta gamma(theta) as lambda -> 0, where the
    normalisation is by the norm in the penalty. Their Thm C.2: an alpha-approximate minimiser has
    margin >= gamma* / (10 alpha^{nu/r}) — a constant-factor (plateau) statement at finite lambda.
(ii) Steepest descent w.r.t. a norm ||.||: update direction = argmin_{||d|| <= 1} <grad, d>. For q = 2
    that is the gradient; for q = inf it is sign(grad); for general q it is sign(g)|g|^{1/(q-1)}.
    Implicit-bias results (Gunasekar et al. 2018 linear; Lyu-Li 2019 / Ji-Telgarsky 2020 gradient flow
    on homogeneous nets; Wang et al. 2021 Adam/sign -> l_inf) say the margin that gets maximised is the
    one normalised by the geometry's norm. When the explicit penalty uses the same norm there is no
    conflict: the trained direction approaches the l_q max-margin direction.
(iii) AdamW at eps = 1e-8 is sign geometry early (memorisation) and l2 geometry late (the eps crossover,
    notes/eps_dial_results.md); Lion is sign geometry with an l2 shrink. Both MIX norms. They are
    excluded from heracles claims and kept only as labelled controls.

## 2. The l_q cost of a frequency circuit (decoupled pricing)

A frequency-k circuit in a degree-nu net is a chain of nu weight blocks with q-norms r_1..r_nu; its
amplitude is the product, its q-penalty the sum of q-th powers:
    A_k = c * prod_i r_i,        cost_k = sum_i r_i^q.
AM-GM at fixed product: the cost is minimised by balanced blocks r_i = (A_k/c)^{1/nu}, giving
    cost_k = nu (A_k / c)^{q/nu}.                                                     (2.1)
If the K circuits are private (decoupled pricing: costs add across frequencies; exact at q = 2 by
Parseval, an ansatz for q != 2), the normalised budget is
    sum_{k in S} A_k^{q/nu} <= const,                                                 (2.2)
an l_{q/nu} quasi-ball on the amplitude vector. It is a non-convex star (concentration pays) iff
q/nu < 1, a simplex (linear, no preference) at q = nu, and a convex ball (spreading pays) for q > nu.

Equal amplitudes on K members: A = (const/K)^{nu/q}. Hence the size-constrained optimal margin is
    gamma_hat_q(K) = const * M_K(p) * K^{-nu/q},                                      (2.3)
    K*(p, nu, q) = argmax_K  M_K(p) / K^{nu/q}.                                       (2.4)
Reading: the coverage M_K grows with K (roughly like K - c(p), c ~ 2 at p ~ 100), the splitting price
K^{-nu/q} falls with K; the count is where the product peaks. The single exponent
    s = nu / q
carries the whole optimizer dependence. q = 2 gives the weight-decay ladder K^{-nu/2} of the nudial
line. q -> inf gives s = 0, the argmax is the largest K: uniform dense (the "pure sign pricing" LP of
nudial_results 8.6, width-priced). q = nu is the Morwani L_{2,nu} boundary (linear budget, dense).

Within-frequency splitting: n_k neurons at amplitude A_k/n_k cost nu (A_k/n_k)^{q/nu} n_k, i.e. a
factor n_k^{1 - q/nu} relative to one neuron. For q < nu fewer neurons per frequency are cheaper
(the l2 theorem's n_k^{1/3} junk-cancellation price is the q = 2, nu = 3 case); at q = nu splitting is
free (Morwani's phase-complete construction at no cost); for q > nu spreading over width is cheaper,
so the count becomes width-limited, not norm-limited.

## 3. Predictions (heracles/kstar.json, equal share, K <= 12, "dense" = all nf frequencies beat every
sparse K)

p = 113:
| nu \ q | 2 | 3 | 4 | 6 | inf |
|---|---|---|---|---|---|
| 3 | 6 | dense | dense | dense | dense |
| 4 | 4 | 8 (flat 4..12) | dense | dense | dense |
| 5 | 4 | 5 | 8 | dense | dense |
| 6 | 3 | 4 | 6 | dense | dense |

The steep, cheap geometry-dial cell is nu = 6: K* = 3 -> 4 -> 6 -> dense -> dense as q = 2, 3, 4, 6, inf.
At nu = 4 the q = 3 ladder is flat (r >= 0.74 over 4..12), so q = 3 at nu = 4 predicts a plateau,
not a number — report it as such.

Sparse-vs-dense margin ratio (how strongly the geometry prefers the sparse optimum), p = 113, q = 2:
2.0x (nu 3), 6.5x (nu 4), 24x (nu 5), 97x (nu 6). The theory's precision scales with this ratio and
with the plateau width; both are reported as the theory's own error bar.

## 4. Admissible optimizers (constant geometry, penalty in the same norm)

q = 2 (weight decay):
- full-batch gradient descent + l2 (the record optimizer of Morwani 2023 and Gromov 2023)
- minibatch SGD + l2 (same expected direction; noise is not a geometry change)
- heavy-ball / Nesterov momentum + l2 (direction is a running average of l2 gradients)
- any of the above with global-norm gradient clipping, warmup, or an lr schedule (step size only)
- constant per-tensor learning rates (a fixed diagonal preconditioner = a weighted l2 norm; for a
  homogeneous chain this only rescales the AM-GM balance and leaves K* unchanged)
- Adam with eps >> |g| from step 0 (numerically momentum GD; admissible but redundant)

q = inf (sign):
- signSGD, signum / sign-of-momentum, with the l_inf penalty realised as a max-norm projection
  (clamp every weight to [-c, c]). Prediction: dense, width-priced.
- Adam with eps -> 0 approximates sign descent (Wang et al. 2021) but its preconditioner is
  time-varying: "approximately" only.

general q (research optimizers, one family):
- l_q steepest descent: theta -= lr * sign(v)|v|^{1/(q-1)}, v = grad CE + wd * sign(theta)|theta|^{q-1}
  (penalty wd/q ||theta||_q^q). q = 2 is gd; q -> inf is sign. Implemented as opt = "lq".

Excluded (mixed or time-varying geometry): AdamW at eps 1e-8 (sign -> l2 switch), Lion with l2
shrink (sign steps, l2 penalty; kept as the MIXED control), RMSprop / Adagrad / Adafactor
(time-varying diagonal preconditioner). Muon / Shampoo have a CONSTANT geometry but a spectral one
(per-matrix operator norm); under it K frequencies at equal amplitude in one block cost the same as
one, so the naive prediction is dense — a separate axis, not run here.

## 5. Limits to state in the paper

- Exact homogeneity: no biases, no normalisation layers, no attention. The heracles MLP satisfies it.
- Global-minimiser premise: Wei et al. is about the minimiser; trained nets sit below the ceiling.
  Finite-lambda corrections are of order 1 / log(1/lambda_eff): about 6% at train CE 1e-7 and about
  30% at train CE 0.05, so plain GD needs the lambda -> 0 sweep (PLAN.md, H3) to approach the limit.
- Decoupled pricing: exact at q = 2 (Parseval per frequency); an ansatz at q != 2 and for coupled
  architectures (quadratic activations, non-abelian irreps).
- The formula predicts a ladder; the mode inside a flat plateau is not a landscape quantity
  (impossibility result, notes/seeded_ensemble_theory.md sec 3). Plateau width is reported.
- M_K for K >= 7 at p >= 97 comes from local search (lower bounds).
