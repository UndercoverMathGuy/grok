# The margin-plateau law (2026-08-30 rework: Q dropped)

Supersedes the 08-29 seeded-ensemble parametric fit. The fitted seed
prior Q(K) ~ lam^K/(K!)^s is RETIRED as cheating: a 2-dof count family
fitted to histograms whose content is a center and a width is barely
falsifiable (and the MLE ran to a degenerate corner lam=4.3e6, s=9.9).
What replaces it has ZERO fitted parameters.

## 1. The general-purpose margin curve gamma_hat(K)

No representation assumptions. Ingredients:
- A task with symmetry group G (here Z_p acting diagonally on (a,b,c));
  logit functions decompose into G-isotypic components ("frequencies").
  supp(f) = set of components carrying nonzero energy. K = |supp(f)| is
  the general notion of committee size: number of irreducible characters
  used.
- An architecture with homogeneity degree nu (measured: scale theta by
  t, logits scale by t^nu) and REPRESENTATION COST
      C_arch(f) = inf { ||theta||_2^nu : f_theta = f },
  the cheapest parameter norm that realises f. ALL architecture
  specificity lives here and only here.
- m(f) = min-margin of f over the task grid (correct logit minus best
  wrong logit, worst case over inputs).

DEFINITION (size-constrained optimal normalized margin):
      gamma_hat(K) = sup { m(f) / C_arch(f) : |supp(f)| <= K }.
Well-defined on rays: f -> t f multiplies m(f) and C_arch(f) by the
same factor t (homogeneity), so the ratio is scale-invariant.
gamma_hat(K) is "the best normalized margin any K-committee network can
have". Trained-net link (measured, nu-dial): AdamW+wd endpoints sit at
the top of the normalized-margin scale, so trained gamma tracks
gamma_hat(K_final).

COROLLARY (decoupled pricing). If C_arch decouples across isotypic
components — private subcircuit per character, costs add, and per-
character amplitude A costs nu*A^(2/nu) (AM-GM through a nu-deep
multiplicative chain) — then the sup is attained at equal amplitudes on
the best coverage set and
      gamma_hat(K) = const * M_K(p) / K^(nu/2),
      M_K(p) = max_{|S|=K} min_{Delta != 0} sum_{k in S} (1 - cos(2 pi k Delta / p)).
This is a THEOREM about the cost functional, not an assumption. Holds
for the ReLU transformer / ReLU MLP (validated to 0.1-6% by
fixed-support training, round 4; nu = 6 measured).

NON-EXAMPLE: the quad transformer (nu = 10 measured). Squaring couples
components (a K-supported logit drags cos(k(a-b)) junk and 2k
harmonics), so C_quad does not decouple; the corollary fails and
gamma_hat must be obtained from the constrained sup directly —
operationally, MEASURED by fixed-support training, the same instrument
that validated the ReLU closed form. Registered: gamma_hat_quad peaks
at K=3 (converged quad cohorts sit at mode 3), a statement about
C_quad, not a framework failure. This is the per-case recipe: closed
form when pricing decouples, measured C_arch when it doesn't.

## 2. Law 1 (landscape, ZERO fitted parameters): plateau concentration

Claim: final committees land on the near-optimal plateau of gamma_hat.
Define r(K) = gamma_hat(K) / max_J gamma_hat(J) for that prime.

Scorecard, 157 vanilla runs, p in {97,113,127,211} (12 possible K bins
per prime, plateau is 2-3 of them):
      100%  of runs have r(K_final) >= 0.56
      97.5% of runs have r(K_final) >= 0.74
      83%   of runs have r(K_final) >= 0.84
Also invariant across wd 0.3/1/3 and frac 0.3/0.4/0.5 (~120 further
runs at p=113), as the formula requires (wd and frac do not enter
gamma_hat).

This claim needs NO distributional family, no beta, no prior. It is
the falsifiable core: a committee off the plateau (e.g. K=8 at p=113,
r=0.36) refutes it.

## 3. The impossibility (why the mode is not a landscape quantity)

At p=113 gamma_hat peaks at K=3 with gamma_hat(4)/gamma_hat(3) = 0.94,
yet the histogram is {3:15, 4:60}. Any P(K) = w(gamma_hat(K))/Z with w
increasing forces P(3) >= P(4). Excluded at p=97 (ratio 0.845, 5 vs 18)
and p=127 (0.918, 1 vs 15) as well. PROOF-GRADE: no gamma_hat-only
distribution — any monotone w — can produce the observed mode. Entropy
rescues (C(nf,K), N_eff) inject p-scaling and were killed by P-N14.

## 4. Law 2 (initialization, MEASURED, stated in prose not fitted)

The mode is set by the init lottery: the count of frequencies within a
fixed ratio of the leader at grok time ("seed") has median ~4 at every
prime tested (nf 48 -> 105) — plausibly an extreme-value effect (for
exponential-class tails the near-max count of many contestants is
asymptotically independent of how many compete), though that mechanism
is motivation, not proof. Final K = seed projected onto the plateau of
Law 1: mode 4 wherever 4 is on the plateau (all four primes);
recruitment upward when the plateau floor exceeds the seed (p211:
seed==final only 6%). The seed count is an independent measurement with
its own instrument (amplitude threshold on grok-time spectra), reported
like nu — NOT a prior fitted to the K-histograms. No parametric Q
appears anywhere in the framework.

## 4b. COMPETING ACCOUNT (2026-08-30, round 8): selection-time exponent

Statics are now exhausted three ways (hard margin, exact CE+wd
equilibrium at every lam — nudial/ce_equilibrium.py — and paired
fixed-support training all say K*), and the +1 is optimizer-borne
(plain GD+l2 cohorts sit at/below K*; AdamW sits at K*+1 in BOTH
architectures). Alternative to Law 2: the mode is the argmax of
M_K(p)/K^s with a SELECTION-TIME exponent s = nu_sel/2, nu_sel in
(4.66, 4.83) — one constant pinned to a 3% window by the four primes
jointly (endpoint nu=6 is jointly excluded). Mechanism candidate:
committee locks mid-training while part of the chain (E, Q/K in
eps-regime; attention locks later) is not yet amplifying; support then
freezes. Discriminators vs the seed account: p1009 mode 5 (vs seed 4);
eps/GD intervention moves the mode to 3 (vs seed: no effect); nu_eff
measured AT LOCK TIME lands in the window. See nudial_results round 8.

## 5. Registered discriminators (unchanged)

- p=1009, nu=6, vanilla recipe: plateau = {4,5} (r(4)=0.992,
  K*(1009)=5), seed ~4 => mode 4-5, essentially no mass at K >= 6.
  Dead entropy models say mode >= 11 (set-level P(12)=0.84). Maximal
  discrimination, zero parameters.
- Quad: fixed-support gamma_hat_quad peaks at 3, not 2.
- f0.2 grokking boundary: starved memorisation weakens the seed; the
  +1 fades (observed 6/24 groks, mode K* = 3 — consistent).
