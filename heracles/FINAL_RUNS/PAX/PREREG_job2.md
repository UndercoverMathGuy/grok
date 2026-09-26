# PAX job 2 pre-registration (written 2026-09-19, BEFORE pax_job2.sh was launched)

Recipe under test (`pax_job2.sh`): x², width 500, init 4, lr = 12.8 constant from step 1, grad-norm clip 1,
**λ = 0 until CE < 0.8 ln p** (cap 3000 steps), then λ = 1e-4 for the rest of a 10 000-step budget.
Baseline = archived `FINAL_RUNS/REP` x² init 4 λ 1e-4 with the 0.05→12.8 ramp and the penalty on from step 1.

Derived escape number (see PRIME_AXIS.md §2):  Ξ = τ_∞/τ* = ‖θ₀‖ / (2 λ τ*(p)).
Archived cells: Ξ = 2.17 (p 113 init 4), 1.67 (p 127), 1.12 (p 157), 0.62 (p 211), 0.29 (p 307).
The warm-up sets Ξ = ∞ at every prime. Archived quality γ/γ_con(K*) rises monotonically with Ξ and its
Ξ ≳ 7 asymptote is 0.45–0.50.

Predictions (each is a KILL if it fails):

* **P1 (escape).** p 157 and p 211 escape in under 3000 steps and reach fp64 accuracy ≥ 0.999 with **positive**
  minimum margin. (Archived: 0/16 fit at p 157, 0/8 and never left zero at p 211.)
* **P2 (quality).** γ/γ_con(K*) ≥ 0.40 at p 113, p 157 and p 211. (Archived p 113 init 4: 0.246.)
* **P3 (junk).** clean fraction ≥ 0.55 at p 113. (Archived: 0.461.)
* **P4 (counts).** K_5pct ∈ 5..10 and N_.05 strictly below the archived value at p 113 (18.0).
* **P5 (null it could fail on).** If γ/γ_con lands at ≈ 0.25 again, the escape-headroom story is wrong about
  quality (it would still be right about escape) and the quality decay with p is something else.

---

# Addendum, written after Phase B (lambda 1e-4) and BEFORE any Phase C (lambda 1e-5) cell was pulled

Phase B quality gamma/gamma_con(K*): 0.523 / 0.456 / 0.424 / 0.362 / 0.314 / 0.258 / 0.180 at p = 59 / 71 / 89 /
97 / 113 / 127 / 157, i.e. the decay with p SURVIVES the escape fix.  Measured median min margin at those primes
is 3.417 / 3.097 / 2.703 / 2.392 / 2.116 / 1.674 / 1.194, and ln(p-1) is 4.06 / 4.25 / 4.48 / 4.56 / 4.72 / 4.84
/ 5.05, so margin / ln(p-1) falls 0.84 -> 0.24.

HYPOTHESIS Q (soft margin).  At finite lambda the net maximises a SOFT minimum of the p-1 level-set gaps at
inverse temperature 1 in logit units, while gamma is the HARD minimum; the two agree only when
margin >> ln(p-1).  The attainable margin at fixed lambda is roughly p-independent (~ ln(1/lambda) + O(1)) while
ln(p-1) grows, so the net drifts out of the hard-margin regime as p grows.

PREDICTION C1: at lambda = 1e-5 the quality rises at EVERY prime, and the RELATIVE rise is larger at larger p
(so the quality-vs-p curve flattens).  KILL: if quality at lambda 1e-5 is flat or falls, or if the rise is
uniform across p, Hypothesis Q is wrong and the residual decay is something else (width, step budget, or an
intrinsic property of GD at larger n_f).
PREDICTION C2: the counts (K_5pct, K_2pct) move by <= 1 between lambda 1e-4 and 1e-5 at every prime, as the
archived lambda/10 continuations did.
