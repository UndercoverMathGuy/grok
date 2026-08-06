# Review brief: the 18.4-nat homeostat

Adversarial-review package. Goal: **disprove** the hypothesis below, or find
the confound that kills it. Companion to `runs/eviction/MARGIN_HYPOTHESIS.md`
(the frequency-selection / margin-floor story); this brief is only about the
loss/amplitude/margin *equilibrium*, which is the most quantitatively solid
and the most attackable claim in the project.

Everything here is reproduced by `homeostat_analysis.py` in this directory
(self-contained — all load-bearing math inlined, only the model definition is
imported; numpy + mlx only, no scipy). `REFERENCE_OUTPUT.txt` is the exact
output to reproduce. Run from the repo root:
`uv run python runs/homeostat/homeostat_analysis.py`.

## Setup (exact)

1-layer transformer, d_model 128, 4 heads, d_mlp 512, ReLU, n_ctx 3, no
LayerNorm, no attention biases. Task (a+b) mod p. Full-batch AdamW, lr 1e-3,
**weight_decay 1.0**, betas (0.9, 0.98), warmup 10 epochs, float64 CE,
frac_train 0.3. Grokked to 100% test acc.

**Sample: n = 15 grokked models.** 7 at p=113 (`runs/og_seed0/seed*`, the
exploratory zoo, 20-30k epochs, shared data_seed=0) + 8 from the matrix farm
(`runs/p-{113,127,157}/seed{2034,3604}/seed*`, 30k epochs, fresh masks).
Committee sizes K: 2×K3, 10×K4, 2×K5, 1×K6.

## Definitions (all inlined in the script)

- **Committee** S: frequencies above the largest log-gap in the sorted final
  |phase-locked coeff| (`spectra.npz['coeffs'][-1]`, the cos(w(a+b-c))
  coefficient per frequency k=1..p//2).
- **Amplitude read-off**: translation-average the final full-grid logits over
  the p "miss" diagonals x = a+b-c mod p to get L(x); its cosine amplitude at
  frequency k is `a_k = (2/p) Σ_x L(x) cos(2πkx/p)`. Correct answer (x=0) has
  logit `A_tot = Σ_k a_k` (positive amps); wrong answer at miss x has
  `A_tot − gap(x)`, `gap(x) = Σ_k a_k (1 − cos(2πkx/p))`.
- **minGap** = min_{x≠0} gap(x) (the confidence margin, in nats).
- **relM** = minGap / A_tot (relative min-margin, purely a set+allocation
  property; equal-amplitude version `relM_equal` is Diophantine, mask-free).
- **Symmetric vs actual CE**: L_sym(a,b,c) = L((a+b−c) mod p) is the
  translation-averaged logit field. `phase = CE(actual logits) / CE(L_sym)` on
  the test rows measures row-to-row phase noise the symmetric picture ignores.

## The hypothesis (H1–H6)

**H1 — Homeostasis.** minGap = A_tot·relM is nearly constant across
committees: mean 18.68, sd 0.66, **CV 3.5%**, while its two factors each vary
~27% (A_tot CV 28%, relM CV 26%) and anticorrelate at **r = −0.92**. Weak-margin
committees compensate with large amplitude to hit the same gap.

**H2 — It is an optimizer constant, not a task property (p-INDEPENDENT).**
corr(minGap, p) = −0.06; p=113 → 18.71, p=127 → 18.56, p=157 → 18.62.
Naively minGap should rise with p (more wrong answers to suppress); it is flat.
Interpretation: 18.4 is the equilibrium of CE (wants margin) vs AdamW weight
decay (wants norm), set by lr·wd, not by the group size.

**H3 — K-slope compensates phase noise.** minGap is NOT flat in K:
minGap ≈ 16.0 + 0.66·K (corr +0.72). Reason: row-phase noise grows with K
(corr(log phase_te, K) = +0.62) and lives on test rows (median phase_train
2.1 vs phase_test 4.3); larger committees need extra symmetric margin to hold
actual CE down.

**H4 — Loss law.** actual CE ≈ phase · symmetric CE, with phase ≈ 3–5× for
clean (K≤4) runs (median phase_te 4.3). A minority are phase-pathological
(37–540×: seed66433, seed11285, seed54735, seed51224).

**H5 — Amplitude is cheap; norm prices members.** d log(norm)/d log(A_tot) ≈
−0.05 (norm independent of amplitude). Within p=113, weight norm tracks K:
K3 859, K4 909, K5 980, K6 1024. (Cross-prime norm is confounded by vocab
size — embedding has p+1 rows — so compare norm only within a prime.)

**H6 — Falsifiable prediction (UNTESTED).** Since the constant is set by the
optimizer, minGap ≈ C − log(lr·wd). A weight-decay sweep should shift it by
**−log(2) ≈ −0.69 nats per doubling of weight decay** (more decay → lower
amplitude → lower margin → higher tolerated CE). No wd-sweep data exists yet;
this is the decisive cheap experiment (~4 wd values × a few seeds).

## Attack surface (please try these first)

1. **THE MAIN ATTACK — is H1 just "all models reach the same loss"?**
   minGap ≈ −log(CEsym) + log(multiplicity). The script's triviality control:
   sd(minGap) 0.66 nats vs sd(−log CEsym_te) **0.50 nats**, corr **+0.82**. So
   H1's constancy *largely restates* that all runs converge to a similar
   symmetric loss — expected at fixed wd/lr/task. **H1 alone is weak.** The
   non-trivial residue: (a) the 27% factor compensation (why does the model
   trade relM for A_tot at fixed product rather than, e.g., always maximize
   relM?), (b) H2 p-independence (loss level flat despite class count growing
   with p), (c) H4 (the amplitude read-off actually predicts CE). A reviewer
   should decide whether (a)-(c) survive once (H1) is discounted as loss
   constancy. If they don't, the "homeostat" is a rebranding of "grokked
   models have low loss."

2. **Calibration convention.** The absolute 18.4 depends on the amplitude
   read-off normalization (the 2/p and the translation-average). minGap is
   exponentially sensitive: a 2.3-nat calibration error = 10× in every CE
   ratio (this is why our earlier act/exp(−minGap) ratios were ~250 before we
   compared to the full symmetric CE instead). The *ratios* (H2 p-independence,
   compensation, K-slope) are convention-robust; the absolute number is not.
   Attack: does a different-but-defensible amplitude definition move 18.4?

3. **n and coverage.** n=15; only 3 primes; matrix has 2 masks/prime; 11/15
   are p=113; K5/K6 are 3 runs total. The K-slope (H3) is leveraged by those
   few high-K points, one of which (seed66433) is the worst phase outlier —
   drop it and the K5 mean falls from 19.77 to 18.85. Is the K-slope real or
   an artifact of 1-2 points?

4. **Loss law (H4) is bimodal, not a clean 3-5×.** 4/15 runs are 27–540×.
   Calling it "~3-5×" cherry-picks K≤4. What determines the pathological runs?
   If it's not K (seed11285 is K4 at 37×), the phase story is incomplete.

5. **Endpoint ≠ fixed point.** Amplitudes read at the last checkpoint. The
   monopoly run in `runs/eviction` visibly drifts (norm still shrinking, CE
   rising) at 10k epochs. Are these 15 at a true equilibrium, or still moving?
   Extended-training runs would tell.

6. **frac_train / mask dependence of A_tot.** A_tot is read from full-grid
   logits mixing memorized-train and generalized-test rows. Does the read-off
   contaminate A_tot with train-row structure? Decompose A_tot by train/test.

## What would disprove each claim

- H1: minGap CV comparable to A_tot/relM CV (compensation is illusory), OR the
  triviality control shows minGap is *nothing but* −log(CEsym).
- H2: minGap rises with p on more primes / larger range (it's a task property).
- H3: K-slope vanishes when the 3 high-K runs are resampled; or phase noise
  does not track K on more high-K runs.
- H4: phase factor has no stable central value across a larger sample.
- H6 (the real test): a wd sweep in which minGap does **not** move by ≈log(2)
  per doubling — kills the optimizer-constant interpretation outright.

## Related context (not part of this hypothesis)

The margin *floor* (committees never in the bottom quartile of set-margin;
`runs/eviction/MARGIN_HYPOTHESIS.md`) and the finding that frequency winners
lock at memorization (init-lottery, not a margin reshuffle) are separate
claims. The homeostat connects to the floor via `relM_floor ≈ 18.4 / A_tot_ceiling`
(below the floor, hitting 18.4 nats needs infeasible amplitude), but that link
is n=15-lucky (exact number match) and only the structural argument is durable.

## Prior work to check

Morwani et al. ICLR 2024 (2311.07568) prove max-margin modular-addition MLPs
have an exact margin γ* — but for the *class-averaged* margin over the *full*
frequency spectrum at infinite width, no weight decay, no finite-loss
equilibrium. The homeostat is a finite-wd *equilibrium* claim about the
worst-case (min) margin of a *sparse* committee, with a specific numeric
constant tied to the optimizer. Reviewer: (a) does any grokking-dynamics paper
report a conserved margin/loss constant at the wd equilibrium; (b) is
minGap = −log(CE) + const a known triviality in the margin-maximization
literature that we are re-deriving?

## Files

- `homeostat_analysis.py` — self-contained harness (run from repo root).
- `REFERENCE_OUTPUT.txt` — exact expected output.
- Runs consumed: `runs/og_seed0/seed*/`, `runs/p-*/seed*/seed*/` (each has
  `config.json`, `checkpoints/epoch_*.safetensors`, `spectra.npz`).
