# opus55 — global-optimality attack (2026-09-22/23)

Read `IMPURE.md` for the impure-neuron result. This file indexes the scripts and records the outputs
of the first session. The original scratchpad (and its logs) was wiped by a `/tmp` cleanup, so those
scripts were rebuilt verbatim from the text that was run, with its fixes applied. The numbers below
are copied from that session's outputs.

## Session 1 (09-22): the whole-problem pass

| script | question | result |
|---|---|---|
| `supp_n.py` | does astra's Fourier support-count lemma (<= 2K readout frequencies per input mode) survive n = 3, 4 operands? | yes: exhaustive at (n,p) = (3,5),(3,7),(3,11),(4,5),(4,7), all supports, K = 1 included |
| `block_nu.py` | does astra's block Hessian certificate extend to nu = 4, 5? | numerically yes: nullity = n (the phase motions), smallest nonzero eigenvalue of -H is 2 at nu = 3, 4, 5 (18/32/50 variables; critical dims 10/18/28) |
| `lemmaL.py` | worst-row feature cost: sum \|c\|^r >= nu^{1-r} (min over torus of Re P)^r? | tight at the block, never beaten (n=1: 1.0000; n=2, r=2/3: N=3 1.0000, N=4 0.9962, N=5 0.9943, N=6 0.8730; n=3, r=1/2: N=4 1.0000, fewer than nu atoms always <= 0). **Proved for n = 1** (weight the killing points by \|c_l\|, which gives e2 >= m e1/4, then the Bernstein integral) |
| `densejunk.py` | does junk pay on a fixed support? | **yes on the dense support at large p**: one random-phase neuron per frequency vs the junk-free 3-neuron block, true normalized margin ratio -0.68 (p31), -0.36 (61), -0.15 (101), 0.40 (211), 0.62 (401), **0.90 (809)**, still rising toward sqrt 3. So per-support purification is false and any proof must use the support's optimality. Not a counterexample to (T): Gamma*/Gamma_dense grows with p |
| `search.py` | unrestricted raw-weight search vs the ansatz (width 3h+3, random + junky inits) | best / ansatz: **0.9986 (p17, certified Gamma)**, 0.976 (p23), 0.973 (p31). No counterexample. Calibration: the dense block reproduces its closed form to 1e-15 |

## Session 2 (09-23): impure neurons — see IMPURE.md

| script | role |
|---|---|
| `xblind_check.py` | verifies Theorem 1 (half-offset densities are exactly blind to cross-frequency content) to 1e-16 |
| `xblind_relax.py` | relaxation R: best blind-family bound over pure nets (p11: 1.30x ansatz) -> `xblind_p11.log` |
| `repair.py` | gives R's skeletons 30 free neurons and maximizes the true margin: 0.995-0.9994, never > 1 -> `repair_p11.log` |
| `repair_diag.py` | anatomy of the repaired nets: X = 0, junk = 0, pure block -> `repair_diag_p11.log` |

Run with `uv run --with torch python <script> ...` (numpy/scipy scripts: `--with numpy --with scipy`).

## Session 3 (09-23): is (T) even true at larger p?  (`purejunk.py`, `exact_gamma.py`)

Pure networks with N neurons per frequency. N <= 2 means every feature carries same-|k| junk, because
a junk-free feature needs >= 3 neurons (Feature Cost Theorem). N = 3 is the optimizer control.

* Reference Gamma*(59) >= **0.313049** (exact vertex enumeration on the support {12,14,19,25,29}, which is
  the dilation of {1,7,11,23,27}). The kstar.json candidate supports give at most 0.306644. The multistart
  baseline printed inside the logs (0.309915) was 1% low. The N = 3 control at "1.0067" is therefore
  0.9966 of the exact support value, which is not an excess.
* In the table, each value is the log number rescaled to the exact reference (multiplied by 0.990):

| N per frequency | supports tried (K from 3 to 18) | best margin / ansatz |
|---|---|---|
| 3 (control) | 6 | 0.997 (on the optimal support) |
| 2 | 16 | **0.397** (K = 14); every K <= 6 is <= 0 |
| 1 | 12 | never positive |

Junk-bearing pure networks lose by more than 2.5x at p = 59. Larger supports hurt junk less (that fits the
dense-junk trend of session 1), but nowhere near enough.

## Session 4 (09-23): closing attempts on the per-frequency junk lemma (L) at n = 2

| script | question | result |
|---|---|---|
| `search.py 3/5/7` | unrestricted search at the smallest primes (impurity is maximal relative to size) | best / ansatz: **1.0000 (p3), 1.0000 (p5), 0.9996 (p7)**; never above |
| `eig_claim.py` | do all eigenvalues of ELEGANT's chain matrix M satisfy Re lambda >= m = min_torus Re P? | **false** (Re lambda = 0.42 < m = 0.68) |
| `eig_claim_r.py`, `det_claim.py` | weaker: sum_j \|lambda_j\|^r cos(r arg lambda_j) >= 3 m^r, or pointwise \|det(I + tau M)\| >= (1 + tau m)^3? | held in block-started searches, but that was a search artefact... |
| `mirror_attack.py`, `mirror_attack2.py` | ...attack with directions that are clean-free and junk-free but carry large mirror (upper-triangle) entries | **both claims false**: Psi/(3 m^r) = 0.68 with two negative eigenvalues. (L) itself holds there with L-ratio 15-25. ELEGANT's determinant route cannot prove (L): the mirrors are exactly the uncontrolled entries it warned about |
| `lemmaL_coef.py` | coefficientwise (L): e2(\|c\|) >= m^2/3, e3(\|c\|) >= m^3/27 | **holds** at N = 3..6 (tight only at the block). Immune to mirror attacks, since cancellation raises e_k and leaves m unchanged. Killing points plus modulus bounds give only e3 >= m e2/12 (=> m^3/36): the proof has to keep the real part, as in (1-omega)(1-omega^2) = 3 at the block |
| `ratio_claim.py` | ratio forms m e1 <= 3 e2 and m e2 <= 9 e3 (they chain to coefficientwise (L); analogue of the n = 1 step m e1 <= 4 e2) | **hold**: minima 1.0024 and 1.0003, tight only at the block |
| `kill_measure.py` | can m e1 <= 3 e2 be proved with ONE population-independent killing measure (kill atom a, keep every other atom's factor <= 3/2)? | **no**: the best constant is about 1.54 (symmetric moments -1/3 give 1.527 by hand; the optimizer found 1.590). Any proof must choose the measure using the other atoms' phases |

## Session 5 (09-23): **(T) PROVED at p = 3** -> `P3_THEOREM.md`

Every network of every width (impure neurons and junk allowed) has normalized margin <= the K* = 1
ansatz at p = 3. The chain:
- the blind certificate family (IMPURE.md Thm 1) is sharp at p = 3 (`xblind_p3.log` <= 0.990);
- it reduces (T) to the one-frequency inequality (V3): sum|c|^{2/3} >= 3^{1/3}(|G| - max_T|J_T|/2)^{2/3};
- (V3) follows from two chain-matrix bounds: ELEGANT's complex one with a Jensen mirror bound
  |U_T| <= delta + 2 sqrt(e1^2-1), and a Hermitian one with each atom's phase absorbed, via Rotfel'd;
- that leaves a 2-variable inequality: analytic at its tight corner, and certified by interval
  arithmetic elsewhere (`v3_certify.py`: 108 cells, 0 failures).

Supporting scripts: `p3_lemma.py` (the lemma holds, max 0.9964), `p3_convention.py` (constant 3, minus sign),
`p3_variants.py` (V1-V5; V3 chosen), `v3_reduction*.py` / `v3_combined*.py` (development),
`v3_hermitian.py`, `margin_map.py` (pointwise margins >= 4.9%), `v3_chain_check.py` / `flag_check.py`
(chain validated on real populations), `bern_check.py`, `cell_probe.py`, `cert_debug.py`, `cb_debug.py`.
Blind relaxation at p = 5, 7 is NOT sharp (1.115, 1.265), so the route stops at p = 3 without the
charging lemma.
