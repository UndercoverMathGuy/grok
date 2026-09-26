# T4 roots-of-unity phases (prop:block and the appendix remark)

Verdict: PARTIALLY_KNOWN. The identity itself is already in the literature. What the paper adds is its role as the equality case of the cost lemma, and its use to reach the exact optimum.

## Our statement
nu product neurons with alpha_l = pi l/nu, inputs cos(theta_t + alpha_l), output cos(theta_c + n alpha_l).
- Their sum is nu 2^{-n} cos(sum theta_t - theta_c).
- Proof: the roots-of-unity filter sum_l exp(2 pi i l u/nu) = nu [nu | u].
- Numerically checked here for nu = 2..5.

## Prior art (all fetched and read this session)
1. **Mohlenkamp and Monzon, "Trigonometric identities and sums of separable functions", Math. Intelligencer 27(2):65-69, 2005, doi:10.1007/BF02985795.** This is the closest prior art.
   - Thm 2: sin(sum_{j=1}^n x_j) = sum_j sin(x_j) prod_{k!=j} sin(x_k + alpha_k - alpha_j)/sin(alpha_k - alpha_j), for any distinct alpha. So sin of a sum of n variables has separation rank (tensor rank) n, not 2^{n-1}.
   - Their Fig. 1 (right) uses alpha = 0, pi/3, 2pi/3 "with amplitude equidistributed". These are exactly our nu = 3 phases.
   - Their Remarks say the result "easily extends" to cos.
   - I checked their identity numerically with equally spaced alpha_j = pi j/nu for nu = 2..5.
   - Our nu-block is the equally spaced (roots-of-unity) cosine case of the same rank-nu separated representation.
2. **Gromov 2023 (arXiv 2301.02679), Eqs. (6)-(16).**
   - Phase constraint (12): phi1 + phi2 = phi3.
   - Unwanted terms are removed only approximately, by i.i.d. uniform random phases: sum_k cos(2 pi k s/p + phi_k) << N, Eq. (15).
   - Quote: "each frequency appears several times with different random phases ultimately leading to a better wave interference."
   - So the cancellation is random and asymptotic, not exact.
3. **Morwani et al. ICLR 2024, App. F.2.**
   - 8 neurons per frequency, with phase triples (0,0,0), (0,pi,pi), (pi/2,-pi/2,0), (pi/2,pi/2,pi), ...
   - The 4 cos/sin terms of cos(a+b-c) are each built by a sign pair. Cancellation is exact, but it uses +-1 and quarter-turn phases, 8 neurons against our 3.
   - App. C (parity): 2^{k-1} neurons over sign vectors sigma; "every term other than Pi x_{S_i} cancels out in the sum".
4. **Li, Liang, Shi, Song, Zhou, AISTATS 2025, App. E.2/E.3 (Lemma E.3).**
   - 2^{2k-1} neurons per frequency: 2^k cos/sin terms (Eq. 25), each built by 2^{k-1} neurons.
   - This uses the +-1 polarization identity 2^k k! prod d_i = sum_{c in {+-1}^k} (-1)^{(k - sum c)/2} (sum c_j d_j)^k.
   - This is the Z_2 (square-root-of-unity) version: exponential width, exact cancellation.
5. **He, Wang, Chen, Yang arXiv 2602.16849, Def. 4.1(iii).**
   - Full diversification requires sum_{m in N_k} exp(i iota phi_m) = 0 for iota in {2,4}. This is a phase-moment cancellation that equally spaced phases satisfy.
   - It is used in quadratic nets (Prop 4.2), where the 2x and 2y noise peaks are left uncancelled.
6. **Standard math: roots-of-unity filter / series multisection** (textbook).
   - Canonical cite if one is wanted: any algebra text. Mohlenkamp-Monzon is the better cite for the rank-nu trigonometric decomposition.

## What remains ours
- Using the nu-th-roots-of-unity block as the unique equality configuration of the cost lemma (appendix remark: orthogonal chain vectors iff zeta is a non-trivial nu-th root of unity).
- Using it to certify the interference-free optimum with the minimum nu neurons per feature. The ML constructions above use 8 (Morwani) and 2^{2k-1} (Li), or approximate random phases (Gromov).
- To our knowledge, no ML paper uses the rank-nu separated form. Mohlenkamp-Monzon conjecture that it is minimal; our Lemma cost gives a norm-optimality version.

## Recommended text change
After "The phases are a rotated copy of the nu-th roots of unity..." (cyclic.tex l.58) add:
"This is the equally-spaced case of the rank-$\nu$ separated representation of a trigonometric function of a sum of $\nu$ variables \citep{mohlenkamp2005trig}. Earlier max-margin constructions cancel the unwanted terms with $\pm1$ sign patterns, using 8 neurons per frequency for $\nu=3$ \citep{morwani2024feature} and $2^{2\nu-3}$ in general \citep{li2025nsum}. \citet{gromov2023grokking} cancels them only approximately, with random phases."
