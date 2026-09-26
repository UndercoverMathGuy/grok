# R21-nu-block: literature check (2026-09-25)

Item: cyclic.tex prop:block and appendix app:block. With nu product neurons and phases alpha_l = pi l/nu,
the logit table is nu lambda^nu 2^{-n} cos(2 pi j(sum a - c)/p). Each neuron has norm nu lambda^2 p/2. One block per support frequency attains the class value.

## Verdict: PARTIALLY_KNOWN
The trigonometric identity behind the block (a nu-term separable decomposition of cos/sin of a sum of nu
variables, using equally spaced phases) is classical. It is a special case of Mohlenkamp and Monzón (2005), Thm 2.
Using the block inside max-margin networks is new as far as I found: its norm accounting, the weights lambda_j^nu proportional to y*_j,
and attainment of Gamma_s. Every ML max-margin construction I checked uses 2^{nu-1} or more neurons per frequency, or random phases.

## Evidence

1. **Mohlenkamp & Monzón, "Trigonometric identities and sums of separable functions", Math. Intelligencer 27(2):65-69, 2005.**
   The DOI 10.1007/BF02985795 was confirmed via Crossref. The preprint was read at http://www.ohiouniversityfaculty.com/mohlenka/research/sine.pdf.
   - Thm 2: sin(sum_{j=1}^n x_j) = sum_j sin(x_j) prod_{k!=j} sin(x_k+alpha_k-alpha_j)/sin(alpha_k-alpha_j), for any distinct alpha. The paper's
     conclusion is that "sin(sum x_j) is exactly separated with separation rank n", compared with 2^{n-1} terms from the standard expansion.
   - The general form has shifts a_j. Its Fig. 1 (right) uses equally spaced alpha = 0, pi/3, 2pi/3.
   - I verified numerically that choosing alpha_k = pi k/nu and shifts a_k = -alpha_k makes every term carry the same phase on all variables.
     Then prod_{k=1}^{nu-1} sin(pi k/nu) = nu/2^{nu-1} gives exactly our identity
     sum_{l=0}^{nu-1} (-1)^l prod_{k=1}^{nu} cos(x_k + pi l/nu) = nu 2^{1-nu} cos(sum x_k).
     Our block is this identity with x_nu = -theta_c. Note that cos(theta_c + n alpha_l) = (-1)^l cos(-theta_c + alpha_l).
   - The paper only conjectures that rank n is minimal ("We believe..."). Minimality is therefore not in this source.
     Our cost lemma, which says nu neurons are cost-optimal, is a separate item.

2. **Morwani et al., ICLR 2024 (arXiv 2311.07568), App. F.2** (read in the PDF). For quadratic L_{2,3} nets they use 8 neurons per frequency.
   They expand cos(a+b-c) into 4 trig-product terms and implement each with a pair of neurons, with phases (0,0,0),(0,pi,pi) and so on.
   They credit Gromov's construction scheme. There is no roots-of-unity block. Our block needs 3 neurons for nu=3, but it is a product/multilinear model,
   not their (u(a)+v(b))^2 w(c) neuron.

3. **Li, Liang, Shi, Song, Zhou, AISTATS 2025 (PMLR 258:523-531), Lemma E.2/E.3** (read in the PDF). The k-sum with power activation uses
   2^{2k-1} neurons per pair of frequencies ±zeta, that is 2^k trig terms times 2^{k-1} polarisation neurons (Lemma E.1 Part 3: sign-vector polarisation).
   There are no roots-of-unity phases. This is the exponential count that Mohlenkamp–Monzón avoid.

4. **Gromov 2023 (arXiv 2301.02679), Eqs. (6)-(16).** Phases satisfy phi1+phi2=phi3 and are otherwise uniform random. Spurious terms cancel
   only approximately ("random phase approximation", |sum cos(...)| << N). The result is approximate, not exact.
   **Doshi, He, Das, Gromov, "Grokking modular polynomials" (arXiv 2406.03495, ICLR 2024 BGPT workshop), Eq. (2)** gives the same
   random-phase scheme for S-term sums.

5. **He, Wang, Chen, Yang, arXiv 2602.16849, Def 4.1(iii) / Prop 4.2.** Their phase-symmetry condition requires the second and fourth phase moments to vanish within each
   frequency group. That is the same moment-cancellation idea (roots of unity satisfy it). It is for the quadratic sum-activation net,
   and noise peaks at 2x and 2y survive. There is no minimal block and no norm optimality.

6. The roots-of-unity filter itself (sum_l e^{2 pi i l u/nu} = nu [nu | u]) is the textbook series-multisection identity. Do not claim it as new.

## Recommendation
- Cite Mohlenkamp–Monzón (2005) at prop:block: "the block is the equally spaced case of the nu-term separable
  identity for the sine of a sum [Mohlenkamp & Monzón 2005, Thm 2]". This matters because the product-model neuron is exactly a separable
  term, so "nu neurons suffice" is precisely their separation-rank-nu statement.
- Contrast with Morwani App F.2 (8/freq), Li et al. Lemma E.3 (2^{2k-1}/freq) and Gromov / Doshi (random phases, approximate).
  Our novel content is the weighted blocks attaining Gamma_s and the cost-optimality (lem:cost), not the identity.
- Cite He et al. 2602.16849 Def 4.1(iii) for phase-moment cancellation in trained nets.

Searches: Morwani/Li/Gromov/Doshi/He PDFs grepped for construction and phases; web search for "Trigonometric identities and
sums of separable functions"; web search for the tensor rank of sin(x1+...+xd) (no minimality proof found).
