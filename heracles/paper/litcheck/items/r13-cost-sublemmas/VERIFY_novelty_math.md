# VERIFY (novelty_math): R13-cost-sublemmas  (task 14)

Verdict: the checker's PARTIALLY_KNOWN holds, and both of its confirmed citations check out. But it missed a classical matrix-analysis route that makes more of the result known than it said.

## Checked the checker's citations (fetched myself)
- Rajan & Tepedelenlioglu, arXiv 1302.5383 (PDF fetched, text extracted). IEEE TIT 61(4):1619-1628, 2015, DOI 10.1109/TIT.2015.2400432 (Crossref agrees).
  Def. 1 is the ergodic capacity order. Eq. (6) is the TBF representation g = a + bx + int log(1+x/s) mu(ds), citing SSV p. 73.
  Property S1: X <=_c Y iff E g(X) <= E g(Y) for all g in TBF. App. A, eqs. (28)-(31): the rho->0 limit for the linear term, then integration against mu. CONFIRMED.
- Schilling, Song & Vondracek, Bernstein Functions, de Gruyter Studies in Math. 37. ETH TOC PDF fetched: Ch. 8 is "Thorin-Bernstein functions", p. 73. Crossref for DOI 10.1515/9783110215311 gives Schilling/Song/Vondracek, de Gruyter, issued online 2009-12-24, print year 2010. CONFIRMED.

## What the checker missed
The whole inequality of Lemma cost (all 0 < r <= 1), together with Remark "other penalties" and even Prop. gen, follows in two lines from the chain lemma plus a classical trace inequality.

1. Chain lemma (with beta = 1): H = sum_l d_l u_l u_l^* is upper triangular with unit diagonal, so |det H| = 1, all sigma_i(H) > 0, and sum_i log sigma_i(H) = 0.
2. Rotfel'd's trace inequality. Thompson, "Convex and concave functions of singular values of matrix sums", Pacific J. Math. 66(1):285-290, 1976, DOI 10.2140/pjm.1976.66.285. I fetched the PDF and read the text. Eq. (1) there, credited to Rotfel'd, and Theorem 3(i) there, which is for arbitrary square matrices: sum_i f(sigma_i(A+B)) <= sum_i f(sigma_i(A)) + sum_i f(sigma_i(B)) for f increasing, concave, f(0)=0. Apply it N-1 times to the rank-one terms d_l u_l u_l^*, whose singular values are (|d_l|, 0, ..., 0). This gives sum_l psi(|d_l|) >= sum_i psi(sigma_i(H)).
3. Jensen on s_i = log sigma_i, whose mean is 0: whenever psi∘exp is convex, sum_i psi(sigma_i) >= nu psi(1).

Every Thorin-Bernstein psi with psi(0)=0 satisfies both hypotheses. It is concave and increasing, and psi(e^s) = cs' + int log(1+t e^s) mu(dt) is convex, because c·e^s is convex and each log(1+t·e^s) is a softplus. So:
- psi = x^r gives Lemma cost (the inequality part).
- psi = log(1+xt) gives Prop. gen.
- General psi gives Remark "other penalties". This route actually gives a strictly larger class: any concave increasing psi with psi(0)=0 and x psi'(x) nondecreasing. For example psi = x on [0,1] and 1 + log x above 1 is not C^2, so it is not TBF, but it is covered.

Neither Prop. esym (e_k >= binom(nu,k)) nor the Bernstein integral is needed for the inequality. At r = 1 the same chain gives the nuclear-norm triangle inequality, sum |d_l| >= ||H||_* >= |tr H| = nu.

Numerical sanity check: 20k random complex rank-one sums, nu in 2..5, N in 1..8, r in (0.05, 1). Both steps hold to about 1e-15 once numerically-zero singular values are dropped. Script: scratchpad rot2.py.

## What remains ours
- Chain lemma: sign constraints JF force H to be upper triangular with constant diagonal (interval products). No prior statement was found.
- Equality case of Lemma cost: exactly nu non-zero, equal moduli, orthogonal chain vectors. The Rotfel'd route does not give this directly. The paper's route (generating function, unique factorization, Hadamard) does, so it keeps its value for the equality statement.
- Prop. esym for every k is a sharper intermediate (a Cauchy-Binet/Hadamard computation, textbook tools), but it is not needed for the inequality.

## Recommendation (amends the checker's change)
Do not describe "the domination itself (Lemma chain and Proposition esym)" as the new ingredient. The domination is a corollary of the chain lemma plus Rotfel'd/Thompson. Suggested text for Remark "other penalties":
"Given Lemma chain, the inequality part of Lemma cost and of this remark also follows from Rotfel'd's trace inequality \citep[Thm. 3]{thompson1976convex} applied to H = sum_l d_l u_l u_l^*, together with Jensen on log sigma_i(H), since |det H| = 1. It holds for every concave increasing psi with psi(0)=0 and psi∘exp convex, which includes the Thorin-Bernstein functions \citep[Ch. 8]{schilling2010bernstein}. The determinant argument above is needed for the equality case."
Keep the Rajan-Tepedelenlioglu S1 citation as optional. It is a correct reference for the Bernstein-transfer step, but it sits in information theory, and Thompson 1976 is the more canonical math citation for the whole inequality.
Weyl 1949 (PNAS 35(7):408-411, PMC1063049; metadata fetched, page images only) is not needed, because AM-GM/Jensen with det suffices.

## Harness note (why this stage is slow)
WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)" on my first query. The arXiv export API returned HTTP 429. From here on, verifiers in this session cannot search the web and fall back to guessing publisher URLs. That is slow, and it weakens the adversarial searches.
