# VERIFY (novelty_math): R12 cost of a feature (lem:cost)

Verdict: the INEQUALITY is KNOWN. It follows in three lines from classical matrix analysis. What is new is the reduction (the chain lemma: JF makes H triangular with diagonal beta) and the equality characterisation in chain-vector form. The checker said PARTIALLY_KNOWN because it found no weighted l_r lower bound. That is wrong: the Rotfel'd-Thompson trace inequality gives one directly.

## The known route (checker missed it)
Setting: H = sum_l beta_l m_l m_l^*. Each term has rank 1, with its one singular value equal to nu|beta_l|, since ||m_l||^2 = nu. By the paper's own chain lemma (lem:chain), H is upper triangular with every diagonal entry beta, so |det H| = |beta|^nu = prod_i sigma_i(H).

1. Rotfel'd (1969) / Thompson (1976), inequality (1). For f increasing and concave on [0, inf) with f(0) = 0, and any square matrices A, B: sum_i f(s_i(A+B)) <= sum_i f(s_i(A)) + sum_i f(s_i(B)). Iterate this over the N rank-one terms with f(x) = x^r, 0 < r <= 1:
   sum_i sigma_i(H)^r <= sum_l (nu|beta_l|)^r.
2. AM-GM applied to the singular values of H, using prod_i sigma_i(H) = |det H| = |beta|^nu:
   sum_i sigma_i(H)^r >= nu |beta|^r.
   Combining 1 and 2 gives sum_l |beta_l|^r >= nu^{1-r} |beta|^r. This is exactly Lemma cost.
3. The count "at least nu neurons" is immediate: rank H = nu because the diagonal is non-zero, and H is a sum of N rank-one matrices, so N >= nu. The Hadamard / e_k machinery is not needed for the count.
4. Remark "other penalties" in the appendix: the same two steps work for any concave increasing f with f(0) = 0 and f(e^s) convex. This covers the paper's Bernstein class, since log(1 + t e^s) and e^s are both convex in s. So the remark is also a special case of Rotfel'd-Thompson.

Equivalent language: for r <= 1, sum_l |c_l|^r over rank-one atoms is at least the Schatten-r quasi-norm to the power r (r-subadditivity, i.e. the Rotfel'd inequality). Weyl's majorant theorem then gives sum sigma^r >= sum |lambda|^r = nu|beta|^r, because every eigenvalue of H equals beta. In short: H has all eigenvalues equal to beta, so its S_r quasi-norm is at least nu^{1/r}|beta|, and any rank-one decomposition must pay at least that.

## What remains ours
- The chain lemma (JF interval equations imply H is triangular with diagonal beta). This is the non-standard step and the real content.
- The equality characterisation for r < 1: exactly nu terms, all of modulus |beta|/nu, with orthogonal chain vectors. From the known route: AM-GM equality forces sigma_i(H) = |beta| for all i. Then H/beta is unitary and triangular, so H = beta I. Equality in Rotfel'd with strictly concave f then pins the decomposition. I found no standard statement of Rotfel'd equality cases for non-psd sums, and did not look further (see limits below). The paper's e_k/Hadamard proof gives the equality case directly, so keep it.
- The Bernstein integral x^r = c ∫ log(1 + xt) t^{-1-r} dt is the standard device in proofs of this kind. Bourin-Uchiyama extend to all concave f via integral representations.

## Citations checked this session
- Thompson, R. C., "Convex and concave functions of singular values of matrix sums", Pacific J. Math. 66(1):285-290, 1976, doi 10.2140/pjm.1976.66.285. I downloaded the PDF from msp.org and read it. Intro, eq. (1): Rotfel'd "stated and ... proved" f(gamma_1)+...+f(gamma_n) <= sum f(alpha_i) + sum f(beta_i) for increasing concave f with f(0) = 0, where alpha, beta, gamma are the singular values of arbitrary n-square A, B, A+B. Theorem 2: |A+B| <= U|A|U^* + V|B|V^* for non-Hermitian A, B. CONFIRMED.
- Bourin & Uchiyama, "A matrix subadditivity inequality for f(A+B) and f(A)+f(B)", LAA 423(2-3):512-518, 2007, doi 10.1016/j.laa.2007.02.019, arXiv math/0702475. Crossref metadata and abstract (Semantic Scholar) fetched. The result is for PSD A, B, concave f >= 0, all symmetric norms. It is secondary here, because H is not PSD. CONFIRMED (metadata and abstract only).
- Weyl 1949, PNAS 35(7):408-411, doi 10.1073/pnas.35.7.408. Crossref metadata only; PNAS and PMC blocked the full text. Not needed, since the AM-GM-on-det step replaces it. NOT READ, do not cite on its content without a read.
- Checker's citations:
  - beylkin2005algorithms: Crossref metadata matches the bib (SISC 26(6):2133-2159, 2005). I did not re-read Lemma 2.4.
  - mohlenkamp2005trigonometric: Crossref matches (Math. Intelligencer 27(2):65-69). I read the preprint PDF myself. The quote "We believe that our Theorems 2 and 5 are an example of such minimal representations" is CONFIRMED, and so is "to find a minimal rank representation ... is still an open problem" (citing Rassias-Simsa Problem 4, p. 158).
  - comon2012typical: Crossref matches (LAA 60(6):657-667, 2012). My fetch of the arXiv abstract failed to parse, so I did not re-confirm the abstract claim.

## Consequence for the checker's recommended text
Drop "Lemma cost proves the lower bound of nu terms and strengthens it to a weighted l_r cost" as a novelty claim. Suggested replacement:

"Given the chain lemma, the inequality is an instance of the Rotfel'd-Thompson trace inequality for concave functions of singular values \citep{thompson1976convex}: H is a sum of rank-one terms with singular values nu|beta_l|, and |det H| = |beta|^nu. Our contribution is the reduction from (JF) to a triangular H and the characterisation of the equality case."

The count "at least nu terms" then settles the minimality that Mohlenkamp-Monzon left open, at least for the real tensor Re(z_1...z_d) in (R^2)^{⊗d}. Projecting each factor onto span{cos, sin} shows that separation rank equals this real tensor rank. Before claiming it as first, the real rank of Re(z_1...z_d) should still be checked in the tensor literature (de Silva-Lim; Blekherman-Sinn arXiv 1511.07044, whose abstract I fetched: not directly on point).

## Search limits (why this run was thin)
- WebSearch session budget exhausted (200 of 200).
- arXiv export API returned HTTP 429.
- Semantic Scholar search returned HTTP 429.
- PNAS and PMC returned 403.
- Sources used: Crossref, msp.org, and arXiv abs pages via curl.
