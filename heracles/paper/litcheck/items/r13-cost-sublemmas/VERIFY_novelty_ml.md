# VERIFY novelty_ml: R13-cost-sublemmas (task 13)

Verdict: the checker's PARTIALLY_KNOWN is upheld. Both of its confirmed citations check out. It missed one ML precedent (Tian) for the equality case, and the "Bernstein-type" wording needs tightening to Thorin-Bernstein.

## Checker citations: re-verified
- rajan2015stochastic (arXiv 1302.5383v2; PDF fetched, text extracted). Def. 1 is the ergodic capacity order, with generator log(1+rho x), rho >= 0. Eq. (6) is TBF: g(x) = a + bx + int_(0,inf) log(1+x/s) mu(ds), citing Schilling et al. p. 73. Property S1: X <=_c Y <=> E g(X) <= E g(Y) for all g in TBF. Appendix A, proof of M1, eqs. (28)-(31): take rho -> 0 for the trace term, then integrate log(1+t rho x) against mu(t). This is the same argument as Lemma cost and Remark "other penalties". Crossref: IEEE TIT 61(4):1619-1628, 2015, DOI 10.1109/TIT.2015.2400432. The bib is correct.
  - Caveat on use: S1 compares random variables. Comparing N numbers against nu copies of 1 needs zero padding, which needs psi(0) = 0 (a = 0). The Remark's psi = cx + int log(1+xt) mu(dt) satisfies this.
- schilling2010bernstein: Crossref DOI 10.1515/9783110215311 gives Schilling, Song and Vondracek, "Bernstein Functions", de Gruyter, online 2009-12-24 (print 2010). Rajan et al. cite [9, pp. 73-79] and [9, p. 73] for the Thorin-Bernstein representation, which independently confirms the Ch. 8 / p. 73 locator. The bib is fine.

## Missed prior art (ML lens)
1. Tian, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets", arXiv 2410.01779 (v4, 2025-09-30; NeurIPS 2025 per the arXiv page). PDF fetched.
   - Appendix, eqs. (86)-(96): 2-layer quadratic-activation net on modular addition. This is the product/power model at nu = 3.
   - Setting: an order-3 per-frequency solution under the cancellation constraints R_c, where the sign-flipped cross terms vanish, sum_j m_j e^{i theta_j} = 0. Tian solves max sum_j m_j x_j^3 - eps sum_j m_j^2 subject to sum_j m_j x_j = 0 (eq. 93). The Lagrangian gives m_j = 1/eps and the phases [1, omega_3, omega_3bar] (eq. 95), i.e. cube roots of unity with equal weights.
   - This is the equality configuration of Lemma cost / Remark "orthogonality as roots of unity" at nu = 3, found in ML, but only as a stationarity computation: l2 penalty, 3 nodes, no global inequality, no general nu.
   - Tian's Theorem 5 proof (eqs. 103-105) uses AM-GM plus concavity: sum |a_j b_j c_j|^{2/3} >= |sum a_j b_j c_j|. That is the trivial, no-cancellation version of the power-sum cost.
   - Tian is not in refs.bib; grep finds only "Tianyi" in Li et al. Other litcheck items already flag Tian (sweep/LIT_ml.md, r2, r8). For this item it should be cited as the nu = 3 special case of the equality structure.
2. Zhang, "The Bernstein Function: A Unifying Framework of Nonconvex Penalization in Sparse Estimation", arXiv 1312.4719 (2013). PDF fetched.
   - Bernstein functions as a unified family of sparsity penalties (LOG, EXP, KEP, LFR), bridging l_q (0 <= q < 1) and l_1. This is ML precedent for the idea that "the result is not special to power penalties".
   - Warning: the transfer does NOT extend to all Bernstein functions, only to Thorin-Bernstein ones. Counterexample with nu = 2 and x = (4, 1/4): e_1 = 4.25 >= 2 and e_2 = 1 >= 1, so prod(1+x_l t) >= (1+t)^2. But the EXP penalty psi(x) = 1 - e^{-x} (Bernstein, not Thorin) gives psi(4) + psi(1/4) = 1.2029 < 2 psi(1) = 1.2642.
   - So the item's phrase "any Bernstein-type penalty" must read "Thorin-Bernstein". The Remark's formula already has the right class. If Zhang is cited, add this caveat.

## Not found
- No ML or learning-theory statement of the chain lemma: under interval-cancellation (JF) constraints, H = sum beta_l m_l m_l^* is upper triangular with constant diagonal. Also none of e_k(|nu beta|) >= binom(nu,k) with orthogonal equality.
- The general-nu cost bound sum |beta_l|^r >= nu^{1-r} |beta|^r under JF was not found either.
- Search coverage was limited, see the infra note below.

## Infra note (why the workflow is slow)
- This run hit "WebSearch budget exhausted (200 of 200 calls this session)" on its first search. The session-wide search cap is shared across the parallel verifiers.
- The arXiv export API returned HTTP 429 (rate-limited).
- Verifiers launched after the cap can only WebFetch known URLs. Raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION or throttle arXiv API use.
