# VERIFY (citation lens): C13-hoeffding

Verdict: SUPPORTED, checker upheld.

## Metadata
- Crossref (fetched this session, https://api.crossref.org/works/10.1080/01621459.1963.10500830):
  "Probability Inequalities for Sums of Bounded Random Variables", Wassily Hoeffding,
  Journal of the American Statistical Association 58(301):13-30, published 1963-03.
- refs.bib entry `hoeffding1963probability` (iclr2027/refs.bib:78-86) matches on every field.
  It lacks a doi field; adding doi={10.1080/01621459.1963.10500830} is optional.

## Content
- Wikipedia "Hoeffding's inequality" (fetched) states: independent X_i with a_i <= X_i <= b_i,
  P(S_n - E S_n >= t) <= exp(-2 t^2 / sum (b_i - a_i)^2), and attributes it to Theorem 2 of
  Hoeffding (1963), JASA 58(301):13-30. I could not get a text-layer copy of the original
  (the umbc mirror now returns HTML, and tandfonline is paywalled).
- How the paper uses it (appendix.tex:189): X_l = cos(2 pi J_l delta/p) in [-1,1], J_l i.i.d.
  uniform on F = {1..(p-1)/2}, and E X = -1/(p-1), which is correct because
  sum_{j=1}^{(p-1)/2} cos(2 pi j delta/p) = -1/2 for delta != 0.
  Event mean > 1/2 means the deviation is t = 1/2 + 1/(p-1) > 1/2.
  Thm 2 gives exp(-2 L^2 t^2 / (4L)) = exp(-L t^2/2) <= e^{-L/8} <= 1/p for L = ceil(8 ln p).
  The union bound over p-1 offsets gives a failure probability of (p-1)/p < 1. All correct.
- Hoeffding 1963 is the canonical citation, so there is no better one.

## Optional polish
Write "Hoeffding's inequality (Thm. 2 of \citet{hoeffding1963probability})" and add the doi.

## Note for the orchestrator
WebSearch refused with "used its web search budget (200 of 200 WebSearch calls)". Verifiers that
run after this point cannot search, only fetch. This probably explains part of the slow or
degraded verify stage.
