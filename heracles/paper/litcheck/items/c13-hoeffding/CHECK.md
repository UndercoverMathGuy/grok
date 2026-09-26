# C13-hoeffding: check of the cited claim

Location: sections/appendix.tex, line 189 (app:scaling, lower bound)
Verdict: SUPPORTED

## Bib details
I checked these with Crossref (api.crossref.org/works/10.1080/01621459.1963.10500830):
- Title: "Probability Inequalities for Sums of Bounded Random Variables"
- Author: Wassily Hoeffding (University of North Carolina)
- Journal: Journal of the American Statistical Association, vol. 58, no. 301, pp. 13-30, March 1963
- DOI: 10.1080/01621459.1963.10500830

The refs.bib entry matches on every field. The only fix is optional: add the DOI.

## Content
Hoeffding (1963), Theorem 2 (standard statement): let X_i be independent with a_i <= X_i <= b_i. Then
P(Xbar - E Xbar >= t) <= exp(-2 n^2 t^2 / sum (b_i - a_i)^2).

- I could not machine-read the scanned PDF mirror, so I have no verbatim quote.
- The DOI page returned 403.
- Theorem 2 is the standard textbook form, and I have no doubt about it.

The paper applies it with X_l = cos(2 pi J_l delta/p) in [-1,1] and n = L:
- The mean is -1/(p-1) < 0, so the event {mean > 1/2} needs a deviation t > 1/2.
- The bound is exp(-2 L^2 t^2 / (4L)) = exp(-L t^2/2) <= exp(-L/8).
- With L = ceil(8 ln p), this gives <= 1/p. That is correct.
- The J_l are drawn i.i.d., so independence holds.
- The union bound over the p-1 offsets gives a total failure probability of (p-1)/p < 1. That is correct.
- The complement event {mean cos <= 1/2} gives sum(1-cos) >= L/2, which matches the text (assuming cov = 1-cos).

Optional precision: write "Hoeffding's inequality (Hoeffding 1963, Thm 2)".
