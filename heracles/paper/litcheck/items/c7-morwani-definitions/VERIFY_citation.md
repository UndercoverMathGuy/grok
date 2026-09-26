# VERIFY (citation lens): C7-morwani-definitions (task 81)

Verdict: the checker's SUPPORTED verdict holds, and so does its one confirmed citation.

## Sources fetched this session
- arXiv PDF 2311.07568v2 (19 Feb 2024), fetched with curl and read with pdftotext.
- The arXiv abstract page gives the title and all five authors, and the comment "Accepted as Spotlight at ICLR 2024".
- OpenReview API search finds forum i9wDX850jR, "ICLR 2024 spotlight", with the same five authors (Kakade listed as "Sham M. Kakade").

## Line-by-line check against the PDF
| Paper text | Morwani location | Status |
|---|---|---|
| prelim l.3: L_{2,b} norm (sum_i ||w_i||_2^b)^{1/b}; "b=nu is the norm of Morwani" | Sec 2 defines ||theta||_{a,b} for general a,b; Sec 3.2 says "we will primarily focus on maximum margin with respect to L2,nu norm" | OK |
| prelim l.3: homogeneous neurons, no biases | Sec 2: "homogeneous polynomial activations ... no biases"; phi(lambda w,x)=lambda^nu phi(w,x) | OK |
| prelim l.5: power neuron at n=2 is Morwani's quadratic neuron | Sec 2: phi({u,v,w},a,b)=(u_a+v_b)^2 w, one-hot inputs, w in R^{|G|} | OK |
| prelim l.5: D_p notation | Sec 4: D_p := {((a,b),a+b)} | OK (same symbol) |
| approach l.3: g, h, gamma*_b = max_{||theta||<=1} h | Sec 3: g = f[y] - max_{y'!=y} f[y']; h = min_D g; gamma* = max_{theta in Theta} h with Theta={||theta||<=1} | OK. Morwani define the normalized margin as h(theta/||theta||); the paper's h(theta)/||theta||^nu is the same by nu-homogeneity, and the paper says "by homogeneity". |
| approach l.3: Wei Thm 4.1, CE + lambda||theta||^b | Morwani Thm 1 restates Wei et al. (2019a) Thm 4.1: "For any norm, a fixed r>0 and any homogeneous f ... lim gamma_lambda = gamma*" | OK. This is a consistency check only; the Wei paper was not re-fetched here. |
| approach l.7: min margin <= averaged class-weighted margin, "[Section 3]" | Sec 3.2: g'(theta,x,y) >= g(theta,x,y) "as g' replaces the max by a weighted sum" | OK. Section 3.2 is more precise. |
| approach l.11: eta_{u,v,w}(delta) | Defined only in App F.1: eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)] | OK. Add the pointer App. F.1. |
| approach l.36: phase condition, "[Section 4]" | Thm 7 part 1, in Sec 4: theta_u* + theta_v* = theta_w* | OK |
| approach l.54: pricing is a single-neuron optimization | Lemma 5 ("Single neuron optimization" part) and the Sec 3.3 blueprint | OK |
| approach l.70 / appendix l.32: sqrt(2/27)/(p^{1/2}(p-1)) is "exactly their maximum L_{2,3}-margin" | Thm 7: "with m >= 4(p-1) neurons. The maximum L2,3-margin ... gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1))" | OK. The arithmetic checks: 4*6^{-3/2} = 0.27217 = sqrt(2/27). Their value carries the width condition m >= 4(p-1). |

## Bib entry
The title, the five authors and ICLR 2024 are all correct. You could add `url={https://openreview.net/forum?id=i9wDX850jR}`. This is optional.

## Other issues found (outside this item's scope)
- intro.tex l.4 has a truncated sentence: "...for quadratic networks under a specific ." It needs fixing before submission.
- intro.tex l.4 says Morwani "introduced the principle of margin maximization". That credits them with too much: max-margin analysis of homogeneous networks predates them (Wei et al. 2019; Lyu & Li 2019, both cited by Morwani). They introduced using it to pin down the learned features. A safer wording is "used margin maximization to characterize the features learned ...".

## Recommended edits (the same as the checker's, all confirmed)
1. approach.tex l.11: \citet[App.~F.1]{morwani2024feature} for eta.
2. approach.tex l.7: change Section~3 to Section~3.2.
3. approach.tex l.70 and appendix.tex l.32: optionally add "(attained at width m >= 4(p-1); their Theorem 7)".

## Process note
The session-wide WebSearch budget is exhausted: 200 of 200 calls used. Any later verifier that relies on WebSearch will fail or stall. This task was finished with direct fetches (curl, WebFetch, the OpenReview API).
