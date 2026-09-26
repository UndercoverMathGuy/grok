# R22 finitely-many-shapes (cyclic.tex prop:shapes; appendix app:shape)

Verdict: PARTIALLY_KNOWN. Both proof ingredients are classical, and one is an almost direct specialisation of a known theorem. Applying them to Gamma_s(p) and the resulting "piecewise-constant shape in s" statement were not found anywhere.

## The three steps of the proof and where each comes from
(i) Minimisers of phi(v) = sum v_j^{1/s}, s>1, over the polyhedron P_p are vertices, and every minimiser is a vertex.
  - This is classical concave minimisation: a concave function on a polyhedron that contains no line attains its infimum at an extreme point. Reference: Hirsch & Hoffman, "Extreme varieties, concave functions, and the fixed charge problem", CPAM 14(3):355-369, 1961 (metadata from Crossref, DOI 10.1002/cpa.3160140313; the Wiley page returned 403, so I could not read the body).
  - Near-exact specialisation: Ge, Jiang & Ye, "A note on the complexity of Lp minimization", Math. Program. 129(2):285-299, 2011 (DOI 10.1007/s10107-011-0470-2, confirmed via Crossref; I read the preprint at web.stanford.edu/~yyye/lpmin_v14.pdf). Problem (1): minimise sum x_j^p subject to Ax=b, x>=0, with 0<p<1. Preprint Theorem 3: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers." Their proof that a non-vertex cannot be a local minimiser uses the same strict-concavity-along-a-segment argument as our step (i). Our P_p = {v>=0 : C v >= 1} becomes their standard form once slack variables are added. The paper also proves global lp minimisation over a polyhedron is strongly NP-hard (Thm 1), which is useful context for why we enumerate supports exhaustively only for small K.
(ii) Only finitely many switches in s. The difference phi(v)-phi(v') is an exponential sum sum c_k e^{lambda_k u}, and Rolle induction bounds its real zeros by N-1.
  - This is Laguerre's extension of Descartes' rule of signs to generalised polynomials / exponential sums (Polya-Szego, Problems and Theorems II, Part V). An expository treatment with the same Rolle proof is G. J. O. Jameson, "Counting zeros of generalised polynomials: Descartes' rule of signs and Laguerre's extensions", Math. Gazette 90(518):223-234, 2006 (DOI 10.1017/S0025557200179628, confirmed via Crossref and the Cambridge page; the body is paywalled, and the title and abstract confirm the scope). The sharper form bounds the zeros by the number of sign changes in c_k.
(iii) Consequences for support, amplitude ratios and shares. These follow directly from (i) and (ii).

## Search for the combined statement (not found)
Queries: "lp quasi-norm minimization polyhedron solution as function of p piecewise constant breakpoints"; "concave minimization polytope parametric exponent sum x_i^p optimal vertex changes finitely many times"; plus the Ge-Jiang-Ye and Hirsch-Hoffman lookups. Parametric-LP results (a linear cost with a parameter, giving piecewise optimal bases) are an analogue in spirit. Nothing I found addresses a parametric exponent in an lp objective, and nothing addresses the coverage polyhedron. The ML sweep (Morwani, Li et al., Kam et al., McCracken) has no result on shape as a function of s.

## Correctness notes on our proof
- (i) is fine. The strict concavity of t -> t^r on [0,inf) is used correctly, and a line-free recession decomposition holds because P_p lies in the nonnegative orthant.
- (ii) is fine. It could say "at most (number of sign changes of c) zeros" (Laguerre), which is sharper, but N-1 is enough.
- Wording: at a switch value of s the minimising set is the union of the two adjacent sets. So "finitely many intervals covering (1,inf)" must allow degenerate one-point intervals, or should say "outside finitely many values of s". This is minor.

## Recommendation
Add one sentence to the appendix proof: "(i) is the classical fact that a concave function attains its minimum over a polyhedron at a vertex [Hirsch-Hoffman 1961]; for lp objectives, 0<p<1, vertices are exactly the local minimisers [Ge-Jiang-Ye 2011, Thm 3 of the preprint]. (ii) is Laguerre's generalisation of Descartes' rule of signs to exponential sums [Jameson 2006]." In the main text, present the proposition as standard concave-minimisation facts applied to Gamma_s, not as a new technique. The novelty is the consequence: the optimum holds a fixed frequency set and fixed amplitude ratios, then jumps.
