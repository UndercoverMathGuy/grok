# VERIFY (novelty_math) — R22 finitely many shapes (cyclic.tex prop:shapes; appendix app:shape)

Verifier: task 20, novelty_math lens.

**Verdict: PARTIALLY_KNOWN (the checker's verdict stands).**
**upheld = false.** One citation the checker marked confirmed, ge2011note, is misapplied: its recommended sentence is false for the paper's polyhedron P_p.

## Search environment
- WebSearch was refused: "this session has used its web search budget (200 of 200 WebSearch calls)".
- DuckDuckGo HTML worked for one round and then served a bot page. Semantic Scholar returned 429.
- I searched through the Crossref, OpenAlex, arXiv and StackExchange APIs, and fetched primary texts directly: author PDFs, the archive.org scan, and the UW-Madison ftp.

## Primary sources read this session
1. **Hirsch & Hoffman, "Extreme varieties, concave functions, and the fixed charge problem"**
   - Read the full text of the IMM-NYU 277 report (Feb 1961) on archive.org (extremevarieties00hirs). This report is the preprint of CPAM 14(3):355-369, 1961 (Crossref metadata confirmed).
   - Thm 3.1: a concave function bounded below on a closed convex K attains its infimum iff its restriction to the extreme points of M = K ∩ L^⊥ does; "when it is assumed, it is assumed on E_M".
   - Cor 3.2: "... in particular ψ assumes its infimum on a closed convex polyhedron."
   - P_p lies in the nonnegative orthant, so it is line-free: L = {0} and M = P_p. φ ≥ 0 is bounded below. Thm 3.1 with Cor 3.2 therefore gives existence of a minimiser and a minimiser at a vertex.
   - The strict half, that *every* minimiser is a vertex, is the paper's own strict-concavity argument (folklore).
   - **The checker's "confirmed" survives.** The theorem numbers come from the report version.
2. **Jameson, Math. Gazette 90(518):223-234 (2006)** (author PDF, maths.lancs.ac.uk/jameson/zeros.pdf)
   - Thm 3.1: the zeros of F(x) = Σ a_j e^{p_j x} number at most S[(a_j)], the number of sign changes.
   - Cor 3.2: a Dirichlet polynomial of length n has at most n−1 zeros, counted with order. The remark after Cor 3.2 gives exactly the paper's proof: multiply by e^{−p_k x} and differentiate, with Rolle.
   - φ(v) − φ(v') as a function of u = 1/s is precisely such an F.
   - **Survives.**
3. **Ge, Jiang & Ye, Math. Program. 129(2):285-299 (2011)** (preprint lpmin_v14.pdf)
   - Metadata and the Thm 3 quote are correct: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers."
   - The setting is standard form, {Ax=b, x≥0}, with the objective Σ x_j^p over ALL variables. The BFS ⇒ local-min direction of the proof uses the entering variable's term (ε d_{m+1})^p > 0.
   - The checker says P_p "reduces to their standard form once slack variables are added". That fails, because slacks carry zero objective weight.
   - **Counterexample (I recomputed it):** P = {v≥0 : v1+v2 ≥ 1, 0.4v1+1.2v2 ≥ 1} has the vertex (0.25, 0.75). The direction (−t, +t) is feasible, and √v1+√v2 falls: 1.3660 at t=0 → 1.3617 at t=0.01 → 1.3416 at t=0.05. So vertices of inequality-form polyhedra need not be local minimisers.
   - The recommended sentence "for ℓp objectives with 0<p<1, the vertices are exactly the local minimisers [ge2011note]" would therefore be FALSE as applied to P_p. The reverse direction (local minimiser ⇒ vertex) does transfer. So does Thm 1 (strong NP-hardness).

## Prior art the checker missed (the s → ∞ end of the same argument)
- **Fung & Mangasarian, JOTA 151(1):1-10, 2011**, DOI 10.1007/s10957-011-9871-x (full tech report 11-02 read)
  - Prop. 1 covers systems Ax=a, Bx≥b, ‖x‖∞≤1, which includes inequality form like P_p.
  - Proof: "(6) has a vertex ... as a solution for each q" (citing Rockafellar Cor. 32.3.3/32.3.4). "Since T has a finite number of vertices, one vertex ... will repeatedly solve (6) for some increasing infinite sequence" of q = 1/p. That vertex solves the ℓ0 problem.
  - This is our finite-vertex-set argument, used along a subsequence and only at the r → 0 end, with no switch count.
- **Mangasarian, Optimization 45:149-162, 1999**, DOI 10.1080/02331939908844431 (abstract read from ftp.cs.wisc.edu tech report 97-05.abs)
  - A concave minimisation over a polyhedron that has a solution has a vertex solution of minimal support.
  - A smooth concave exponential approximation of the step function solves the minimum-support problem exactly at a finite parameter.
  - This is the same parametric-family, finite-vertex pigeonhole.
- **Peng, Yue & Li, IEEE TIT 2015**, DOI 10.1109/tit.2015.2429611 ("NP/CMP equivalence")
  - Crossref metadata only; I did not read it. The same small-p ℓp = ℓ0 endpoint, for equalities.
- Free corollary for the paper: on the last s-interval (r → 0, φ(v) = |supp v| + r Σ log v_j + O(r²)), the maximiser is a minimum-support vertex of P_p. Ties are broken by the geometric mean of its entries.

## Novelty search (math lens)
- Queries covered: ℓp/quasi-norm minimisation paths in p; parametric concave minimisation with finitely many optimal-vertex changes; Laguerre/Descartes applied to power sums in optimisation; MathOverflow/MSE.
- None found stating that the argmin vertex set of Σ v_j^r over a polyhedron is piecewise constant in the exponent with finitely many switches.
- Parametric-LP (Gass–Saaty type) covers only costs linear in the parameter.
- The result is a routine but, as far as I can find, unpublished combination: finite vertex set, plus concave min at a vertex (Hirsch–Hoffman), plus Laguerre's rule (Jameson). Its application to Γ_s(p) is new. PARTIALLY_KNOWN is right.

## Proof check
- (i) is correct. φ is strictly concave on R^n_{≥0}, since distinct points differ in some coordinate. Sublevel sets are compact because v_j ≤ φ^{1/r}.
- (ii) is correct. Between consecutive zeros, every non-tied pair keeps a fixed sign, so the whole φ-order of the vertices, and hence the argmin set, is constant.
- The checker's wording fix has **already been applied** in the current cyclic.tex ("constant on each component of (1,∞)∖E").
- Optional sharpening: the count of switches for a pair is at most the number of sign changes of (c_k) (Jameson Thm 3.1).

## Recommended text (appendix app:shape, one sentence after the proof)
"Step (i) is the classical fact that a concave function bounded below on a line-free polyhedron attains its minimum at a vertex \citep[Thm.~3.1, Cor.~3.2]{hirsch1961extreme}; strict concavity makes every minimiser a vertex. Step (ii) is Laguerre's extension of Descartes' rule of signs to exponential sums \citep[Thm.~3.1, Cor.~3.2]{jameson2006counting}. The same finite-vertex argument appears at the $s\to\infty$ end, where small-exponent $\ell_p$ minimisers over polyhedra are minimum-support vertices \citep{mangasarian1999minimum,fung2011equivalence}."

Do NOT cite ge2011note for "vertices are exactly the local minimisers" of P_p.
