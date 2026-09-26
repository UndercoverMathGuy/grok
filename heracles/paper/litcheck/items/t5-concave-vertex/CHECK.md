# T5-concave-vertex: "concave nondecreasing fn over line-free polyhedron minimized at a vertex; strict concavity => all minimizers are vertices"

Location: sections/appendix.tex line 151, proof of Prop. shapes (i).

## Verdict: KNOWN (textbook classical). Our proof is a correct self-contained argument; add a citation pointer.

## Sources (fetched / confirmed this session)
- Rockafellar, Convex Analysis, Princeton Univ. Press 1970 (Princeton Math. Series 28). Metadata confirmed:
  Princeton UP page (press.princeton.edu/books/paperback/9780691015866/convex-analysis) and Crossref DOI 10.1515/9781400873173.
  Section 32 = "The Maximum of a Convex Function" (pp. 342-346; confirmed only via a search snippet).
  The exact theorem text could NOT be fetched (JSTOR/Google Books/De Gruyter blocked). From memory: Thm 32.2 covers
  supremum attained at an extreme point for compact sets; Thm 32.3 extends this to closed convex sets containing no lines;
  Cor. 32.3.x covers polyhedral sets. THE THEOREM NUMBER IS UNVERIFIED, so cite "Sec. 32" rather than a theorem number.
- Horst & Tuy, Global Optimization: Deterministic Approaches, Springer 1996 (3rd ed.), DOI 10.1007/978-3-662-03199-5
  (Crossref-confirmed). The standard reference for concave minimization. Theorem number not confirmed.
- Benson, "Concave Minimization: Theory, Applications and Algorithms", in Horst & Pardalos (eds.), Handbook of Global
  Optimization, Springer 1995, pp. 43-148, DOI 10.1007/978-1-4615-2025-2_3 (Crossref-confirmed). A survey of the
  property that the minimum sits at an extreme point.
- vom Dahl & Loehne, arXiv 1905.11228 (J. Global Optim. 2020). The proof of their Thm 1 uses the same argument we do:
  decompose the polyhedron as polytope Q + recession cone, use "Concavity of f implies that f attains its minimum at a
  vertex of Q", and use monotonicity along recession directions (their eq. (5)). This shows the argument is standard.

## Check of our proof
- Every point of P is a convex combination of its vertices plus a recession direction d >= 0. P lies in the nonnegative
  orthant, so its recession cone does too. Because phi is nondecreasing, phi(v) >= phi(conv part) >= min over vertices. Correct.
- Strictness: for v' != v'', at least one coordinate varies along the segment, and t^r (0<r<1) is strictly concave on
  [0,inf), so phi(v) > average. Correct. The strictness half is a one-line folklore corollary; no citation needed.
- The recession/monotone step is the part that isn't pure Rockafellar (he needs f bounded above along half-lines / the
  supremum attained). Our monotonicity hypothesis supplies that directly, so the proof is self-contained.

## Recommendation
Add after "Hence the minimum is attained at a vertex": "(a classical fact; see Rockafellar 1970, Sec. 32; Horst & Tuy 1996)".
No novelty claim is affected.
