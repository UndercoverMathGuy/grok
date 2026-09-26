# VERIFY (novelty_ml): T5-concave-vertex, task 49

Claim (app:shape (i)): a concave nondecreasing function over a line-free polyhedron is minimized at a vertex. With strict concavity, every minimizer is a vertex.

## Verdict: KNOWN. The checker is upheld, but it missed the ML prior art and its edit points at a sentence that no longer exists.

### 1. The text has changed since the check
- `sections/appendix.tex:134` no longer contains "Hence the minimum is attained at a vertex."
- Proof (i) is now self-contained:
  - It shows existence through compact sublevel sets, since v_j <= phi(v)^{1/r}.
  - It then uses the midpoint/strict-concavity argument.
- It no longer calls on the concave-nondecreasing/line-free vertex theorem, so the item statement is stale.
- The checker's recommended_change cannot be applied as written.
- Proof check: correct. A non-extreme point of a convex set is the midpoint of two distinct points of that set. phi = sum v_j^r is strictly concave along any nondegenerate segment in the orthant.

### 2. ML prior art the checker missed (all fetched this session)
Minimizing sum_j v_j^{1/s} with 1/s < 1 over P_p (Prop. shapes) is exactly l_r quasi-norm (0<r<1) / concave-penalty minimization over a polyhedron in the nonnegative orthant. The ML and sparse-optimization literature states the vertex property in exactly this form, and uses it to explain sparsity and feature selection. That is this paper's own theme.

- **Mangasarian, "Machine Learning via Polyhedral Concave Minimization"**
  - Applied Mathematics and Parallel Computing, Physica-Verlag, 1996, pp. 175-188, DOI 10.1007/978-3-642-99789-1_13 (Crossref).
  - Full text read from UW tech report 95-20 (.ps, decoded locally).
  - It casts feature selection as concave minimization over {Ax<=b, x>=0}, f concave and bounded below.
  - It says: "By [Rockafellar, Corollary 32.3.4] it follows that f attains its minimum at a vertex of the feasible region". It uses Cor. 32.3.3 for existence of the minimum.
- **Mangasarian, "Minimum-support solutions of polyhedral concave programs"**
  - Optimization 45(1-4):149-162, 1999, DOI 10.1080/02331939908844431 (Crossref).
  - Full text read from UW tech report 97-05.
  - The abstract is motivated by ML: "if such problems have a solution, they have a vertex solution with a minimal number of zeros".
  - Lemma 2.2 assumes a concave f bounded below on a polyhedral S with no lines, and uses "a concave function bounded below on a polyhedral set not containing lines going to infinity in both directions attains a minimum at a vertex of the polyhedral set [Rockafellar, Corollaries 32.3.3 & 32.3.4]".
- **Ge, Jiang & Ye, "A note on the complexity of L_p minimization"**
  - Math. Program. 129(2):285-299, 2011, DOI 10.1007/s10107-011-0470-2 (Crossref).
  - I read the 2009 Jiang-Ye preprint (web.stanford.edu/~yyye/lpmin.pdf).
  - Its Theorem 3 says that for min sum x_j^p s.t. Ax=b, x>=0 (0<p<1), "the set of all basic feasible solutions ... is exactly the set of its all local minimizers".
  - The second half of its proof is word-for-word the paper's midpoint/strict-concavity argument: a non-extreme point has a feasible direction +-d, and one of them descends.
  - The theorem number is from the preprint. The published numbering and author list (Ge was added) were not checked against the journal PDF.
- Consequence for the paper: none of these threaten any claim. The novelty is the reduction from networks to the programme and the structure of P_p, not the vertex fact. But an ML reviewer will recognize "concave l_r penalty over a polyhedron => vertex => sparse support <= #active constraints" (Prop. bottleneck) as the classical concave-penalty sparsity mechanism. The paper should say so in one sentence. No prose in the paper mentions l_p / concave-penalty sparsity or Mangasarian (grep).

### 3. Checker's confirmed citations
- **rockafellar1970convex**
  - Crossref DOI 10.1515/9781400873173 gives Convex Analysis, Rockafellar, PUP, 1970.
  - The Princeton UP page shows the Princeton Mathematical Series / Landmarks and the blurb "minimum or maximum of a convex function over a convex set". Vol. 28 is not shown on the page, but it is the standard number.
  - I could not fetch the Sec. 32 text (De Gruyter 405, archive.org copy convexanalysis0000rock is lending-only).
  - New secondary evidence: two independent Mangasarian papers cite **Cor. 32.3.4** for exactly this statement (vertex minimum of a concave function bounded below on a line-free polyhedral set) and Cor. 32.3.3 for existence. So "Sec. 32" is safe, and "Cor. 32.3.4" is very likely right, but it should only be printed after someone checks the book.
- **horst1996global**
  - Crossref DOI 10.1007/978-3-662-03199-5 gives Horst & Tuy, Springer, 1996, ISBN 9783642082474 / 9783662031995.
  - OpenLibrary (ISBN 9783642082474) gives the subtitle "Deterministic Approaches".
  - Crossref lists the 1990, 1993 and 1996 books, so 1996 is the 3rd edition and edition=3 is right.
  - The content was not seen. It is fine as a generic monograph pointer.
- The Benson 1995 and vom Dahl-Loehne evidence was not re-used. vom Dahl & Loehne appears in Crossref as J. Glob. Optim. 2020, pp. 37-47.

### 4. Recommended change (replaces the checker's, whose target sentence is gone)
- appendix.tex, end of proof (i): "(This is the classical vertex property of concave minimization over a line-free polyhedron \citep[Sec.~32]{rockafellar1970convex}; for $\ell_r$ objectives with $r<1$ see also \citet{ge2011note}.)"
- cyclic.tex, after the display defining P_p / Prop. shapes, optional but recommended for positioning: "Minimizing $\sum_j v_j^{1/s}$, $1/s<1$, over a polyhedron is a concave ($\ell_{1/s}$) problem whose minimizers are vertices, the same mechanism behind concave-penalty feature selection \citep{mangasarian1996machine,mangasarian1999minimum} and $\ell_p$ sparse recovery \citep{ge2011note}; what is specific here is the polyhedron $\curlyp_p$."
- Horst-Tuy is optional. Drop it if space is tight.

### Process
WebSearch budget exhausted (200/200). All sources were reached by direct URL (Crossref API, UW math-prog tech-report FTP mirror over https, Stanford preprint, OpenLibrary, Princeton UP). PostScript was decoded locally by extracting string literals.
