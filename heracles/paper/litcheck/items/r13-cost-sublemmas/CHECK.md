# R13-cost-sublemmas: lit check (2026-09-25)

Item: appendix lem:chain, lem:minor, lem:hadamard, prop:esym, prop:gen, lem:bernstein, proof of lem:cost, Remark "other penalties".

## Verdict: PARTIALLY_KNOWN
The combinatorial core (chain lemma giving an upper-triangular H with diagonal beta; det(I+tH) = (1+t)^nu gives e_k(x) >= binom(nu,k)) is our own argument. Its ingredients are textbook: Sylvester's determinant identity, principal-minor/Cauchy-Binet expansion, Hadamard's inequality. We found no prior statement of the result.
The **transfer step is known** as a general principle. It says: prod(1+x_l t) >= (1+t)^nu for all t implies sum psi(x_l) >= nu psi(1) for every psi = cx + int log(1+xt) mu(dt), and in particular for x^r, 0<r<1. The class in the Remark is exactly the Thorin-Bernstein functions with a=0.

## Evidence (fetched)
1. Rajan & Tepedelenlioglu, "Stochastic Ordering of Fading Channels Through the Shannon Transform", IEEE Trans. Inf. Theory 61(4):1619-1628, 2015 (Crossref, DOI 10.1109/TIT.2015.2400432; arXiv 1302.5383).
   - Def. 1 is the "ergodic capacity order": X <=_c Y iff E log(1+rho X) <= E log(1+rho Y) for all rho >= 0.
   - Property S1 is: "X <=_c Y <=> E[g(X)] <= E[g(Y)], for all g in TBF". TBF means Thorin-Bernstein functions, their eq. (6): g(x) = a + bx + int log(1+x/s) mu(ds).
   - The proof in App. A (Property M1, eqs. 28-31) is our proof. They take rho -> 0 to get the linear term, then integrate log(1+t rho x) against a positive measure. Applying it to the empirical measure of x (with zero padding, allowed because a=0) gives our Remark verbatim. Our Lemma cost for 0<r<1 is the special case g = x^r.
2. Schilling, Song, Vondracek, *Bernstein Functions: Theory and Applications*, de Gruyter Studies in Math. 37. The 1st ed. is 2010 (TOC fetched): Ch. 6-7 cover complete Bernstein functions and Ch. 8 covers Thorin-Bernstein functions (p. 73). The 2nd ed. is 2012, DOI 10.1515/9783110269338. This is the standard source for the representation g(x) = a + bx + int log(1+x/s) mu(ds). The same material contains the formula x^r = (r sin(pi r)/pi) int_0^inf log(1+xt) t^{-1-r} dt, which is our Lemma bernstein and a direct Beta-integral computation.
3. The step from e_k inequalities to the generating inequality is trivial: nonnegative coefficients plus termwise comparison. The majorization literature I checked (power majorization, Cuttler-Greene-Skandera symmetric-mean inequalities, trumping) contains no statement that is closer. The implication runs e_k domination -> log-generating domination -> Thorin-Bernstein sums, and that is the ergodic-capacity order above.

## What is ours
- The chain lemma. The linear sign constraints (JF) make H = sum beta_l m_l m_l^* upper triangular with constant diagonal, so the spectrum is pinned.
- Using det(I+tH), minor expansion and Gram determinants <= 1 to get e_k(|nu beta|) >= binom(nu,k), i.e. the log-generating domination x >=_c (nu copies of 1).
- The equality case: exactly nu atoms of modulus |beta|/nu on orthogonal chain vectors.

## Searches
- "Thorin-Bernstein function representation ... Schilling Song Vondracek"
- "elementary symmetric polynomials inequality e_k(x) >= e_k(y) implies sum x^p ... log(1+tx)"
- Followed up: arXiv 1302.5383 full text and Crossref metadata for both sources.

## Recommended change
Remark "other penalties": replace the last sentence with "The admissible psi are exactly the Thorin-Bernstein functions vanishing at 0 \citep[Ch. 8]{schilling2010bernstein}. Our Proposition gen says that x dominates nu copies of 1 in the log-generating (ergodic-capacity) order, and the transfer to all such psi is Property S1 of \citet{rajan2015stochastic}." In the proof of Lemma cost, add "(a standard Thorin-Bernstein representation; see \citealp{schilling2010bernstein})" after Lemma bernstein. Keep the chain lemma and prop:esym presented as our own.
