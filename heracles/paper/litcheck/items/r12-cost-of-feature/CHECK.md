# R12-cost-of-feature: lit check (2026-09-25)

Item: cyclic.tex lem:cost / appendix app:cost. Under (JF), sum_l |beta_l|^r >= nu^{1-r}|beta|^r for r in (0,1]. For r<1, equality holds iff there are exactly nu atoms of modulus |beta|/nu with pairwise orthogonal chain vectors.

## Verdict: PARTIALLY_KNOWN (the count and existence parts). The weighted l_r bound and the equality case were not found.

The paper does not currently say this, but it should: (JF) is a statement about exact rank-1 decompositions. On the frequency-j block, each product neuron is a real rank-1 term in (R^2)^{⊗nu}. (JF) says the terms sum exactly to cos(theta_1+...+theta_n-theta_0), which is the tensor Re(z_1...z_n conj(z_0)). So:
- The r->0 content of the lemma ("at least nu neurons") is a real-tensor-rank lower bound, rank_R >= nu, for the "cosine of a sum" tensor. Over C this tensor has rank 2: it is the sum of the two complex exponentials.
- The nu-block (Prop. block) is a rank-nu decomposition with equally spaced phases.

## Evidence (fetched)
1. Beylkin & Mohlenkamp, "Algorithms for numerical analysis in high dimensions", SIAM J. Sci. Comput. 26(6):2133-2159, 2005 (Crossref-confirmed; PDF read).
   - §2.1.1 "Sine of a sum", Lemma 2.4 (eq. 2.9): sin(sum_j x_j) = sum_{j=1}^d sin(x_j) prod_{k != j} sin(x_k+alpha_k-alpha_j)/sin(alpha_k-alpha_j). This gives d real separable terms, compared with 2^{d-1} from the angle-addition formulas.
   - Def. 2.2 defines the condition number kappa = (sum s_l^2)^{1/2}/||F||, an l_2 cost of the coefficients. §3 remarks that "choosing alpha_j equally spaced in the identity leads to kappa ~ 7", i.e. equal spacing is the well-conditioned choice. That is the same qualitative message as our equality case, with l_2 in place of l_r, r<1, and without proof.
2. Mohlenkamp & Monzón, "Trigonometric identities and sums of separable functions", Math. Intelligencer 27(2):65-69, 2005 (Crossref-confirmed; preprint PDF at ohiouniversityfaculty.com/mohlenka/research/sine.pdf read).
   - Theorem 2 is the general n-variable identity.
   - On minimality they say: "to find a minimal rank representation ... is still an open problem. We believe that our Theorems 2 and 5 are an example of such minimal representations." So they give the upper bound n but do NOT prove the lower bound.
3. Comon & Ottaviani, "On the typical rank of real binary forms", Linear Multilinear Algebra 60(6):657-667, 2012 (arXiv 0909.4865; Crossref-confirmed). Abstract: "a real binary form of degree d with d real roots has rank d."
   - Re((x+iy)^d) has d real roots, so its real Waring rank is d. This is the symmetric (power-neuron) analogue of "at least nu terms". It is not a weighted cost bound.
   - Blekherman, "Typical real ranks of binary forms", FoCM 2015 (arXiv 1205.3257, fetched), proved their conjecture on typical ranks. Only needed if you discuss typical ranks.
4. Huang, Tang, Ye, Zhu, "The Waring problem of harmonic polynomials", arXiv 2601.03560 (Jan 2026, fetched abstract). Real binary harmonic forms have Waring rank equal to their degree. Same count statement, symmetric, unweighted.
5. Complex multiplication (nu=3) has real bilinear rank 3. This is classical (Gauss trick, 3 multiplications) and the lower bound is folklore. It is our nu=3 count.
6. The Thorin-Bernstein transfer step is known (Rajan-Tepedelenlioglu 2015, Property S1; Schilling-Song-Vondracek). See items/r13-cost-sublemmas/CHECK.md.
7. The ML precedents for phase cancellation (He et al. Def 4.1(iii), Gromov, Morwani App F.2, Li App E) are covered in items/r9 and r10.

## Not found
- A proof that the real (non-symmetric) tensor rank of Re(z_1...z_d) in (R^2)^{⊗d} equals d. Mohlenkamp-Monzón explicitly leave it open. Our Prop. esym (e_nu(x) >= 1) proves it, as a byproduct, for exact decompositions whose factors have unit modulus, and every rank-1 term can be normalised that way. **This is worth stating.** Please double-check the tensor literature (Hackbusch's book, Wang-Seigal 2023) before claiming it as new; I could not read them in full.
- Any l_r (r<1) quasi-norm lower bound on the coefficients of a CP/separated decomposition of this tensor, with an equality characterisation (equal weights, orthogonal chain vectors, i.e. rotated roots of unity). Searches: sine-of-sum separation rank lower bound; real rank of the 2x...x2 product tensor; polyphase/roots-of-unity cancellation with l_p weights; det(I+tA) plus Bernstein function majorization. None gave a hit.

## Recommended change
After Lemma cost (cyclic.tex), add: "In tensor language, (JF) asks for an exact real decomposition of the tensor $\mathrm{Re}(z_1\cdots z_n\bar z_0)$ (the function $\cos(\theta_1+\dots+\theta_n-\theta_0)$), which has complex rank $2$. The $\nu$-term decomposition is the `sine of a sum' identity of \citet{beylkin2005algorithms,mohlenkamp2005trigonometric}, who conjectured it minimal; the symmetric analogue, real Waring rank $d$ for $\mathrm{Re}((x+iy)^d)$, is \citet{comon2012typical}. Lemma~\ref{lem:cost} gives the lower bound (at least $\nu$ terms) and strengthens it to a weighted $\ell_r$ cost, with equality exactly at equally weighted, orthogonal (roots-of-unity) decompositions." Keep "cost of a feature" as our own result.

## Bib
@article{beylkin2005algorithms, title={Algorithms for Numerical Analysis in High Dimensions}, author={Beylkin, Gregory and Mohlenkamp, Martin J.}, journal={SIAM Journal on Scientific Computing}, volume={26}, number={6}, pages={2133--2159}, year={2005}, doi={10.1137/040604959}}
@article{mohlenkamp2005trigonometric, title={Trigonometric Identities and Sums of Separable Functions}, author={Mohlenkamp, Martin J. and Monz{\'o}n, Lucas}, journal={The Mathematical Intelligencer}, volume={27}, number={2}, pages={65--69}, year={2005}, doi={10.1007/BF02985795}}
@article{comon2012typical, title={On the typical rank of real binary forms}, author={Comon, Pierre and Ottaviani, Giorgio}, journal={Linear and Multilinear Algebra}, volume={60}, number={6}, pages={657--667}, year={2012}, doi={10.1080/03081087.2011.624097}}
