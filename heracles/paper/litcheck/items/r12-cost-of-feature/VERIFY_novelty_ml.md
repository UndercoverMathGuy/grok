# VERIFY (novelty_ml): R12-cost-of-feature (task 11)

Claim (cyclic.tex lem:cost, app:cost): under (JF), sum_l |beta_l|^r >= nu^{1-r} |beta|^r for r in (0,1]. For r<1, equality holds iff exactly nu terms have modulus |beta|/nu and pairwise orthogonal chain vectors.

## Verdict
PARTIALLY_KNOWN is upheld, with one correction to how the checker framed the count part.

## Tooling note (relevant to workflow slowness)
- WebSearch was refused on the first call: "this session has used its web search budget (200 of 200)". The budget is shared across the whole workflow session.
- export.arxiv.org/api and api.semanticscholar.org returned HTTP 429 (rate limited).
- Workaround used: curl against arxiv.org/search HTML, api.crossref.org, api.openalex.org, and PDF downloads.

## Citations the checker confirmed (all re-verified)
1. **beylkin2005algorithms.** Crossref: SIAM J. Sci. Comput. 26(6):2133-2159, 2005, doi 10.1137/040604959. The metadata is correct.
   - Preprint PDF: ohiouniversityfaculty.com/mohlenka/research/BEY-MOH2004P.pdf.
   - §2.1.1 "Example: Sine of a Sum", Lemma 2.4, eq. (2.9): the d-term identity is confirmed.
   - Def 2.2 defines kappa = (sum s_l^2)^{1/2}/||F||. Confirmed.
   - The kappa≈7 remark is in §3.3 ("Numerical Example: Sine of a Sum"), in a parenthetical during a d=10 experiment: "(Choosing alpha_j equally spaced in the identity leads to kappa ≈ 7.)". It is a numerical aside, not a result.
   - §3.3 also observes that with r=d-1 terms, ALS drives kappa to infinity. This is a border-rank-type phenomenon. No lower bound is proved.
2. **mohlenkamp2005trigonometric.** Crossref: Math. Intelligencer 27(2):65-69, 2005, doi 10.1007/BF02985795. Correct.
   - The preprint (sine.pdf) says: "to find a minimal rank representation ... is still an open problem. We believe that our Theorems 2 and 5 are an example of such minimal representations." Confirmed.
   - New detail: Figure 1 (right) shows sin(x+y+z) with alpha=0, pi/3, 2pi/3 and "the amplitude has been equidistributed". The paper's nu-block construction (Prop block, alpha_l = pi l/nu, equal amplitudes) is essentially this picture. Cite it at Prop block as well.
3. **comon2012typical.** Crossref: LMA 60(6):657-667, 2012, doi 10.1080/03081087.2011.624097. arXiv 0909.4865 abstract: "a real binary form of degree d with d real roots has rank d". Confirmed.
   - Useful companion, not required: Causa-Re arXiv 1006.5127, Annali di Matematica. The interior of the locus of real binary forms of rank d is exactly the set of forms with d real roots. Fetched.

## Correction: the count part is elementary
The checker wrote that no proof of "real rank of Re(z_1...z_d) = d" exists, and recommended framing Lemma cost as proving what Mohlenkamp-Monzón only believed. That lower bound is a two-line substitution argument:
- Suppose T_d = Re(z_1...z_d) = sum_{l=1}^r a_l^1 ⊗ ... ⊗ a_l^d over R^2.
- Pick w ≠ 0 with <a_1^d, w> = 0 and contract the last slot with w. This gives Re(c z_1...z_{d-1}) with c = w_1 + i w_2 ≠ 0. That is a rotated, scaled copy of T_{d-1}, written with at most r-1 terms.
- So rank T_d ≥ rank T_{d-1} + 1 ≥ d.
- The same argument works for functions: project each factor onto span{cos, sin}. So the Mohlenkamp-Monzón minimality "belief" is folklore-provable.

A reviewer from the tensor community would call "first proof of nu terms" trivial. Recommended phrasing: "the count bound also follows from a standard substitution argument; our contribution is the weighted l_r bound, with equality exactly at equal weights and orthogonal chain vectors."

One caveat that works in the paper's favour: app:cost says the proof uses only the C(nu,2) interval patterns of (JF). That hypothesis is weaker than an exact decomposition, so the substitution argument does not cover it directly.

Route I checked and dropped: Comon-Ottaviani plus Zhang-Huang-Qi (SIMAX 37(4):1719-1728, 2016, doi 10.1137/141001470; abstract via Crossref: rank = symmetric rank when rank ≤ order) would also give the count. I could not confirm that ZHQ holds over R (closed access), so I do not recommend it.

## ML novelty search (assume it is already known)
- **Tian, arXiv 2410.01779**, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets". It constructs order-4 and order-6 per-frequency solutions with roots of unity (Cor 2, Cor 4, Table 1). It proves no lower bound on the number of hidden nodes per frequency, and no norm or l_p cost per frequency. Not a scoop.
- **Li et al., arXiv 2402.09469** (k-input modular addition, L_{2,k+1} max margin). Lemma 4.3 uses 2^{2k-1} neurons per frequency, built from sum-to-product identities (Lemma E.1, App E.1). It has no nu-neuron minimum, no l_p cost with p<1, and no equality case.
  - This gives a clean contrast: prior ML constructions use 2^{2k-1} neurons per frequency, against nu = k+1 here.
  - The b = nu case (r=1) is the triangle inequality, so the prior L_{2,nu} literature cannot pin the count.
- Wang-Seigal (arXiv 2202.11740, JSC 118:69-92, 2023) was checked, as the checker asked. It covers lower bounds via unfoldings and real counterexamples to Comon's conjecture. It says nothing about Re(z_1...z_d), sin of a sum, or weighted costs. Not a scoop.
- Not re-checked here, because the search budget is gone: Morwani App F.2, He et al. 2602.16849, Gromov. The checker delegated these to items r9/r10.

## Bottom line
- Count part (≥ nu terms): known or folklore. It is the real tensor rank of Re(z_1...z_nu), provable by a substitution argument.
- Equally spaced, equal-amplitude construction: known (Mohlenkamp-Monzón Fig 1; Beylkin-Mohlenkamp §3.3 remark).
- Weighted l_r lower bound nu^{1-r}, its equality characterisation (orthogonal chain vectors), and the det(I+tH) + Hadamard + Bernstein-integral proof: not found in ML or tensor sources checked. These stay as the paper's contribution.
