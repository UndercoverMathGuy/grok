# VERIFY (novelty_math): R9 interference-free definition (cyclic.tex def:if)

Verdict: the checker's PARTIALLY_KNOWN stands, and all six confirmed citations check out. The checker missed one mathematics precedent: Beylkin-Mohlenkamp 2002.

## Process note (relevant to workflow speed)
- WebSearch was unavailable: the session had used its full budget (200 of 200 calls).
- The arXiv export API (export.arxiv.org) and the Semantic Scholar API both returned HTTP 429 (rate limited).
- What worked instead: fetching arxiv.org/pdf directly, plus the Crossref and OpenAlex APIs.

## Checker's confirmed citations (checked against the PDF text)
- morwani2024feature (2311.07568v2). App. F.2 has phases (0,0,0) and (0,pi,pi), and says "value on all incorrect labels c', satisfying C.1" (pdf line ~2312). The sparse-parity section says "every term other than Pi x_{S_i} cancels out in the sum" (line 2581). The bib entry is correct.
- li2025nsum (2402.09469v4). Their Condition 3.8 is Morwani's C.1. They use 2^{2k-1} neurons per frequency: "each term can be constructed by 2^{k-1} neurons ... 2^{k-1} 2^k total". For k=3 they write "identical result across all incorrect labels c', adhering to Condition 3.8". The bib entry is correct.
- gromov2023grokking (2301.02679v1). Contains "constructive interference", "all other waves ... interfere destructively" and "destructive interference condition (15)". The bib entry is correct.
- doshi2024polynomials (2406.03495v1, ICLR 2024 Workshop BGPT header). Contains "all other terms will vanish due to random phase approximation". The bibtex is correct.
- he2026mechanism (2602.16849v1; He, Wang, Chen, Yang). Def. 4.1 (Full Diversification), item (iii): sum over m in N_k of exp(i*iota*phi_m) = 0 for iota in {2,4}. Prop. 4.2 says spurious peaks at 2x and 2y survive. The bibtex is correct.
- zhong2023clock (2306.17844v2). Pizza: H_ab = |cos(w_k(a-b)/2)| (cos w_k(a+b), sin w_k(a+b)). The bibtex is correct (NeurIPS 2023).

## Mathematics lens: what the definition is, in math terms
Consider the neurons at frequency j. Each neuron is a real product of one sinusoid per slot, cos(theta + alpha); this is a real rank-one tensor in span{cos, sin}^{(x)nu}. Interference-free at j then means exactly this: the neurons' frequency-j parts sum to an exact real separated (CP) representation of cos(2 pi j (a_1+...+a_n - c)/p). The minimal number of neurons per used frequency is therefore the real separation rank of cos(x_1+...+x_nu). The naive angle-sum expansion needs 2^{nu-1} terms, which is the regime of Li et al.'s construction.

### Missed precedent (page fetched and text read)
- Beylkin and Mohlenkamp, "Numerical operator calculus in higher dimensions", PNAS 99(16):10246-10251 (2002), doi:10.1073/pnas.112329799, PMC124898.
- Lemma 1: "a sine wave in the diagonal direction, sin(x_1+...+x_d) ... the usual trigonometric formulas for sums of angles ... r = 2^{d-1} terms. The numerical algorithm ... uncovered a trigonometric identity in d dimensions, using exactly d terms", valid for all alpha_j with sin(alpha_k - alpha_j) != 0.
- It also notes that there are "entire families" of optimal separated representations.
- The nu-block (Prop. prop:block) is an identity of the same kind: nu equal-weight shifted-cosine products that give cos of a sum of nu variables. I checked it numerically for n = 2, 3, 4.
- The nu-versus-2^{nu-1} contrast is exactly Beylkin-Mohlenkamp's d-versus-2^{d-1}.
- So the count nu is not new as a trigonometric-identity fact. What the paper adds: the l_r cost lemma (a weighted bound, not just a count), the equality case, and the margin optimisation over the class.

### Found but not read (metadata only; do NOT cite without reading)
- Mohlenkamp and Monzon, "Trigonometric identities and sums of separable functions", Math. Intelligencer 27(2):65-69 (2005), doi:10.1007/BF02985795. Metadata verified via Crossref and OpenAlex. The text is paywalled (Springer); no open-access copy found.
- This is the dedicated paper on these identities. It may state whether d terms is minimal over the reals for sin(x_1+...+x_d); I could not check. If it does, that is the count (r -> 0) shadow of Lemma lem:cost.
- The authors should read it before submission.

### Observation (no citation fetched)
The equality case of Lemma lem:cost asks for nu pairwise orthogonal vectors in C^nu with unimodular entries (the chain vectors m_l). Those are the rows of a complex Hadamard matrix, and the nu-block uses the Fourier matrix. Using the standard name could help readers.

## Recommendation
- Keep the checker's sentence after def:if.
- Add at prop:block (or after lem:cost): "The $\nu$-block is an instance of the exact $d$-term separated representations of a sinusoid of a sum of $d$ variables \citep{beylkin2002numerical}, against $2^{d-1}$ terms from the angle-sum formulas (cf. the $2^{2n-1}$ neurons per frequency of \citealp{li2025nsum}); Lemma~\ref{lem:cost} adds that no interference-free population is cheaper in any $\ell_r$ sense."
- Novelty claim: do not claim that the nu-term cancellation is new. Claim the class, the sign rule, the (JF) system, the l_r cost lemma with its equality case, and the class-level margin theorem.
