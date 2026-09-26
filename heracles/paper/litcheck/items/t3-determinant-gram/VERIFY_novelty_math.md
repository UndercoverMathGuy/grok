# VERIFY (novelty_math): T3-determinant-gram, Lemma cost / app:cost

Verifier verdict: **KNOWN, and more than the checker said.** Both the inequality and each step of the technique are classical:
- Step 1, the determinant/Gram comparison, is the Seiler–Simon determinant inequality (1975).
- Step 2, integrating the logarithm of a generating-function comparison to get power sums, is Weyl's own 1949 proof device.

The checker's verdict stands, but one confirmed citation has a wrong bib field (Horn–Johnson DOI), so `upheld = false`.

## 1. Step 1 is Seiler–Simon (1975)
Source: E. Seiler, B. Simon, "An inequality among determinants", PNAS 72(9):3277–3278, 1975, doi:10.1073/pnas.72.9.3277. I read page images of both pages via PMC432972.
- Thm 1: det(1+|A+B|) <= det(1+|A|) det(1+|B|).
- Thm 1'': |det(1+A+B)| <= det(1+|A|) det(1+|B|).
- Thm 2' (coefficientwise): |d_m(A+B)| <= sum_k d_k(|A|) d_{m-k}(|B|). Here d_m is the m-th elementary symmetric function of the eigenvalues.
- Their proof method: "expanding det(1+μA+λB) and det(1+μ|A|)det(1+λ|B|) as polynomials ... and prove the coefficients of one dominate those of the other" (exterior algebra). The paper's principal-minor/Gram expansion does the same thing.

How it applies: take A_l = t d_l u_l u_l^* (rank one, |A_l| has the single eigenvalue t|d_l|). Apply Thm 1'' once, then Thm 1 inductively. This gives |det(I+tH)| <= prod_l (1+t|d_l|), which is exactly eq. (cost-generating). Iterating Thm 2' gives |e_k(λ(H))| <= e_k(|d|), which is the paper's e_k(x) >= C(ν,k).

Numerical sanity check: 2000 random instances, max ratio 0.999999.

## 2. Step 2 is Weyl's device (1949)
Source: H. Weyl, PNAS 35(7):408–411, 1949. I read page images pp. 408–411 via PMC1063049.
- The theorem: for φ increasing with φ(e^ξ) convex and φ(0)=0, sum φ(λ_i) <= sum φ(κ_i), where λ_i = |α_i|^2 and κ_i are the eigenvalues of A*A. In particular sum λ^s <= sum κ^s for all s>0.
- The proof (eqs. 8–10) first shows a generating-function comparison: g(z) = prod max(1, λ_i z) <= f(z) = prod max(1, κ_i z) for z >= 0. It then integrates: φ(λ) = ∫_0^∞ log max(1, λz) dψ(z), so sum φ(λ_i) = ∫ log g dψ.
- This is exactly the paper's move, with the kernel log max(1,xz) replaced by log(1+xt).
- On p. 410–411 Weyl also notes that prod(1+λ_i^s z) is majorized coefficient for coefficient by prod(1+κ_i^s z).

The kernel log(1+xt) is the Thorin–Bernstein kernel. After t -> 1/t, the class in Remark "other penalties", ψ(x) = cx + ∫ log(1+xt) μ(dt), is precisely the Thorin–Bernstein functions with ψ(0)=0. Source: Schilling–Song–Vondraček, *Bernstein Functions: Theory and Applications*, 2nd ed., De Gruyter 2012, doi:10.1515/9783110269338. Crossref confirms Ch. 8 is "Thorin–Bernstein functions", pp. 109–116. I could not fetch the chapter text, so the theorem number is not verified.

The constant x^r = (r sin πr/π) ∫_0^∞ log(1+xt) t^{-1-r} dt checks numerically to 1e-12 (r in {0.3, 0.5, 2/3, 0.9}). The paper only needs 0 < J_r(1) < ∞, which it proves.

## 3. The inequality itself: the checker's two-theorem proof is correct
- **Weyl 1949 (verified above):** H is triangular with unit diagonal, so sum σ_i(H)^r >= sum |λ_i|^r = ν.
- **Thompson 1976, eq. (1) (Rotfel'd):** I downloaded the msp.org PDF (pjm-v66-n1-p30) and read it. It says: "f(γ1)+…+f(γn) <= f(α1)+…+f(αn)+f(β1)+…+f(βn) when f is an increasing concave function of a nonnegative real variable, with f(0)=0". Iterating over the rank-one summands gives sum σ_i(H)^r <= sum |d_l|^r.
- **Link to Step 1:** Rotfel'd with f = log(1+tx) is Seiler–Simon Thm 1. So the paper's route is the classical route restricted to the generating kernel, then integrated.

## 4. Equality case
The equality statement (exactly ν non-zero coefficients of equal modulus, with orthogonal chain vectors) is the part the paper actually works out itself. The classical route needs extra work here: equality in Weyl forces |λ_i| = σ_i, hence H is normal, hence H = βI, and then Rotfel'd equality still has to be handled.

The orthonormal unimodular chain vectors m_l/√ν are the rows of a (dephased) complex Hadamard matrix of order ν. The ν-block is the Fourier matrix. Reference: Tadej & Życzkowski, "A concise guide to complex Hadamard matrices", arXiv quant-ph/0512154 (fetched), OSID 13:133–177 (2006). This could be one clause in Remark "orthogonality as roots of unity". It is optional.

## 5. Checking the checker's confirmed citations
| key | status |
|---|---|
| hornjohnson2012 | **Bib field wrong.** DOI 10.1017/CBO9780511810817 is the **1st edition (1985)**. I verified this on the cambridge.org page the checker cited, which shows the 1985 print year, and in Crossref (edition 1). The 2nd edition (2012) DOI is **10.1017/CBO9781139020411** (Crossref: edition 2, published 2012-10-22). Also a caution: in HJ, "Sylvester's (determinant) identity" usually names the bordered-minor identity, not det(I+AB)=det(I+BA). The paper should state the identity explicitly rather than rely on the name. Section numbers are unverified. |
| thompson1976 | Confirmed. PDF read; Pacific J. Math. 66(1):285–290, doi 10.2140/pjm.1976.66.285 (Crossref). Eq. (1) is quoted correctly and attributed to Rotfel'd. |
| weyl1949 | Confirmed. Page images read; PNAS 35(7):408–411, doi 10.1073/pnas.35.7.408. Weyl's statement uses squared quantities (|α|^2 against the eigenvalues of A*A); it is equivalent to the checker's paraphrase. |
| bhatia1997 | Metadata confirmed via Crossref: Springer New York, GTM, 1997, doi 10.1007/978-1-4612-0653-8. The checker's URL was Amazon; use the DOI. The locators "Thm II.3.6" and "Ch. V integral representation" are **unverified** (Springer page blocked). Do not print theorem numbers unless someone checks the book. |

## 6. Recommended change
- Fix the Horn–Johnson DOI.
- In app:cost, attribute the technique rather than only the pieces, for example: "Step 1 is the determinant inequality of Seiler and Simon [seiler1975] applied to rank-one summands (their Thm 2' gives the coefficientwise form). Step 2 is Weyl's device [weyl1949] of integrating the logarithm of a generating-function comparison, here with the Thorin–Bernstein kernel log(1+xt) [schilling2012, Ch. 8]. The inequality also follows from Weyl's majorant theorem with Rotfel'd–Thompson [thompson1976]. The equality case is what we need."
- In Remark "other penalties", name the class: Thorin–Bernstein functions.

Searches: my WebSearch budget was already exhausted, so I did not run any web searches. I used direct fetches only: cambridge.org and doi.org, Crossref lookups by DOI and ISBN, the msp.org Thompson PDF, PMC page images for Weyl 1949 and Seiler–Simon 1975, and arXiv quant-ph/0512154. Bhatia's Springer page and the de Gruyter pages were blocked by bot challenges.
