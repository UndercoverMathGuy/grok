# T3-determinant-gram (appendix app:cost, proof of Lemma cost)

Verdict: KNOWN (components). The assembled inequality has a two-line classical proof.

## What the appendix does
Setup: H = sum_l d_l u_l u_l^* with unit vectors u_l. H is upper triangular with unit diagonal, so det(I+tH) = (1+t)^nu.
1. Sylvester, det(I+UΔU*) = det(I+ΔU*U), plus the principal-minor expansion, gives e_k(|d|) >= C(nu,k), using Hadamard's bound 0 <= Gram det <= 1.
2. So prod(1+x_l t) >= (1+t)^nu for every t >= 0.
3. The Bernstein/Stieltjes integral x^r = (r sin(pi r)/pi) ∫ log(1+xt) t^{-1-r} dt turns this into sum x_l^r >= nu for 0 < r < 1.

## Standard pieces (textbook)
- Sylvester's determinant identity, the principal-minor expansion of det(I+tX), and the Hadamard inequality for PSD/Gram matrices are all in Horn & Johnson, Matrix Analysis, 2nd ed. (CUP 2012).
  - Confirmed: https://www.cambridge.org/core/books/matrix-analysis/9CF2CB491C9E97948B15FAD835EF9A8B
  - I did not re-verify exact section numbers. From memory: Hadamard is Thm 7.8.1, the det(I+tA) coefficients are around 1.2.x, Sylvester is 0.8.x (unverified).
- The integral representation of x^r (0<r<1) is the classical Stieltjes/complete-Bernstein representation. Bhatia Ch. V (operator monotone functions) gives x^r = (sin rπ/π) ∫ x/(x+λ) λ^{r-1} dλ. The log form is that representation integrated by parts. Section and equation numbers are unverified.
  - Schilling-Song-Vondraček, Bernstein Functions (de Gruyter) is the monograph reference. Unconfirmed: the page fetch failed.
- Transfer from generating function to power sums: I found no named precedent in the majorization literature. Ordering by sum log(1+tx) for all t is not ordinary (weak) majorization. It controls exactly the cone of complete Bernstein functions, and x^r lies in that cone. The step is standard in spirit and elementary. It does not need a citation beyond the integral formula.

## Main finding: the whole lemma inequality is classical
sum_l |beta_l|^r >= nu^{1-r} |beta|^r for 0<r<=1 under the chain condition. This follows directly from two classical theorems. I checked the argument by hand.
- Weyl's majorant theorem: Weyl, PNAS 35 (1949) 408-411, https://www.pnas.org/doi/10.1073/pnas.35.7.408. Also Bhatia Thm II.3.6, as quoted by secondary sources.
  - Statement: if phi(e^s) is convex increasing, then sum phi(|lambda_i(H)|) <= sum phi(sigma_i(H)).
  - With phi = x^r and every eigenvalue equal to 1, this gives nu <= sum sigma_i(H)^r.
- Rotfel'd/Thompson subadditivity: R.C. Thompson, Pacific J. Math. 66(1) (1976) 285-290. I fetched the PDF. Its eq. (1) reads: "f(γ1)+...+f(γn) <= f(α1)+...+f(αn)+f(β1)+...+f(βn) when f is an increasing concave function of a nonnegative real variable, with f(0)=0", where α, β, γ are the singular values of A, B, C = A+B.
  - Apply it to f = x^r across the N rank-one summands d_l u_l u_l^*. Each has the single singular value |d_l|. This gives sum sigma_i(H)^r <= sum_l |d_l|^r.
- Combining the two: sum_l |d_l|^r >= nu, which is the lemma.

The paper's route (Gram determinants, then the Bernstein integral) is correct and self-contained. It also gives the equality case (exactly nu equal moduli, orthogonal chain vectors) cleanly. The alternative proof needs extra work for equality.

## Recommendation
Keep the proof. Add one sentence after the lemma saying the inequality part also follows from Weyl's majorant theorem combined with the Rotfel'd-Thompson concave subadditivity of singular values. Cite Horn-Johnson for Sylvester, the minor expansion and Hadamard. Do not claim the determinant technique as a novel tool.

Searches run: Thompson 1976 title; Rotfel'd concave singular values; Weyl majorant Bhatia II.3.6; Weyl 1949 PNAS; the Bernstein integral representation of x^r; Schilling-Song-Vondracek.
