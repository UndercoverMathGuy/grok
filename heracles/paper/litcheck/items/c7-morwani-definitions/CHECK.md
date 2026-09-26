# C7-morwani-definitions: CHECK

Source: Morwani, Edelman, Oncescu, Zhao, Kakade, "Feature emergence via margin maximization: case studies in algebraic tasks", ICLR 2024, arXiv:2311.07568. I fetched the PDF from https://arxiv.org/pdf/2311.07568 and extracted it with pdftotext.

Verdict: SUPPORTED. Only small precision fixes are needed.

## Comparison, line by line
- prelim.tex: "the L_{2,b} norm ... b=nu is the norm of Morwani".
  - Their Sec. 2 defines ||theta||_{a,b} = (sum_i ||omega_i||_a^b)^{1/b}.
  - Sec. 3.2 ends: "we will primarily focus on maximum margin with respect to L2,ν norm in this work".
  - The homogeneity definition is the same, phi(lambda omega, x) = lambda^nu phi(omega, x) (Sec. 2).
  - Result: MATCH.
- prelim.tex: "at n=2 the [power neuron] is the quadratic neuron of Morwani".
  - Their Sec. 2: phi({u,v,w},a,b) = (u_a + v_b)^2 w, with one-hot inputs, no biases, and w in R^{|G|}. This is our power neuron with u_1=u, u_2=v and nu=3.
  - Result: MATCH.
- approach.tex: g, h and gamma*.
  - Their Sec. 3: g(θ,x,y)=f(θ,x)[y] − max_{y'≠y} f(θ,x)[y']; h(θ)=min_{(x,y)∈D} g; γ* = max_{θ∈Θ} h(θ) with Θ={||θ||≤1}.
  - They define the normalized margin as h(θ/||θ||). Ours is h(θ)/||θ||^ν. The two are equal by ν-homogeneity. MATCH, with a harmless difference in notation.
- approach.tex: Wei Thm 4.1.
  - Morwani restate it as their Theorem 1: "For any norm, a fixed r>0 and any homogeneous f ... lim γ_λ = γ*".
  - Our λ||θ||_{2,b}^b is the case r=b. Consistent.
  - The fact that Wei's Thm 4.1 is binary and Thm C.1 covers the multi-class case is handled in a separate item.
- approach.tex: "averaged class-weighted margin [Morwani, Section 3]".
  - This is Sec. 3.2 (Multi-Class Classification): g' ≥ g, with class weights τ. More precisely, it is Sec. 3.2 and Lemma 5/6.
- approach.tex: "analogue of η_{u,v,w}(δ)".
  - This is defined only in Appendix F.1 (proof of Thm 7): η_{u,v,w}(δ) := E_{a,b}[(u(a)+v(b))^2 w(a+b−δ)]. It does not appear in the main text, so the pointer should say App. F.1.
- approach.tex: "phase condition of [Morwani, Section 4]".
  - Thm 7 (Sec. 4), part 1, gives θ_u*+θ_v*=θ_w*. App. F.1 Eq. (7) makes the cos(θ_u+θ_v−θ_w) term maximal.
  - Result: MATCH.
- approach.tex / appendix: "exactly their maximum L_{2,3}-margin sqrt(2/27)/(p^{1/2}(p−1))".
  - Thm 7: "with m ≥ 4(p−1) neurons ... γ* = sqrt(2/27)·1/(p^{1/2}(p−1))", for prime p>2.
  - Result: MATCH. Their value needs width m ≥ 4(p−1). Our bound is width-free, so "exactly" holds at that width.
- intro.tex: "one frequency per neuron, every frequency used". These are Thm 7 parts 1 and 2. MATCH.

## Suggested fixes (optional)
1. approach.tex l.11: change "the quantity η_{u,v,w}(δ) of \citet{morwani2024feature}" to "... of \citet[App.~F.1]{morwani2024feature}".
2. approach.tex l.7: change "\citep[Section~3]" to "\citep[Section~3.2]".
3. approach.tex l.70: optionally add "(attained for width m ≥ 4(p−1), their Theorem 7)".
4. approach.tex l.3: optionally note that their normalized margin h(θ/||θ||) equals h(θ)/||θ||^ν by homogeneity. The text already says "by homogeneity".
