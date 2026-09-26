# VERIFY (citation lens) — F6-where-dense-stops, task 70

Claim: Morwani find the best individual neurons, then combine them. Combining is free exactly at L_{2,nu}. For b<nu, concentrated output amplitude is cheaper, so each feature has a price.

Location: the intro paragraph "Where the dense theorem stops" is no longer in sections/intro.tex (the intro is down to 2 paragraphs). The same content now sits in approach.tex l.54-73 (single-neuron pricing "as in Morwani"; subadditivity) and cyclic.tex l.30 and l.52. Apply the recommendations wherever the sentence ends up.

## Citations fetched this session
1. morwani2024feature. arXiv abs 2311.07568: title, the 5 authors in order, and "Spotlight at ICLR 2024" all match refs.bib. arXiv HTML v2:
   - Sec 3.3 "Blueprint for the case studies" confirmed.
   - Lemma 3 (binary) second bullet, "Combining neurons": "If b=ν ... for any neuron scaling factors Σλ_i^ν=1 ... θ={λ_i ω_i*} belongs to Θ_q*". Lemma 5 is the multi-class version.
   - App. E.1 (Binary Classification) contains the remark that for b≠ν "we would not get the same flexibility to build a θ* satisfying Equation [1]".
   - Caveat: Lemma 3 gives only the "if" direction (b=ν is sufficient). "Exactly at L_{2,ν}" also needs the only-if direction. That comes from Morwani's informal E.1 remark plus this paper's own b<ν results (cyclic.tex), not from a Morwani theorem. Wording: "their combining lemma requires b=ν".
2. hoff2017lasso. arXiv 1611.00040 abstract: "a subclass of L_q penalties with q≤1 can be expressed as sums of L_2 penalties" via a Hadamard-product parametrisation. Crossref confirms CSDA vol 115, pp 186-198, 2017, DOI 10.1016/j.csda.2017.06.007. Bib is correct; add the DOI.
3. kolb2025deep. arXiv 2502.02496: title and author order (Kolb, Weber, Bischl, Rügamer) match; comments say "accepted at ICLR 2025". The l_{2/D} equivalence is Theorem 1 in the BODY: min L(w)+λ||w||_{2/D}^{2/D} has the same global and local minima as min L(ω_1⊙...⊙ω_D)+(λ/D)Σ||ω_d||². The abstract says only "non-convex sparse regularization". So the checker's "abstract says l_{2/D}" is imprecise, but the substance holds. Kolb credits Hoff 2017 as the origin.
4. dai2021representation. NeurIPS 2021 proceedings page: title, authors (Dai, Karzand, Srebro) and NeurIPS 34 confirmed. PDF: Table 1 and Sec 3.1.2 give the depth-d diagonal network's representation cost as R(β)=d||β||_{2/d}^{2/d} (l_{2/d} quasi-norm). Table 1 also lists "Convolutional Network: matrix l_{2/d,2} quasi-norm on Fourier domain". Bib is correct.

## Missed, more canonical and closer precedent
- Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", NeurIPS 31 (2018), arXiv 1806.00468. Fetched: arXiv abs and papers.nips.cc 2018 hash 0e98aeeb54acf612b9eb4e48a269814c. Abstract: GD on depth-L linear conv nets "converges to a linear predictor related to the ℓ_{2/L} bridge penalty in the frequency domain" (max-margin setting), in contrast to the l2 SVM for fully connected nets.
- Why it matters: this is the closest known instance of "margin + l2 on multiplicative factors gives a concave cost on Fourier amplitudes, which gives frequency sparsity". It should be cited next to Hoff/Kolb. The novelty claim must be scoped to nonlinear product/power networks on modular addition and groups: the feature count, the budget-splitting programme, and the log p scaling. The claim that "Fourier features acquire a price under margin" in general is not new.

## Verdict
The checker's SUPPORTED verdict stands. All four confirmed citations exist, have correct bib fields and say what is claimed, with two caveats: the Kolb content is in Theorem 1, not the abstract, and Morwani gives only the sufficiency direction. Add Gunasekar et al. 2018.

Operational note: this session's WebSearch budget is exhausted (200/200), so this verification used WebFetch plus the Crossref and arXiv APIs only.
