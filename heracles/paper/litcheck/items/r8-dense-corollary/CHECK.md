# R8-dense-corollary: Cor. cor:dense (approach.tex l.69-70; appendix app:dense)

Claim: Gamma_1(p) = p/(p-1), unique flat maximizer y_j = 2/(p-1); at b=nu recovers Morwani's max L_{2,3} margin sqrt(2/27)/(p^{1/2}(p-1)).

## Verdict: KNOWN (endpoint constant), PARTIALLY_KNOWN overall
The corollary's content consists of two known results rewritten in our notation. Frame it as a consistency check, not as a new result.

### Constant: verified exactly
- Morwani et al. ICLR 2024 (spotlight), arXiv 2311.07568v2, **Theorem 7** (Sec. 4, "Cyclic groups"):
  "Consider one-hidden layer networks f(theta,a,b) of the form given in section 2 with m >= 4(p-1) neurons. The maximum L2,3-margin of such a network on the dataset D_p is gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1))."
  Neuron: (u_a + v_b)^2 w, i.e. our power neuron with n=2, nu=3. Parts 1-2: every neuron is single-frequency, and "For every frequency zeta ... at least one neuron in the network uses this frequency."
- Li, Liang, Shi, Song, Zhou, AISTATS 2025 (PMLR 258:523-531), arXiv 2402.09469v4, **Theorem 4.1** (informal version of Thm G.2):
  neurons (u_1(a_1)+...+u_k(a_k))^k w; L_{2,k+1}; m >= 2^{2k-1}(p-1)/2; gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}); "For each frequency zeta ... there exists one neuron using this frequency only."
- Our b=nu value n! B_p Gamma_1(p), with B_p = 2(2p nu)^{-nu/2} and nu = n+1, simplifies algebraically to Li's gamma*. I checked it numerically for n = 2, 3, 4 and p = 7, 113: the ratio is 1 to 1e-15. At n=2 it equals Morwani's constant.
- **So the general-nu dense endpoint belongs to Li et al.**, and the corollary should cite Li Thm 4.1 as well as Morwani Thm 7.

### Flat maximizer / "equal amplitude on every frequency": implicit in Morwani
- Morwani App. F.3 ("Proof that all frequencies are used") uses Lemma 6 to show that every max-margin f splits as f1 = F(a,b), plus f2 = lambda * 1[a+b=c]. It follows that f-hat(j,j,-j) = lambda p^2 at every nonzero j. That is the same diagonal Fourier amplitude at every frequency, which is our flat y. Morwani stated only the weaker "at least one neuron per frequency", but equal amplitude follows directly from their proof. Li's construction (Lemma 4.3 / E.3) is also flat.
- Our averaging bound min_delta <= (1/(p-1)) sum_{delta != 0} is the coverage-space version of Morwani's uniform class weighting tau[c'] = 1/(p-1) (Sec. 4 proof outline). Li uses the same weighting, Eq. (5).
- What is ours: the explicit uniqueness statement for the programme Gamma_1 (via the Fourier inversion of zeta), and the fact that the endpoint drops out as the s=1 case of a programme that is sparse for s>1. The uniqueness proof is elementary.
- Concurrent contrast: Kam, Bessafi, Cadet arXiv 2609.17926, Prop 5. They also find uniform allocation optimal, but with an l2 amplitude budget. Our s=1 flat result sits under an l1 budget, which is where density appears in our family. See LIT_ml.md.

### Internal wording issue (not a novelty issue)
appendix app:models (power neuron) says "At n=2, b=3 the bound is attained by the network of Morwani (Corollary dense), so C_3 = 2B_p". Meanwhile app:dense uses C_3 = 2B_p to get the constant. The argument isn't actually circular, because C_3 <= 2B_p is proved in app:models and Morwani Thm 7 supplies attainment. But the text reads as if it were. Fix: say "C_3 <= 2B_p (App. models); equality holds because Morwani's Theorem 7 network attains the bound".
The product-network attainment (Prop block at y* flat) is ours. The power-network attainment at b=nu comes from Morwani/Li, not from our construction.

## Searches run
- Local PDF text of arXiv 2311.07568v2 and 2402.09469v4 (pdftotext): Thm 7, App. F.2/F.3, Sec. 2 norm definition; Li Thm 4.1, Lemma 4.2/4.3, Sec. 3 norm definition.
- WebSearch "maximize minimum over nonzero x of sum a_k(1-cos(2 pi k x/p)) subject to sum a_k=1 uniform optimal": no harmonic-analysis source. The l1 version is the trivial averaging bound.
- WebSearch "max margin modular addition all frequencies equal amplitude p/(p-1) ...": found only Morwani, McCracken, Kam, He et al., and Huang-Li 2511.23443 (sine activations, margin-based generalization, not the L_{2,nu} optimum). All are related only.
- Venue pages fetched: iclr.cc/virtual/2024/poster/18073 (ICLR 2024 spotlight), proceedings.mlr.press/v258/li25b.html (AISTATS 2025, PMLR 258:523-531).
