# T12-dense-uniqueness: flat maximizer unique via Fourier inversion of constant coverage (appendix app:dense)

Verdict: PARTIALLY_KNOWN. The technique is Morwani et al.'s. Our sharpening to *equal* amplitudes is a small step they left out.

## Our argument (sections/appendix.tex, "Proof of Corollary cor:dense")
1. Upper bound: average the coverage over the p-1 values delta != 0, then use sum_{delta != 0} cov_j(delta) = p. This gives Gamma_1 <= p/(p-1).
2. Equality forces the coverage to be constant in delta. So zeta = c0 + (1-c0)[delta=0]. Fourier inversion then gives y_j = 2/(p-1) for every j, which is the unique maximizer.

## Morwani, Edelman, Oncescu, Zhao, Kakade (ICLR 2024 spotlight, arXiv 2311.07568v2). I fetched and read the HTML.
- Step 1 is their uniform class-weighting. Sec 4 proof outline: "We will use the uniform class weighting: tau(x,y)[y'] = 1/(p-1) for all c' != a+b". Lemma 5 and Lemma 6 (class-weighted margin certificate) turn max-min into an average. Our averaging over delta != 0 is the same device, written in the reduced coverage variables.
- Step 2 is their App. F.3, "Proof that all frequencies are used". Lemma 6 (second part: every point in spt(q*) is on the margin) says f has equal margin across classes. So f(a,b,c) = F(a,b) + lambda * 1_{a+b=c}. Its 3-D DFT is "lambda p^2 if j=k=-l" on the diagonal. "Hence, when j=k=-l != 0, we must have f^(j,k,l) > 0 ... there is at least one neuron for each frequency."
  The ingredients match ours: constant margin, then Fourier transform of the indicator, then every frequency is nonzero. They conclude only that each frequency is nonzero (at least one neuron per frequency). Their decomposition actually gives the same value lambda p^2 at every diagonal frequency, so equal amplitudes follow from their argument, but they do not state it. Their Theorem 7 claims (1) single-frequency neurons and (2) all frequencies used. It does not claim equal amplitudes.
- The value: their Thm 7 margin sqrt(2/27)/(p^{1/2}(p-1)). The corollary already reproduces this and cites them.

## What is ours
- The reduced statement: uniqueness of the whole amplitude vector y (flat, y_j = 2/(p-1)) for the budget programme Gamma_1.
- Through Thm universal, this holds for any network with b = nu, not only the quadratic L_{2,3} network.
- The proof is a two-line Fourier inversion in one variable (delta) rather than a 3-D DFT.

## Adjacent fields
The averaging step followed by "equality iff constant off-diagonal correlation" has the same shape as Welch/simplex-type bounds and equal-norm tight frames (flat spectrum iff constant autocorrelation). This is folklore. I searched for no specific citation and recommend none, because Morwani is the right credit.

## Recommended change
In app:dense (or after cor:dense), add one sentence: "The averaging step is the uniform class-weighting certificate of Morwani et al. (2024, Lemmas 5-6), and the Fourier-inversion step parallels their App. F.3, which shows every frequency is present; the reduction lets us sharpen 'present' to 'equal amplitude'."
