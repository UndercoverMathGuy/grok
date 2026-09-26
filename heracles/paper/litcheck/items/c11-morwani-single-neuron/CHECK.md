# C11 morwani single-neuron: check

Sentence (approach.tex line 54): "It remains to price amplitudes in norm, which, as in \citet{morwani2024feature}, is a single-neuron optimization."

Source: Morwani, Edelman, Oncescu, Zhao, Kakade, "Feature emergence via margin maximization: case studies in algebraic tasks", ICLR 2024 (Spotlight), arXiv 2311.07568 (v2, 19 Feb 2024). Checked against the arXiv abstract page and the full PDF text.

## What Morwani actually proves
- Sec 3.1, Lemma 3 (binary; proof in App E.1), and Sec 3.2, Lemma 5 (multi-class, class-weighted margin; App E.2). Their "Single neuron optimization" bullet: fix a dual distribution q. Any theta in argmax_{||theta||_{a,b}<=1} E_q[g(theta)] "has directional support only on" Omega*_q = argmax_{||omega||_a<=1} E_q[psi(omega)].
- The argument: "the expected margin of the network decomposes into the sum of expected margin of individual neurons", E_q g = sum_i ||omega_i||_a^nu E_q psi(omega_i/||omega_i||_a) <= gamma max_{||w||_b<=1} ||w||_nu^nu.
- The combining-neurons part needs b = nu. The Remark in App E.1 says other b values do not give "the same flexibility to build a theta* satisfying Equation 1".
- These lemmas are used inside the certificate-pair (max-min) method of Lemma 2 / Lemma 6. For modular addition, Theorem 7 and App F maximise sum_j u^(j) v^(j) w^(-j) over a unit single neuron.

## Does the sentence hold?
It is partly supported. The shared idea is real. A network objective is a sum over neurons, and by homogeneity maximising it reduces to maximising a per-unit-norm single-neuron quantity. That is Morwani's device.

The objective is different, though:
- Morwani maximises a LINEAR functional: the expected margin under a fixed dual q. They do this to build a primal-dual certificate, and their construction works only at b = nu.
- The paper defines C_b = max_{||omega||_2<=1} sum_j |beta_j(omega)|^{b/nu}. This is a concave l_{b/nu} "budget" of one neuron's Fourier coefficients. It feeds a subadditivity upper bound (Thm universal), with no dual distribution and no certificate, and it covers 2 <= b <= nu.
- The step that needs the reduction differs too. Here it is subadditivity of t^r plus homogeneity. In Morwani it is linearity of expectation plus homogeneity.

Saying "as in Morwani" without qualification suggests we reuse their lemma. We reuse the decomposition idea, not the lemma. (Separately, Morwani's certificate route is blocked for b < nu, since there is a duality gap. So the difference matters, and a reviewer who knows their paper could flag it.)

## Verdict
PARTIALLY_SUPPORTED. The citation is right, but the attribution needs a qualifier and a lemma pointer.

## Suggested fix
"It remains to price amplitudes in norm. As in the single-neuron step of \citet[Lemmas 3 and 5]{morwani2024feature}, homogeneity reduces this to one neuron. Here the per-neuron quantity is an $\ell_{b/\nu}$ budget rather than an expected margin under a dual distribution."
