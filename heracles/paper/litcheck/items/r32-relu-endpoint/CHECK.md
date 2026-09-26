# R32-relu-endpoint: literature check

**Claim** (appendix.tex:283, "ReLU neuron" paragraph): the ReLU neuron sigma(sum_t u_t(a_t)) w has nu=2. So plain weight decay is L_{2,2}, which is the b = nu endpoint. The paper proves C_2 <= p^{-n}/2 using Cauchy-Schwarz, Parseval and Var(sigma(X)) <= Var(X) for 1-Lipschitz sigma. From this bound, Thm universal and the flat maximizer of Cor dense apply. The paper says explicitly that this is not a theorem about the ReLU network: the sign lemma is not claimed, and nobody knows whether the bound is attained.

**Verdict: PARTIALLY_KNOWN.** The empirical side is known. That L2+ReLU sits at b=nu follows at once from Morwani's definitions. I found no paper that states the C_2 budget bound for ReLU, or that says weight decay on ReLU nets is priced like the dense L_{2,nu} endpoint.

## What prior work has (all fetched 2026-09-25)

- **Morwani et al., ICLR 2024 (arXiv 2311.07568).**
  - Sec 2 defines the homogeneity constant nu. Thm 1 (quoting Wei et al.) covers any norm and any homogeneous f. Thm 2's "combining neurons" step needs b = nu.
  - All of their theory is for x^2 / x^k activations with the L_{2,nu} norm. The conclusion says: "Our proof holds for the case of x2 activations ... and L2,nu norm ... Empirical findings suggest that the results may be transferable to other architectures and norms."
  - There is no ReLU theorem. ReLU appears only in experiments:
    - Fig 1a: a ReLU + L2 network at mod 71.
    - Fig 4b: "For 1-hidden layer ReLU network trained with L2 regularization, the final distribution of maximum power seems to be concentrated around 0.9, meaning neurons are nearly 1-sparse in frequency space but not quite."
    - **App. C, Fig 7(a):** "Final distribution of the neurons corresponding to a particular frequency in (a) ReLU network trained with L2 regularization ... the final distribution across frequencies is close to uniform." Also: "for both ReLU and quadratic activation, the distribution is close to uniform."
  - So the observation that ReLU + L2 is dense and flat is Morwani's (empirical). Our bound gives the first budget-level reason for it, and it is not a proof.
- **Gromov, arXiv 2301.02679 (2023).**
  - His analytic solution (Eqs 6-7) uses the full frequency set k = 0..(p-1)/2 with random phases, and is built for quadratic activation.
  - Sec 3 / App. B: "The weights (6)-(7) also work for other activation functions, including ReLU, however the 100% accuracy is achieved at higher width." And: "We do not have an analytic proof of this fact."
  - There is no norm, no margin, and no regularisation ("in the absence of any regularization"). This is a dense construction that works for ReLU, not a max-margin statement.
- **McCracken et al., arXiv 2505.18266 (2025).**
  - Sec 4.4: "Previous work shows floor(n/2) types of neurons are found, each specializing to a sinusoid with one frequency with 1 hidden layer" (citing Gromov and Morwani). Adding depth or trainable embeddings moves the network to far fewer frequencies.
  - This is empirical and makes no norm argument. It matches our scope note: a shared linear layer makes a network non-neuron-additive, and such networks are outside the paper.
- **He, Wang, Chen, Yang, arXiv 2602.16849 (2026).**
  - Experiments use ReLU; theory uses x^2.
  - Observation 3 (Sec 3.1): "every frequency k is represented among the neurons", stated as an empirical finding.
  - Def 4.1 ("full diversification"): exactly N neurons per frequency.
  - No ReLU margin theorem.
- **Memana, arXiv 2608.04451 (2026).** A single ReLU neuron under Clarke gradient flow can end with multi-frequency energy 2/(p-1) per class. This is about dynamics and does not overlap with the claim.

## Searches run
- "ReLU network modular addition max margin all frequencies weight decay theorem one hidden layer"
- "\"modular addition\" ReLU \"maximum margin\" OR \"max-margin\" frequencies dense uniform theory 2025 2026 arXiv"
- "two-layer ReLU network modular addition weight decay provably Fourier features all frequencies theorem"

Also read the full texts of Morwani (PDF, grepped for ReLU and frequency) and Gromov (PDF), McCracken Sec 4.4 and 5.1 (HTML), and He et al. (HTML).

## Notes
- Var(sigma(X)) <= Var(X) for 1-Lipschitz sigma is standard (Var Y = E(Y-Y')^2 / 2). It needs no citation.
- Morwani Fig 4b says ReLU neurons are *not* exactly single-frequency. This fits the paper's decision not to claim the sign lemma for ReLU. Worth saying out loud.

## Recommended edit (appendix.tex:283)
After "...is the b = nu endpoint.", add:
"This matches what is observed: ReLU networks trained with L2 regularisation spread their neurons almost uniformly over all frequencies (Morwani et al., 2024, App. C, Fig. 7a; see also Gromov, 2023; He et al., 2026, Obs. 3), and their neurons are close to, but not exactly, single-frequency (Morwani et al., Fig. 4b)."

Then keep the existing caveat sentence. Keys needed: morwani2024feature, gromov2023grokking (both already in refs.bib), and he2026mechanism (add if not present). Do not use McCracken here, except optionally for the one-layer vs. embedding contrast in the "Where the approach does not apply" paragraph.
