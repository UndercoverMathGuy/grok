# VERIFY (novelty_ml) — R32-relu-endpoint (task 35)

Claim (appendix.tex, ReLU neuron paragraph): weight decay on a one-hidden-layer ReLU net (nu=2) is the b=nu endpoint;
C_2 <= p^{-n}/2 via Var(sigma(X)) <= Var(X); the universal bound's maximizer is flat (dense). No theorem about the network.

Verdict: checker's PARTIALLY_KNOWN is UPHELD.

## Checker citations: re-verified from full text this session
- Morwani et al. 2311.07568v2 (PDF text):
  - Fig. 7 caption: "Final distribution of the neurons corresponding to a particular frequency in (a) ReLU network trained with L2 regularization ... the final distribution across frequencies is close to uniform". App. C text: "for both ReLU and quadratic activation, the distribution is close to uniform". The pointer [App. C, Fig. 7a] is correct.
  - Fig. 4(b) caption: ReLU + L2 neurons are "nearly 1-sparse in frequency space but not quite". Correct.
  - Conclusion: "Our proof holds for the case of x2 activations ... and L2,nu norm ... Empirical findings suggest that the results may be transferable to other architectures and norms." Correct. No theorem covers ReLU.
  - Minor slip in the checker's evidence: the b=nu neuron-combining step is **Lemma 3** (arXiv v2), not "Thm 2". The recommended text does not use that number, so nothing needs fixing.
- Gromov 2301.02679: Sec. 3.1 says "The weights (6)-(7) also work for other activation functions, including ReLU". App. B ("Other activations") says "We do not have an analytic proof of this fact". No norm and no margin. Correct.
- He, Wang, Chen, Yang 2602.16849v1: Observation 3(ii) says "every frequency k is represented among the neurons". Def. 4.1 (Full Diversification) is correct. The experiments use ReLU. The only ReLU theory is Prop. 6.3, which is about dynamics; there is no margin theorem. Correct.
- McCracken et al. 2505.18266, Sec. 4.4: "floor(n/2) types of neurons are found ... with 1 hidden layer [6,7]", with Fig. 4. The MLPs are ReLU. This is empirical. Correct.

## Adversarial search (ML lens): nothing found that bounds a ReLU neuron's frequency budget or proves ReLU max-margin density
- Tian 2509.21519: Lemma 6 / Thm 5 contain "ReLU ... s* ∝ p", but this is about how *memorisation* solutions spread across data probabilities p_g. It has nothing to do with frequencies or primes. Not overlapping.
- Mohamadi, Li, Wu, Sutherland 2407.12332: quadratic activation and l_inf only. ReLU appears only in the references.
- Huang & Li 2511.23443: a ReLU two-layer modular-addition *margin lower bound* built from a Pizza/aCRT construction (Thm 6.3). It has no upper bound and says nothing about frequency density. Nearest neighbour, but not overlapping.
- Tan, Gai, Zhang 2606.05863 and 2606.08985; Sivasankar 2606.12966: not overlapping.
- Li et al. 2402.09469 (li2025nsum): ReLU appears only in related work.
- HF papers search, 4 queries: only Morwani and He come back as relevant.

## Better or canonical citation for the "b=nu endpoint" half
- Wei, Lee, Liu, Ma 2019 (already `wei2019regularization`), arXiv 1810.05369v4, App. B.4 "Connection to the l1-SVM", Prop. B.11: the max normalized margin of an l2-regularized two-layer ReLU network equals the l1-SVM margin over the lifted ReLU features (gamma_l1/2 = gamma*_{n+1} = ... = gamma*_inf), crediting Neyshabur et al.
  This is the classical statement that l2 weight decay on 2-homogeneous ReLU neurons is an l1 combination of unit-norm neurons, i.e. exactly the b=nu=2 endpoint. It is the canonical citation for that half, alongside Morwani's Lemma 3. The statement is binary; the multiclass analogue is Morwani's Lemma 3.
  Suggested wording: "...*is* the $b=\nu$ endpoint (the familiar equivalence of $\ell_2$ weight decay on two-layer ReLU networks with an $\ell_1$ combination of neurons; \citealp[App.~B.4]{wei2019regularization}; \citealp[Lemma~3]{morwani2024feature})."

## Novelty verdict
- The equivalence ReLU+L2 = b=nu is known and should be cited, not claimed.
- Dense frequency usage of ReLU+L2 networks is known empirically (Morwani Fig. 7a, Gromov, McCracken Fig. 4, He Obs. 3).
- New: the per-neuron budget C_2 <= p^{-n}/2 and how it feeds into Thm universal / Cor dense. No prior art found.
- The checker's recommended insertion is sound. Add the Wei pointer.

## Incidental bib error
- refs.bib `mccracken2025` is wrong. arXiv 2505.18266 is "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks" by Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Létourneau, Doina Precup and Jonathan Love. The entry has the title "Emergence of sparse and dense circuits..." and "Andrew McCracken".

## Tooling note (why verify is slow)
In this session, WebSearch returned "200 of 200 WebSearch calls" used; the budget is shared across the session. The arXiv export API and the Semantic Scholar API both returned HTTP 429. The shared scratchpad holds the same PDFs downloaded several times by parallel verifiers (for example gromov.pdf, g.pdf and 2301.02679.pdf are the same file). Searches for this item were done by scraping arxiv.org/search HTML and the HF papers search.
