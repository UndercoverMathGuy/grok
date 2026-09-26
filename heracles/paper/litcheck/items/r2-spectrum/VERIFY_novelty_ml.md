# R2-spectrum: independent verification, lens novelty_ml

**Item:** `approach.tex` eq:spectrum and `appendix.tex` app:spectrum.
- Claim: eta_theta(delta) = E_x f(theta,x)[y(x)-delta] depends only on the diagonal modes (j,...,j;-j).
- Hence eta = mean + sum_{j in F_p} Re(beta_j rho^{j delta}), with beta_j = 2 p^{-(n+1)} fhat(j,...,j;-j).

**Checker verdict:** KNOWN.
**My verdict:** KNOWN. The checker is upheld, with minor pointer fixes and several missed adjacent precedents listed below.

## 0. Is the formula correct?

I checked it numerically (scratchpad `chk.py`, numpy, random real tables), using the paper's DFT convention.
- p=5, n=2: the reconstruction of eta from beta has max error 6e-17.
- p=7, n=3: max error 1e-16.
- Product neuron: 2p^{-(n+1)} fhat(j,..,j;-j) = 2p^{-nu} prod_t uhat_t(j) conj(what(j)), to 1e-17.

Both forms agree for real w: approach.tex writes conj(what(j)) and appendix.tex writes what(-j).

## 1. Checker's citations: fetched and re-read myself

I downloaded both PDFs from arxiv.org/pdf and read them with pdftotext.

### morwani2024feature

Source: arXiv 2311.07568 v2. The abs page says "Accepted as Spotlight at ICLR 2024". Authors: Morwani, Edelman, Oncescu, Zhao, Kakade. Bib fields are correct.

**App. F.1 ("Proof that Equation 3 is satisfied"):**
- Defines eta~_{u,v,w}(delta) := E_{a,b}[u(a)v(b)w(a+b-delta)].
- Derives "= 1/p^3 sum_j uhat(j) vhat(j) what(-j) rho^{j delta} (only terms where j+l = k+l = 0 survive)".
- **Pointer fix:** this identity is the unnumbered display *before* Eq. (5). Eq. (5) is the objective and Eq. (6) is its value, 2/((p-1)p^2) sum_{j!=0} uhat vhat what(-j).
- The cos(theta_u+theta_v-theta_w) form is the display leading to Eq. (7), not Eq. (6).
- Cite it as "App. F.1", not "App. F.1, Eq. (5)".

**Theorem 7 "Proof outline" (main text, p.10):**
- It states the *consequence* E_{a,b}[psi'] = 2/((p-1)p^2) sum_{j!=0} uhat(j)vhat(j)what(-j), i.e. Eq. (6).
- It does not state eta~(delta) itself. The checker's "repeats the identity" is loose but harmless.

**Missed by the checker: App. F.3 ("Proof that all frequencies are used").** This is already a network-level diagonal-mode statement:
- It introduces the multidimensional DFT fhat(j,k,l) of the whole network f(a,b,c).
- It shows each neuron's product term is nonzero only at j,k,l = ±zeta.
- It decomposes the max-margin f = F(a,b) + lambda·1[a+b=c].
- It notes fhat_2(j,k,l) = lambda p^2 iff j = k = -l.
- It concludes "when j = k = -l != 0, we must have fhat(j,k,l) > 0".
- This is the closest network-level precedent for reading the logit table on the diagonal. Li et al. Lemma F.1 is its k=3 copy.

### li2025nsum

Source: arXiv 2402.09469 v4. The abs page comment reads "AIStats 2025". Authors: Li, Liang, Shi, Song, Zhou. Bib fields are correct.

**App. D.4, Lemma D.7 and its proof:**
- Def. D.5 defines eta with the *power* neuron: E[(u_1(a_1)+...+u_k(a_k))^k w(a_1+...+a_k-delta)].
- Inside the proof of Lemma D.7:
  - "We note that all terms are zero rather than w(·)·prod u_i(a_i)", i.e. only the multilinear monomial survives, which gives the k! factor.
  - Then eta~(delta) := E[w(a_1+...+a_k-delta) prod_i u_i(a_i)] = p^{-(k+1)} sum_j what(-j) rho^{j delta} prod_i uhat_i(j).
- The k=3 worked case shows the same computation, noting "only terms where j1+j4 = j2+j4 = j3+j4 = 0 survive".
- The checker says "Def. D.5 and Lemma D.7 proof". eta~ is defined inside the proof of D.7, not in D.5. Cite as "proof of Lemma D.7".

**Lemma F.1 / Theorem G.1 (k=3):**
- Lemma F.1: "for any j1 = j2 = j3 = -j4 != 0, we have fhat(j1,j2,j3,j4) > 0" for the max-L_{2,4}-margin f.
- Theorem G.1 invokes it to show that every frequency is used.
- Confirmed.

### nanda2023progress and gromov2023grokking

Already in refs.bib. I read the PDFs, not only the abstracts.

**Nanda §4.2:**
- "We approximate the output logits as the sum_k alpha_k cos(w_k(a+b-c)) ... This approximation explains 95% of the variance in the original logits."
- This is an empirical network-level projection onto diagonal modes. It supports the "feature = frequency with A_j > 0" wording.
- §5.1 "restricted loss" ablates non-key Fourier terms of the logits.

**Gromov Eqs. (10)-(16):**
- Expands one quadratic neuron into 8 cosine terms.
- Only cos 2π(n+m-q)/p survives phase averaging; the rest cancels.
- Result: sum_k cos 2πk(n+m-q)/p ≈ δ(n+m-q).

Neither paper states the averaging identity. The checker's characterization is accurate.

## 2. Adversarial search (ML lens)

**Search log:**
- The WebSearch budget was exhausted for the session.
- The Semantic Scholar and export.arxiv.org APIs returned 429.
- I therefore used arxiv.org/search (abstract field) pages via WebFetch:
  - "modular addition" margin
  - "modular addition" Fourier (21 hits, through 2609.x)
  - "modular arithmetic" margin
  - grokking margin Fourier
  - "max-margin" Fourier features
  - group operations/composition Fourier irreducible
  - "modular addition" + number of frequencies (40 hits, through Sept 2026)
- I then downloaded and grepped the PDFs of every plausible hit.

**Found, and missed by the checker.** All are adjacent. None is the network-level all-delta statement plus the Re(beta_j)·cov_j split.

1. **Tian, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets", NeurIPS 2025** (arXiv 2410.01779 v4; abs comment "NeurIPS 2025 camera ready"; the PDF header says 39th NeurIPS).
   - Theorem 1: the MSE loss of a quadratic 2-layer net on an Abelian group is l = d^{-1} sum_{k!=0} l_k + (d-1)/d.
   - l_k = -2 r_kkk + sum_{k1,k2} |r_{k1k2k}|^2 + (1/4) sum |sum_{k'} r_{p,k',-k',k}|^2 + ..., with r_{k1k2k} = sum_j z_{ak1j} z_{bk2j} z_{ckj}.
   - Lemma 1: global optimum iff r_kkk = 1 (diagonal) and r_{k1k2k} = 0 off-diagonal.
   - This is the MSE analogue of "the diagonal modes carry the signal and the off-diagonal modes are interference". The linear term -2 r_kkk is (up to constants) eta(0) = sum of diagonal modes.
   - Worth citing next to the definition of interference.
2. **Kunin, Marchetti, Chen, Karkada, Simon, DeWeese, Ganguli, Miolane, "Alternating Gradient Flows", NeurIPS 2025** (arXiv 2506.06489 v4; abs comment "NeurIPS 2025").
   - §5: the initial utility of a quadratic neuron on modular addition is U = (2/p^3) sum_{k!=0} |xhat[k]|^2 uhat[k] vhat[k] what[k] xhat[k].
   - This is the single-neuron diagonal-mode correlation at delta=0 (Thm 5.1).
3. **Marchetti, Kunin, Myers, Acosta, Miolane, "Sequential Group Composition", ICML 2026** (arXiv 2602.03655; abs comment "Accepted at ICML 2026").
   - App. B.1, Lemma B.1: sum_g <w, x_{g1...gk}> prod_i <u_i, x_{gi}> = |G|^{-1} sum_rho <uhat_1† xhat ... uhat_k† xhat, what† xhat>_rho.
   - Lemma B.2: for a polynomial activation, "the only monomial where all the group elements g1,...,gk appear is k! prod <u_i, x_gi>; for any other monomial the term ... will vanish".
   - This is the k-operand, general-group, delta=0 single-neuron version, including the power-to-multilinear reduction the paper uses for the power neuron.
   - Relevant to the power-neuron claim and to the groups section.
4. **He, Wang, Zhang, Chen, Yang, "Neural Networks Provably Learn Spectral Representations for Group Composition"** (arXiv 2606.02993, June 2026, preprint).
   - Energy functional Omega_m = sum_rho d_rho tr(xihat_m[rho]^* thetahat^2_m[rho] thetahat^1_m[rho]); for abelian groups this is sum Re(xihat^* thetahat^2) (App. C.1, Eq. C.2).
   - It is the per-neuron correlation with the label, i.e. the group-Fourier diagonal term.
   - Dynamics paper; no margin.
5. **He et al. 2602.16849** (Feb 2026, already known).
   - Calls cos(w_k(x+y-j)) the "primary signal" and the other terms "noise" (Eq. B.1).
   - Terminological parallel only.
6. **Doshi, He, Das, Gromov, "Grokking Modular Polynomials"** (ICLR 2024 Workshop BGPT, arXiv 2406.03495).
   - Many-term addition with analytic power-activation solutions.
   - Cosine expansion in which only the n1+...+nS-q term survives; same flavour as Gromov.
   - Not the averaging identity.

**Checked and irrelevant to this item:**
- 2511.23443 (Huang & Li): sine activation, Natarajan bounds.
- 2609.26679 (Pracher et al., Sept 2026): NTK and weight decay; no Fourier averaging.
- 2608.04451 (single-neuron ReLU counterexample).
- Zhong et al. 2306.17844 (Clock/Pizza logit forms).
- Chughtai 2302.03025 (the character form 2cos(ω(a+b-c)), empirical).

**Not found anywhere:**
- The identity stated for an *arbitrary* logit table at *all* offsets delta, with the off-diagonal remainder named interference and shown to average out (Lemma avg).
- The pairing that gives Re(beta_j)(1 - cos 2πjδ/p).

Both are one-line consequences of character orthogonality, so they are not a contribution. The paper's text does not claim them as one.

**Harmonic-analysis origin (annihilator / Poisson summation on finite abelian groups):** unconfirmed. I tried Wikipedia "Poisson summation formula" and "Fourier transform on finite groups"; neither states the finite-group version. No bib entry is emitted. It is not needed: E_a[rho^{ka}] = [k=0] is all that is used.

## 3. Recommendation

- Keep the checker's recommended sentence, with corrected pointers:
  > "This is the network-level form of the single-neuron identity of \citet[App.~F.1]{morwani2024feature} and \citet[proof of Lemma~D.7]{li2025nsum}; \citet[App.~F.3]{morwani2024feature} already read the max-margin network's multidimensional DFT on the diagonal $j=k=-\ell$."
- Optional, one clause where interference is defined:
  > "(the analogue, for squared loss, of the split between diagonal and off-diagonal sum potentials in \citet[Theorem~1, Lemma~1]{tian2025composing})".
- Optional, in the power-neuron appendix:
  > "(the same monomial-survival argument appears in \citet[proof of Lemma~D.7]{li2025nsum} and \citet[Lemma~B.2]{marchetti2026sequential})".
- Do not present eq:spectrum as a contribution. The paper currently does not, so there is no overclaim to remove.

## Bib entries (all fetched and confirmed this session)

```bibtex
@inproceedings{tian2025composing,
  title={Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets},
  author={Tian, Yuandong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2410.01779}
}
@inproceedings{kunin2025alternating,
  title={Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks},
  author={Kunin, Daniel and Marchetti, Giovanni Luca and Chen, Feng and Karkada, Dhruva and Simon, James B. and DeWeese, Michael R. and Ganguli, Surya and Miolane, Nina},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2506.06489}
}
@inproceedings{marchetti2026sequential,
  title={Sequential Group Composition: A Window into the Mechanics of Deep Learning},
  author={Marchetti, Giovanni Luca and Kunin, Daniel and Myers, Adele and Acosta, Francisco and Miolane, Nina},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  note={arXiv:2602.03655}
}
```

**URLs fetched:**
- https://arxiv.org/abs/2311.07568 and https://arxiv.org/pdf/2311.07568
- https://arxiv.org/abs/2402.09469 and https://arxiv.org/pdf/2402.09469
- https://arxiv.org/abs/2301.05217 and https://arxiv.org/pdf/2301.05217
- https://arxiv.org/abs/2301.02679 and https://arxiv.org/pdf/2301.02679
- https://arxiv.org/abs/2410.01779 and https://arxiv.org/pdf/2410.01779
- https://arxiv.org/abs/2506.06489 and https://arxiv.org/pdf/2506.06489
- https://arxiv.org/abs/2602.03655 and https://arxiv.org/pdf/2602.03655
- https://arxiv.org/abs/2606.02993 and https://arxiv.org/pdf/2606.02993
- https://arxiv.org/pdf/2602.16849
- https://arxiv.org/pdf/2406.03495
- https://arxiv.org/abs/2511.23443
- https://arxiv.org/abs/2609.26679
- https://arxiv.org/abs/2608.04451
- arxiv.org/search result pages as listed above
