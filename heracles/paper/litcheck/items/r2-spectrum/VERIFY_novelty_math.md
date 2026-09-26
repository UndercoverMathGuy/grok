# R2-spectrum: independent verification (lens: novelty_math)

**Item:** approach.tex `eq:spectrum` and appendix `app:spectrum`. The claim: eta_theta(delta) = E_x f(theta,x)[y(x)-delta] depends only on the diagonal modes fhat(j,...,j;-j), so eta = mean + sum_{j in F} Re(beta_j rho^{j delta}) with beta_j = 2 p^{-(n+1)} fhat(j,...,j;-j).

**Verdict: KNOWN (upheld), and the case is stronger than the checker made it.**
- The network-level identity, not just the single-neuron one, is a direct instance of a textbook formula: the Poisson summation formula on F_p^{n+1}.
- The network-level diagonal mode also already appears in Morwani App. F.3.
- The checker's two citations are real, and they say what the checker claims. The only problems are two imprecise pin-cites (details below).

## 0. Math re-derived and checked numerically
- Take H = {(a,c) in Z_p^{n+1} : c = a_1+...+a_n} and z = (0,...,0,-delta). Then eta(delta) is the average of the table over the coset H+z.
- The annihilator is H^perp = {(j,...,j,-j)}, and chi_gamma(z) = rho^{j delta}.
- The paper's formula follows exactly, using the unnormalised fhat (factor p^{-(n+1)}).
- Numerical check: a random real table at p=5, n=2 matches the formula at all five delta to 1e-10.
- Re(beta_k) equals the least-squares coefficient of cos(2 pi k (a+b-c)/p) fitted over the full p^3 table (checked numerically). This is exactly the alpha_k that Nanda et al. fit in Sec. 4.2.
- Script: scratchpad r2v/chk.py.

## 1. Textbook source (the checker did not fetch one)
- **O'Donnell, *Analysis of Boolean Functions*, CUP 2014 (arXiv 2105.10386, fetched and read).**
  - Section 3.3, "Poisson Summation Formula" (p. 78, arXiv edition), called "the following classical fact": for H <= F_2^n and z in F_2^n, E_{h~H}[f(h+z)] = sum_{gamma in H^perp} chi_gamma(z) fhat(gamma).
  - Section 8.5 (p. 229): "all of the notions from Chapters 3.2 and 3.3 regarding affine subspaces and restrictions thereto generalize easily to L^2(F_p^n)."
  - Theorem 8.60 is the convolution theorem on finite abelian groups. It covers the product-neuron factorisation.
- **Consequence:** app:spectrum is a one-line application of Poisson summation. It is also the discrete projection-slice identity: the DFT of the projection of the table along the hyperplanes c = sum a - delta equals the slice of its (n+1)-dimensional DFT on the diagonal line.

## 2. Checker's citations, re-fetched
- **Morwani et al., ICLR 2024 (arXiv 2311.07568).** I fetched the PDF and confirmed the venue on proceedings.iclr.cc (listed as "ICLR 2024 Conference"; authors Morwani, Edelman, Oncescu, Zhao, Kakade).
  - **Content: correct.** App. F.1 defines eta~_{u,v,w}(delta) := E_{a,b}[u(a)v(b)w(a+b-delta)] and derives eta~ = p^{-3} sum_j uhat(j)vhat(j)what(-j) rho^{j delta}, with the remark "(only terms where j + l = k + l = 0 survive)".
  - **Pin-cite 1 is imprecise.** That identity is the unnumbered display just before Eq. (5). Eq. (5) is the objective (2p/(p-1))(eta~(0) - E_delta eta~(delta)), Eq. (6) is 2/((p-1)p^2) sum_{j != 0} uhat vhat what(-j), and the cosine/phase form is Eq. (7). Cite "App. F.1 (display before Eq. (5)); Eqs. (6)-(7)", not "Eq. (5)".
  - **Pin-cite 2 is imprecise.** The Theorem 7 proof outline (Sec. 4) states the consequence E_{a,b}[psi'] = 2/((p-1)p^2) sum uhat vhat what(-j), not the eta~ identity itself.
  - **Missed by the checker:** App. F.3 already defines the multidimensional DFT of the network function f(a,b,c) and uses the network-level diagonal mode: "when j = k = -l != 0, we must have fhat(j,k,l) > 0". So the network-level diagonal-mode object is Morwani's, not ours.
  - **Bib entry: OK.**
- **Li, Liang, Shi, Song, Zhou (arXiv 2402.09469 v4, fetched).** Venue confirmed on proceedings.mlr.press/v258: AISTATS 2025, PMLR 258:523-531.
  - Def. D.5 defines the power-neuron eta with (u_1+...+u_k)^k.
  - The product-term eta~ and its formula p^{-(k+1)} sum_j what(-j) rho^{j delta} prod_i uhat_i(j) appear inside the proof of Lemma D.7. The k=3 version is in the Sec. D proof, with "only terms where j1+j4 = j2+j4 = j3+j4 = 0 survive".
  - Lemma F.1 (k=3): "for any j1 = j2 = j3 = -j4 != 0, we have fhat(j1,j2,j3,j4) > 0". Theorem G.1 uses it.
  - **Appendix numbering:** the PMLR paper is 9 pages, so the lemma numbers refer to the arXiv version.
  - **Bib entry: OK.** Optionally add series=PMLR, volume=258, pages=523--531.
- **Nanda et al. 2023 (full PDF fetched; the checker read only the abstract).** Sec. 4.2 fits the logits as sum_k alpha_k cos(w_k(a+b-c)) by OLS, with 95% of variance explained. By Sec. 0 above, the fitted alpha_k is Re(beta_k). Nanda states no averaging lemma.
- **Gromov 2023 (full PDF fetched; arXiv preprint only, 11+5 pages).**
  - Sec. 3.1, Eqs. (10)-(16), expands the constructed network's output into waves. The (n+m-q) wave is "constructive interference"; every other term must "interfere destructively" (Eq. (14)-(15), random phases).
  - This is the same signal-versus-interference split as the paper's diagonal-versus-interference split, shown for one construction.
  - Gromov states no averaging identity.

## 3. Other prior art found (adjacent, not scooping)
- **Tian, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets", NeurIPS 2025.** Source: arXiv 2410.01779 (fetched); the comments say "NeurIPS 2025 camera ready" and the PDF header says NeurIPS 2025.
  - Thm. 1 writes the MSE loss of a quadratic 2-layer net on an abelian group through network-level "sum potentials" r_{k1 k2 k} = sum_j z_{a k1 j} z_{b k2 j} z_{c k j}.
  - Only the diagonal r_kkk couples to the target (the term -2 r_kkk). The off-diagonal terms enter as |r|^2 penalties, and Lemma 1 sets them to 0.
  - This is the MSE analogue of "only diagonal modes survive / off-diagonal = interference".
  - It is optional related work. It does not state the margin-averaging identity.
- **Doshi, He, Das, Gromov, "Grokking modular polynomials"** (arXiv 2406.03495, ICLR 2024 BGPT workshop, fetched). Eqs. (18)-(19) show the n-term analytic construction yields cos(sigma(k)(c_1 n_1+...+c_S n_S - q)). This is the same construction-level observation for n operands. No identity is stated.
- **Mohamadi et al. 2024** (arXiv 2407.12332, fetched): nothing relevant.

## 4. Search limits
- The WebSearch budget for the session was exhausted, and the arXiv API returned 429.
- I therefore fetched known primary sources directly by URL: arXiv PDFs, proceedings.iclr.cc and proceedings.mlr.press.
- I did not fetch Tao-Vu or Terras. O'Donnell is enough for a textbook pin.

## 5. Recommended change (refines the checker's)
- **After eq:spectrum:** "This is the network-level form of the single-neuron identity of \citet[App.~F.1]{morwani2024feature} and \citet[proof of Lemma~D.7]{li2025nsum}; the diagonal network mode already appears in \citet[App.~F.3]{morwani2024feature}."
- **At the start of app:spectrum:** "The identity is the Poisson summation formula on $\sZ_p^{n+1}$ \citep[Sec.~3.3, 8.5]{odonnell2014analysis} for the subgroup $\{c=a_1+\dots+a_n\}$, whose annihilator is $\{(j,\dots,j;-j)\}$."
- **Optional, after "we call it interference":** cite \citet[Sec.~3.1]{gromov2023grokking} for the constructive/destructive interference language.
- **Optional remark:** "$\mathrm{Re}\,\beta_j$ is the least-squares coefficient of $\cos(2\pi j(a_1+\dots+a_n-c)/p)$ in the logit table, the quantity fitted by \citet[Sec.~4.2]{nanda2023progress}."
- Do not claim eq:spectrum as a contribution. The notation nit stands: approach.tex writes conj(what(j)) and the appendix writes what(-j).
