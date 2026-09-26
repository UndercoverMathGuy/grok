# R2-spectrum: literature check

**Item:** approach.tex eq:spectrum and appendix app:spectrum. The claim: eta_theta(delta) depends only on the diagonal Fourier modes (j,...,j;-j), giving
eta = mean + sum_j Re(beta_j rho^{j delta}) with beta_j = 2 p^{-(n+1)} fhat(j,...,j;-j).

**Verdict: KNOWN.** This is a standard computation, and the single-neuron version already appears in both papers we build on. The network-level form follows by linearity. Keep it, but attribute it.

## Evidence (read from the PDFs this session)

- **Morwani et al. ICLR 2024 (arXiv 2311.07568), App. F.1, Eq. (5).**
  - They define eta~_{u,v,w}(delta) := E_{a,b}[u(a)v(b)w(a+b-delta)] and derive
    eta~(delta) = p^{-3} sum_j uhat(j) vhat(j) what(-j) rho^{j delta},
    noting "only terms where j+l = k+l = 0 survive".
  - Eq. (6) then gives the class-weighted margin 2/((p-1)p^2) sum_{j != 0} uhat vhat what(-j).
  - With phases this becomes 4/((p-1)p^2) sum |uhat||vhat||what| cos(theta_u+theta_v-theta_w). This is the cosine/phase form behind our Lemma phase.
  - The Theorem 7 proof outline states the same identity in the main text.
  - This is our formula for a product neuron at n=2.
- **Li, Liang, Shi, Song, Zhou, AISTATS 2025 (arXiv 2402.09469), App. D, Lemma D.7 and Definition D.5.**
  - For k inputs, eta~_{u1..uk,w}(delta) = E[w(a1+...+ak-delta) prod u_i(a_i)] = p^{-(k+1)} sum_j what(-j) rho^{j delta} prod_i uhat_i(j).
  - This is exactly our product-neuron beta_ij for general n.
  - At the network level, Lemma F.1 and Thm G.1 state that the max-margin f has fhat(j1,j2,j3,j4) > 0 on the diagonal j1=j2=j3=-j4 != 0. So they already single out the diagonal network-level mode.
- **Nanda et al. ICLR 2023 and Gromov 2023:** trained and analytic solutions compute cos(w(a+b-c)), i.e. the diagonal mode. This supports the terminology only. Neither states the averaging lemma. Confirmed from the arXiv abstract pages.
- **Harmonic analysis:** averaging a function on Z_p^{n+1} over the coset {c = a_1+...+a_n - delta} keeps only the Fourier coefficients on the annihilator of that subgroup. This is Poisson summation for finite abelian groups (textbook fact; I did not fetch a specific textbook page).

## What is ours

- Stating it for an arbitrary network f_theta, not just for the neuron class. This lets interference be defined as the off-diagonal part, and it lets the same beta be used for ReLU (appendix, "ReLU neuron", diagonal hat h).
- Pairing j with -j to get Re(beta_j) cov_j(delta).

Both are a trivial extension.

## Recommended change

- After eq:spectrum, add: "This is the network-level form of the single-neuron identity of \citet[App.~F.1, Eq.~(5)]{morwani2024feature} and \citet[Lemma~D.7]{li2025nsum}."
- In app:spectrum, say "the computation is that of \citet[App.~F.1]{morwani2024feature}, for a general table".
- Minor consistency issue: approach.tex writes the product-neuron factor as conj(what(j)), while the appendix writes what(-j). These are equal for real w. Consider using one notation throughout.

No new bib entries are needed. morwani2024feature, li2025nsum and nanda2023progress are already in refs.bib and were verified.
