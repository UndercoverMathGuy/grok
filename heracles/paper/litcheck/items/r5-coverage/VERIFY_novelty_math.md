# R5-coverage: independent verification (lens: novelty_math)

Item: approach.tex prop:coverage (lines 47-52), h(theta) <= Gamma_s(p) ||A||_{1/s}.
Checker verdict: NOVEL_AS_FAR_AS_FOUND. Checker citations: kam2026symmetry, morwani2024feature.

## Verdict
The checker's label is too generous. I would call this PARTIALLY_KNOWN.
- The step is a one-line rescaling of a positively 1-homogeneous objective. That is textbook.
- **The s=1 case is in the literature, stated explicitly.** Kam et al. (2026) prove it in the modular-addition setting, and it is classical in graph-spectral form (Fiedler 1973).
- For s>1, the l_{1/s}-quasi-norm version was not found. That version is the new part, and it inherits all its novelty from the programme (R4).

## Citations the checker confirmed, re-fetched by me

### 1. kam2026symmetry: confirmed, but the checker described it inaccurately
Source: https://arxiv.org/abs/2609.17926 (abs page). App. A.3.2 read verbatim from https://arxiv.org/html/2609.17926.
- Title: "Symmetry without a manifold: intrinsic dimension on orbits".
- Authors: Chon-Fai Kam, Miloud Bessafi, Frédéric Cadet. v1 dated 15 Sep 2026.
- Comments field: "Under review at NeurReps 2026 (NeurIPS workshop, Proceedings Track)". The bib entry is correct.

What App. A.3.2 actually says (quadratic two-layer network, sigma(z)=z^2):
- Setup: "write beta=(beta_k)_{k in S} subject to ||beta||_2 = B".
- Eq. (15): Delta_S(beta) = min_{s!=0} sum_{k in S} beta_k (1-cos(omega_k s)).
- The loss bound L <= log(1+(p-1)e^{-Delta_S}).
- Prop. 5, "Optimal allocation": "Let S={1,...,K_max}. The uniform allocation beta_k=B/sqrt(K_max) is optimal, attaining Delta*_S(B) = B(K_max+1/2)/sqrt(K_max) = Bp/sqrt(2(p-1))".
- **Proof, Eq. (18):** "Delta_S(beta) <= (p/(p-1)) sum_{k in S} beta_k <= (p/(p-1)) sqrt(K_max) ||beta||_2". It uses sum_{s!=0} M_{sk} = p.
  - The first inequality is exactly prop:coverage at s=1 with Gamma_1(p) = p/(p-1).
  - It is also the averaging step in the paper's proof of cor:dense (appendix.tex line 28), word for word: sum_{delta!=0} kappa_j(delta) = p, then average.
  - The second inequality is Cauchy-Schwarz. That is where their l2 budget enters.
- Numerics: for proper subsets S, the Nelder-Mead optimum beats uniform by 25% (K=4) and 28% (K=10) at p=47. They never optimise over S under a concave budget.

Correction to the checker:
- The checker says the "margin <= return x budget" rescaling is "implicit" in Kam, "with exponent 2 in place of 1/s". That undersells the overlap.
- The l1 (s=1) inequality is written out explicitly in Eq. (18).
- The difference from ours: Kam applies it to idealised amplitude-only logits, not as a bound for arbitrary networks. They have no phases, no Re(beta) <= |beta| step, and no interference.
- Kam is dated 15 Sep 2026, 10 days before the deadline. So it is contemporaneous, and citing it is courtesy, not obligation.

### 2. morwani2024feature: confirmed
Source: https://arxiv.org/abs/2311.07568. Title and authors match the bib. v1 13 Nov 2023, v2 19 Feb 2024. Comments: "Accepted as Spotlight at ICLR 2024".

The paper text (v2) contains the s=1 budget linearity at the neuron level:
- Lemma 3 / Lemma 5, "Combining neurons": "If b = nu and omega_1*,...,omega_m* in Omega'*_q, then for any neuron scaling factors sum lambda_i^nu = 1, lambda_i >= 0, we have that theta = {lambda_i omega_i*} belongs to Theta'*_q."
- Sec. 4, uniform class weighting: E[psi'] = 2/((p-1)p^2) sum_{j!=0} u^(j) v^(j) w^(-j).
- So the b=nu (s=1) "margin = best per-unit return x total budget" structure is Morwani's.
- Our Fourier-amplitude, worst-offset, l_{1/s} form is the generalisation.
- The checker's description ("Fourier-amplitude/phase-condition margin analysis at b=nu") is fine. Lemma 5 is the more precise pointer.

## Adversarial search: maths (adjacent fields)

### A. Graph spectra: Gamma_s is the l_{1/s}-budget spectral gap of a weighted circulant
F_y(delta) = sum_j y_j kappa_j(delta) is the delta-th Laplacian eigenvalue of Cay(Z_p, +-j) with weight y_j/2 on each generator. Its minimum over delta != 0 is the algebraic connectivity lambda_2.

**Fiedler (1973): the s=1 bound is classical.**
- M. Fiedler, "Algebraic connectivity of graphs", Czechoslovak Math. J. 23(2):298-305, 1973.
- Pages: https://dml.cz/handle/10338.dmlcz/101168. PDF fetched and read: https://dml.cz/bitstream/handle/10338.dmlcz/101168/CzechMathJ_23-1973-2_11.pdf.
- 3.5: "a(G) <= [n/(n-1)] min v_i <= 2|E|/(n-1)". This rests on the Lemma: "Let M be a symmetric positive semidefinite n by n matrix such that Me = 0. Then the second smallest eigenvalue lambda_2 of M satisfies lambda_2 <= [n/(n-1)] min m_ii."
- 3.6: "For the complete graph K_n ... a(K_n) = n."
- The Lemma holds for any weighted Laplacian. The diagonal of the weighted circulant Laplacian is sum_j y_j.
- Hence Gamma_1(p) <= p/(p-1) sum_j y_j, with equality for uniform weights (the complete graph). This is exactly prop:coverage at s=1 composed with the inequality of cor:dense.
- (The uniqueness part of cor:dense is not in Fiedler.)

**Sun, Boyd, Xiao, Diaconis (2006): the same problem at s=1, and the same homogeneity move.**
- J. Sun, S. Boyd, L. Xiao, P. Diaconis, "The fastest mixing Markov process on a graph and a connection to a maximum variance unfolding problem", SIAM Review 48(4):681-699, 2006. DOI 10.1137/S0036144504443821 (read in the PDF).
- Pages: https://web.stanford.edu/~boyd/papers/fmmp.html and https://web.stanford.edu/~boyd/papers/pdf/fmmp.pdf.
- Problem (1): "maximize lambda_2(w) subject to sum d_ij^2 w_ij <= 1, w >= 0". They note: "Since lambda_2 is a positive homogeneous function of w, it can be made as large as we like by scaling w ... the inequality constraint always holds with equality at the solution".
- Sec. 3.2 (eq. 2) rescales to "minimize sum d_ij^2 w_ij subject to lambda_2(w) >= 1". That is the same rescaling as the paper's "Reduction to P_p" (appendix.tex line 147), Gamma_s = (min phi)^{-s}.
- **Gamma_1(p) is the FMMP on K_p with unit costs.** By symmetry the optimum is circulant and uniform.
- Gamma_s for s>1 replaces the linear budget with the concave sum w^{1/s}. That problem is non-convex and is not FMMP.
- Other checks:
  - I confirmed the related Boyd, Diaconis, Parrilo, Xiao paper ("Fastest mixing Markov chain on graphs with symmetries", SIAM J. Optim. 20(2):792-819, 2009; arXiv 0709.0955). It is discrete-time and minimises SLEM, so it is not the same objective. No bib needed.
  - FMMP Sec. 5.3 treats cycles C_n (circulants with a single generator).

### B. Nothing found for the concave (l_q, q<1) budget
I found nothing on maximising lambda_2 or cosine coverage under a concave or quasi-norm budget. The search engines I could use were thin: WebSearch was exhausted (200/200), and the S2 and arXiv APIs returned 429. I used OpenAlex, the StackExchange API (MathOverflow and Math.SE) and direct fetches.
- OpenAlex queries, all with no relevant hit:
  - "maximize algebraic connectivity weighted circulant graph"
  - "algebraic connectivity maximization concave cost edge weights"
  - "maximizing algebraic connectivity sparse edge weights lp quasi-norm"
  - "fastest mixing random walk Cayley graph abelian group optimal generator weights"
  - "fastest mixing Markov chain circulant"
  - "optimal weights spectral gap Cayley graph cyclic group"
  - "max-min cosine sum nonnegative coefficients sparse"
  - "minimize peak sidelobe nonnegative weights sparse frequencies lp quasi-norm"
  - "maximum margin Fourier coefficients quasi-norm homogeneous network"
  - "worst-case separation modular addition amplitude budget"
- OpenAlex did surface Gunasekar et al. 1806.00468. That is the homogeneity to l_{2/L} Fourier quasi-norm template, handled under R4; I did not re-verify it.
- MathOverflow / Math.SE queries: "sum of 1-cos maximize minimum", "cosine sum minimum", "circulant spectral gap", "Cayley graph spectral gap generators log", "epsilon-biased set Z_p", "weighted Cayley graph eigenvalue", "maximize smallest eigenvalue circulant", "Chowla cosine".
  - Only adjacent, non-overlapping hits came back.
  - MO 209071 "Maximal minimum for a sum of two (or more) cosines": https://mathoverflow.net/questions/209071. It maximises the global minimum of the sum of cos(r_i x) over R.
  - MO 209236 "Why hexagons?": https://mathoverflow.net/questions/209236. It maximises the minimum over R^2 of a weighted cosine sum under sum a_k^2 = 1, with a conjectured sparse (3-frequency) optimum.
  - Both are Chowla-type problems: they raise the trough of the sum. Ours lowers the largest off-origin value of a positive-coefficient cosine sum, a sidelobe/spectral-gap problem. They are different extremal problems. No citation.

## Recommended change
The proposition itself is fine. It is correct and needs no proof change. Add attribution at two places.

1. After prop:coverage (one sentence):
   "Since $F_y(\delta)=\sum_jy_j\cov{j}{\delta}$ is the $\delta$-th Laplacian eigenvalue of the weighted circulant graph on $\sZ_p$ with weight $y_j/2$ on $\pm j$, $\Gamma_s(p)$ is its largest spectral gap under an $\ell_{1/s}$ weight budget; at $s=1$ this is the fastest-mixing problem of \citet{sun2006fastest} restricted to circulant rates, and the rescaling above is theirs. The $s=1$ bound also appears, for idealised cosine logits under an $\ell_2$ budget, in concurrent work \citep[App.~A.3.2, Eq.~18]{kam2026symmetry}."
2. At cor:dense (or app:dense), cite Fiedler for the inequality:
   "The inequality is Fiedler's bound $\lambda_2\le\frac n{n-1}\min_i L_{ii}$ \citep[\S3.5]{fiedler1973algebraic} applied to this circulant Laplacian; equality at the complete graph is \citet[\S3.6]{fiedler1973algebraic}."
   The uniqueness proof via Fourier inversion is still the paper's own.

Do not present prop:coverage as a contribution. It is a normalisation. The contribution is the s>1 programme and what is proved about it.

## Bib entries (all from pages fetched this session)
@article{fiedler1973algebraic,
  title={Algebraic connectivity of graphs},
  author={Fiedler, Miroslav},
  journal={Czechoslovak Mathematical Journal},
  volume={23},
  number={2},
  pages={298--305},
  year={1973}
}
@article{sun2006fastest,
  title={The Fastest Mixing {M}arkov Process on a Graph and a Connection to a Maximum Variance Unfolding Problem},
  author={Sun, Jun and Boyd, Stephen and Xiao, Lin and Diaconis, Persi},
  journal={SIAM Review},
  volume={48},
  number={4},
  pages={681--699},
  year={2006},
  doi={10.1137/S0036144504443821}
}
@article{kam2026symmetry,
  title={Symmetry without a manifold: intrinsic dimension on orbits},
  author={Kam, Chon-Fai and Bessafi, Miloud and Cadet, Fr{\'e}d{\'e}ric},
  journal={arXiv preprint arXiv:2609.17926},
  year={2026}
}
(morwani2024feature is already in refs.bib and correct.)
