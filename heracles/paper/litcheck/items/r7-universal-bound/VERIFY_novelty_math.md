# VERIFY (novelty_math) — R7 universal bound (approach.tex thm:universal)

Claim: for any one-hidden-layer nu-homogeneous network, any width, 2<=b<=nu:
gamma*_b <= C_b^{nu/b} Gamma_{nu/b}(p).
Proof = (i) phase/averaging lemma h <= sum_j A_j cov(j,delta); (ii) coverage programme with
l_{1/s} budget; (iii) single-neuron pricing sum_j |beta_j|^{b/nu} <= C_b ||omega||^b (AM-GM);
(iv) subadditivity of t^r, r=b/nu<=1, across neurons.

## Operational note (important)
WebSearch budget was EXHAUSTED at the first call of this task ("200 of 200 WebSearch calls").
Every later verifier in this workflow session will also get no search results. My independent
adversarial search was therefore limited to WebFetch on primary sources whose URLs I already knew.

## Missed prior art (technique) — fetched and read this session
Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional
Networks", NeurIPS 2018 (arXiv 1806.00468; PDF header: "32nd Conference on Neural Information
Processing Systems (NeurIPS 2018)").
- Lemma 6: depth-L diagonal network, R(beta) = min_{P_diag(w)=beta} ||w||_2^2 = L ||beta||_{2/L}^{2/L}.
- Lemma 7: same for full-width linear conv nets with beta replaced by its DFT (quasi-norm in the
  Fourier domain).
- Thm 2a: GD limit direction is a first-order stationary point of min ||beta_hat||_{2/L} s.t.
  y_n <beta, x_n> >= 1 (a max-margin problem).
- Thm 4: homogeneous polynomial parameterisations (degree nu) -> stationary point of
  min ||w||_2^2 s.t. margin >= 1.
This is exactly the pricing step at b=2 (amplitude A of a nu-fold product costs nu*A^{2/nu}),
in the Fourier domain, inside a max-margin problem. It is the canonical ML source and predates
Hoff/Dai/Fan as recommended by the checker. The general-b case is the same AM-GM.
So: the proof technique of thm:universal = Morwani-style max-min over amplitudes + the
Gunasekar/Dai representation cost + the r-triangle inequality of l_r quasi-norms (textbook).

## Checker's confirmed citations — re-verified
- morwani2024feature (arXiv 2311.07568 HTML): Thm 7 max L_{2,3} margin sqrt(2/27)/(p^{1/2}(p-1)),
  width m >= 4(p-1). Confirmed. Fig 4b ReLU+L2 "nearly 1-sparse ... but not quite". Confirmed.
- li2025fourier (PMLR 258:523-531): title/authors/pages confirmed; max L_{2,k+1} margin with
  m >= 2^{2k-2}(p-1). Confirmed.
- hoff2017lasso (arXiv 1611.00040): title, author, sentence "a subclass of L_q penalties with
  q<=1 can be expressed as sums of L_2 penalties" confirmed. Journal ref (CSDA) NOT confirmed
  (doi.org/10.1016/j.csda.2017.06.007 resolves to Elsevier pii S0167947317301469; page 403).
  Keep the arXiv bib entry as given.
- dai2021representation (NeurIPS 34, proceedings page + PDF): confirmed; PDF table lists
  "Diagonal Network ... l_{2/d} quasi-norm" and "quasi-norm on Fourier domain" for CNNs.
- fan2023euclidean (arXiv 2012.03436): TMLR Jan 2023, authors confirmed; "monotonic
  transformations of tensor Schatten-p quasi-norm" confirmed.
- kam2026symmetry (arXiv 2609.17926, submitted 2026-09-15): exists; abstract is about intrinsic
  dimension on orbits (modular addition). HTML App. A.3.2: Eq 15 Delta_S(beta) =
  min_{s!=0} sum_{k in S} beta_k (1-cos(omega_k s)) under ||beta||_2 = B for the quadratic net;
  Prop 5: uniform allocation beta_k = B/sqrt(Kmax) optimal. Checker's summary confirmed.

## Verdict
PARTIALLY_KNOWN upheld. b=nu endpoint known (Morwani Thm 7, Li et al. 2025); every proof
ingredient is known (pricing = Gunasekar et al. 2018 Lemmas 6-7 / Dai et al. 2021; quasi-norm
subadditivity textbook); the stated bound for b<nu on modular addition, uniform over width and
neuron type, was not found. Caveat: novelty for b<nu rests on the checker's searches plus my
targeted fetches; no fresh math-literature search was possible (search budget exhausted).

## Suggested edits
- Add gunasekar2018implicit and cite it FIRST for the pricing step ("as for depth-L diagonal and
  convolutional linear networks, where l2 on the factors induces the l_{2/L} quasi-norm of the
  Fourier coefficients \citep[Lemmas 6--7]{gunasekar2018implicit}; see also dai2021..."). Keep
  Hoff/Fan optional.
- In the checker's recommended sentence, say "at b=2" for the l2-factor fact (Hoff/Gunasekar/Dai
  are all b=2); general b is the same AM-GM.
