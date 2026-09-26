# VERIFY (novelty_math): T12-dense-uniqueness

Claim (appendix app:dense, Cor. cor:dense): Gamma_1(p) = p/(p-1). The unique maximizer is flat, y_j = 2/(p-1). Proof: average over delta != 0, then Fourier inversion of the constant coverage.

Checker verdict: PARTIALLY_KNOWN. Citation: morwani2024feature (confirmed).
My verdict: KNOWN, both the technique and the conclusion. The checker's citation is confirmed. The checker's "new parts" list is too generous.

## 1. Morwani et al. 2311.07568v2: re-read from the arXiv HTML myself
- The abs page gives authors Morwani, Edelman, Oncescu, Zhao, Kakade. Comment: "Accepted as Spotlight at ICLR 2024". v1 is 2023-11-13 and v2 is 2024-02-19. The bib entry is correct.
- Sec 4 proof outline: "We will use the uniform class weighting: tau(a,b)[c'] := 1/(p-1) for all c' != a+b". The wrong class is c' = a+b-delta, so this is exactly our average over delta != 0.
- Lemma 6, second bullet: "all points in the support of q* are on the margin for any maximum margin solution".
- App F.3, "Proof that all frequencies are used": f = F(a,b) + lambda 1_{a+b=c}. The transform of f_1 vanishes for l != 0. The transform of f_2 is "lambda p^2 if j=k=-l". Their conclusion is "f^(j,k,l) > 0 ... at least one neuron for each frequency".
- Their computation gives more than they state. Every diagonal coefficient f^(j,j,-j), j != 0, equals the same lambda p^2. That coefficient is our G_j, and beta_j = 2p^{-nu} G_j. So *equal* network-level amplitude on every frequency already follows from their F.3 in one line. Their Theorem 7 states only "at least one neuron uses each frequency" and never says the amplitudes are equal. So "equal amplitudes" is not a new conclusion. At most it makes explicit something their computation already contains.

## 2. Mathematics lens: the lemma is Rankin's simplex bound and its equality case, for a cyclic code
Put v_delta := (sqrt(y_j) e^{2 pi i j delta/p})_{j in F_p}, with sum_j y_j = 1. Then:
- the v_delta are unit vectors;
- <v_delta, v_delta'> = zeta(delta - delta');
- the coverage sum_j y_j cov_j(delta) = 1 - zeta(delta) = ||v_delta - v_0||^2 / 2.

So Gamma_1 is the largest possible minimum distance (half its square) of a p-point cyclic spherical code, i.e. an orbit of Z_p with no trivial component.

- R. A. Rankin, "The closest packing of spherical caps in n dimensions", Proc. Glasgow Math. Assoc. 2(3):139-144, 1955, doi:10.1017/S2040618500033219. I fetched the Cambridge page and the PDF.
  - Theorem 1(ii) (via Lemma 2) is the simplex bound N(alpha) <= 2 sin^2 alpha / (2 sin^2 alpha - 1). For m points this says the maximum pairwise inner product is at least -1/(m-1).
  - Lemma 1 proves it by the same sum-of-squared-distances averaging we use.
  - Lemma 3 shows the regular simplex attains the bound.
  - With m = p this gives Gamma_1 <= 1 + 1/(p-1) = p/(p-1), which is exactly our first display.
- Equality case: all pairwise inner products equal and sum_delta v_delta = 0, i.e. a regular simplex. For a cyclic code, this means the circulant Gram matrix is (p/(p-1)) I - J/(p-1). Its DFT eigenvalues are p/(p-1) at k != 0 and 0 at k = 0, so y is flat. That is our "Fourier inversion" step. It is the standard fact that a circulant matrix is diagonalized by the DFT, the same as "two-level (ideal) autocorrelation <=> flat power spectrum" in sequence design.
- Strohmer & Heath, "Grassmannian frames with applications to coding and communication" (arXiv math/0301135; I fetched it). It treats Rankin's spherical-code bounds and the Welch bound together, and Cor. 2.5 says the regular simplex (N = m+1) is the unique optimum up to equivalence. It is a useful survey pointer; I do not recommend citing it.
- I checked the proof's arithmetic. sum_{delta != 0} cov_j(delta) = (p-1) - (-1) = p. From c0 + (1-c0)/p = 0 we get c0 = -1/(p-1), and then y_j = 2(1-c0)/p = 2/(p-1), which sums to 1 over the (p-1)/2 frequencies. All correct.

## Verdict
- KNOWN as mathematics: the value, the unique flat maximizer, and both proof steps (averaging, then circulant/Fourier inversion).
- Implicit in the ML paper: Morwani F.3 already gives equal diagonal coefficients lambda p^2.
- The only genuinely new ingredient is the reduction that makes Gamma_1 bound *every* network at b = nu (Thm universal). That belongs to a different item.

## Recommended wording (replaces the checker's sentence)
"The averaging step is the uniform class-weighting certificate of \citet{morwani2024feature} (Lemma 6), and equivalently the simplex bound of \citet{rankin1955closest} applied to the cyclic code $v_\delta=(\sqrt{y_j}\rho^{j\delta})_j$; the equality case (a regular simplex, whose circulant Gram matrix has flat spectrum) gives the flat maximizer, which is also implicit in App. F.3 of \citet{morwani2024feature}, where every diagonal Fourier coefficient equals $\lambda p^2$. What is new here is only that, through Theorem~\ref{thm:universal}, this pins the amplitudes of every network attaining the bound at $b=\nu$."

Do not say "stronger conclusion that the amplitudes are equal" relative to Morwani. Their computation already gives it.

## Harness note (for the orchestrator / user's slowness question)
In this verifier session, WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)" on the first call. The workflow's shared session has used up its WebSearch quota. From here on, verifiers can only WebFetch known URLs. That probably explains some of the slowness and weaker verify outputs: agents retry or route around the refused searches. Raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or give each verifier its own session.
