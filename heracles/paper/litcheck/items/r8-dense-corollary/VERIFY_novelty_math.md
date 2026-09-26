# VERIFY (novelty_math): R8-dense-corollary, task 6

Claim: Gamma_1(p) = p/(p-1), with unique flat maximizer y_j = 2/(p-1) over F = {1..(p-1)/2}. It recovers Morwani's maximum L_{2,3}-margin sqrt(2/27)/(p^{1/2}(p-1)).

Checker verdict: KNOWN. **Upheld.**

## What I checked myself
1. **Morwani et al. (arXiv 2311.07568v2).** I downloaded the PDF and ran pdftotext.
   - Theorem 7 (Sec. 4): m >= 4(p-1), gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1)). Every frequency in {1..(p-1)/2} is used by at least one neuron.
   - App. F.3 ("Proof that all frequencies are used"): by Lemma 6, f = F(a,b) + lambda*1[a+b=c]. So the diagonal DFT has the same value at every frequency, which makes flatness implicit.
   - The ICLR 2024 page (iclr.cc/virtual/2024/poster/18073) lists it as a Spotlight Poster with the same 5 authors.
2. **Li, Liang, Shi, Song, Zhou (PMLR v258).** I downloaded arXiv 2402.09469v4 and the PMLR PDF.
   - Theorem 4.1 (informal version of Thm G.2): m >= 2^{2k-1}(p-1)/2, max L_{2,k+1}-margin gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}), and every frequency is used.
   - The PMLR page gives vol 258, pp. 523-531, AISTATS 2025.
3. **Algebra.** n! * B_p * p/(p-1), with B_p = 2(2p nu)^{-nu/2} and nu = n+1, reduces to Li's gamma* with k = n. Numerically the ratio is 1 to 1e-16 for p in {5,7,113} and n in {2,3,4,5}. At n = 2 it equals Morwani's constant.
4. **LP check.** I solved the linear programme for Gamma_1 with scipy linprog at p = 5, 7, 113. The optimum is p/(p-1) and the maximizer is flat, y_j (p-1)/2 = 1.
5. **Proof audit (app:dense).** The averaging identity sum_{delta != 0} kappa_j(delta) = p is correct. The Fourier-inversion uniqueness step is correct: c0 = -1/(p-1) and y_j = 2/(p-1).

## Mathematics-lens finding: even the uniqueness part is classical
Set x_delta = (sqrt(y_j) cos(2 pi j delta/p), sqrt(y_j) sin(2 pi j delta/p))_{j in F}. Then:
- ||x_delta||^2 = sum_j y_j = 1;
- <x_0, x_delta> = zeta(delta);
- ||x_0 - x_delta||^2 = 2 sum_j y_j kappa_j(delta).

So Gamma_1 is the largest minimum squared distance (halved) of a p-point **cyclic group code** on the sphere (Slepian's group codes) with no DC component.

- The averaging bound is the **simplex (Rankin) bound**: ||sum_delta x_delta||^2 >= 0 gives max_{delta != 0} <x_0, x_delta> >= -1/(p-1).
- Equality holds iff the code is a regular simplex. A regular simplex has Gram matrix (p/(p-1)) I - (1/(p-1)) J. For a circulant Gram matrix, that forces equal eigenvalues (Fourier weights) on every non-zero frequency, which is the flat y.

So the Gamma_1 lemma, uniqueness included, is the classical fact that a cyclic (harmonic) code meets the simplex bound iff it uses all non-trivial characters with equal weight. The paper's proof is fine and elementary. It should not be presented as new mathematics. A one-line remark ("this is the simplex bound for cyclic group codes") would pre-empt a reviewer.

Sources for that remark:
- Rankin 1955, "The closest packing of spherical caps in n dimensions". I fetched the Cambridge page and confirmed the bibliographic fields (Glasgow Math. J., originally Proc. Glasgow Math. Assoc., 2(3):139-144, 1955). The theorem text was not visible, so the exact theorem number is unverified.
- Slepian 1968, group codes. The IEEE page returned empty, so this is **not** recommended as a citation.

## What survives as new
Only the framing: this dense endpoint is the s=1 member of a programme that is sparse for s > 1. The constant is Morwani's Theorem 7 (n=2) and Li et al.'s Theorem 4.1 (general n). Flatness is implicit in Morwani App. F.3.

## Bib fix
The li2025nsum entry lacks the PMLR fields. Add series = PMLR, volume = 258, pages = 523--531 (confirmed on proceedings.mlr.press/v258/li25b.html).

## Process note
WebSearch returned "200 of 200 WebSearch calls" used. The session-level search budget is exhausted, so this verifier used only WebFetch, direct PDF download and prior knowledge. That affects every remaining verify task in the workflow.
