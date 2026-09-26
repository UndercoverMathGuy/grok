# VERIFY (novelty_ml): T10-bessel-bound, task 59

**Verdict: upheld. KNOWN (standard special-function fact). No novelty is claimed and none is needed.**

## What the paper does (appendix.tex 198-214)
- It defines I_0 by its integral representation.
- The Fourier coefficients of e^{z cos th} are I_k(z) >= 0.
- It proves psi(z) = z - ln I_0(z) <= a_s ln(1+z^r), with a_s = 3 + s/2 (cyclic.tex:112).
- It uses the elementary cutoff bound I_0(z) >= e^{z-1/2}/(pi sqrt z) for z >= 1, and I_0 >= 1 for z <= 1.

## Independent checks (this session)
- **DLMF 10.32.1** (fetched): I_0(z) = (1/pi) int_0^pi e^{+-z cos th} dth. Matches line 198.
- **DLMF 10.35.2** (fetched): Jacobi-Anger, e^{z cos th} = I_0(z) + 2 sum_{k>=1} I_k(z) cos(k th). This supports the "Fourier coefficients are I_k" statement.
- **DLMF 10.40.1** (fetched): I_nu(z) ~ e^z/(2 pi z)^{1/2} sum (-1)^k a_k(nu)/z^k. This supports the optional sharp-constant remark.
- **Derivation re-done:**
  - For z >= 1, the cutoff z^{-1/2} <= 1 < pi.
  - On [0, z^{-1/2}], z cos th >= z - 1/2.
  - So I_0 >= z^{-1/2} e^{z-1/2}/pi. Correct.
- **Numerical check (scipy i0e):** ln I_0 - ln(bound) = 0.88 at z=1, 0.75 at z=5, and tends to 0.7258 (ratio about 2.07) up to z=1e6. The bound holds everywhere tested.
- **Constant chain:** ln pi + 1/2 = 1.645 < 3 ln 2 = 2.079, and (s/2) ln(1+z^r) >= (s/2) r ln z = (1/2) ln z. This is consistent with a_s = 3 + s/2 >= 2.
- **DLMF citation page** (https://dlmf.nist.gov/help/cite, fetched):
  - The editor list in the checker's bib is exact.
  - The bib is missing the release. The current release is "Release 1.2.8 of 2026-09-15".
  - DLMF recommends putting the release in howpublished and bracing {Olde Daalhuis}.
- **Current refs.bib:** it has no DLMF entry, so this entry must be added.

## ML-lens novelty search
- Bounds on log I_0 / log I_nu are routine in ML, for example in von Mises-Fisher log-normalizer approximations.
- Nothing in the ML literature could make this scalar bound novel. It is a textbook Laplace-method estimate.
- I could not fetch or confirm an ML paper that uses exactly psi(z) = z - ln I_0(z) <= O(ln z) in a margin or coverage argument, so I cite none.
- The interesting part is the averaging of e^{-tF_y} over delta, lower-bounded by a product of I_0 (eq. posfourier). That is the separate item t9-positive-fourier, not this one.
- Baricz, "Bounds for modified Bessel functions of the first and second kinds", Proc. Edinburgh Math. Soc. 53 (2010) 575-599, doi:10.1017/S0013091508001016.
  - Its existence is confirmed via Crossref, and I fetched the abstract.
  - The abstract does not state the specific e^x/sqrt(2 pi x) lower bound, so I do NOT recommend it as a citation for that inequality.

## Tooling problems hit (relevant to workflow slowness)
- WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)" on the first query. Every verifier from here on gets no WebSearch.
- export.arxiv.org API and api.semanticscholar.org both returned HTTP 429 Too Many Requests. Parallel verifiers are rate-limited.
- Only Crossref and direct page fetches (DLMF, cambridge.org) worked.

## Recommended change
- Adopt the checker's line-198 / line-214 edits, citing [Eq. 10.32.1], [Eq. 10.35.2] and optionally [Eq. 10.40.1] of DLMF.
- Use the bib entry below, which includes the release.
