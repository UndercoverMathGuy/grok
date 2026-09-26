# VERIFY (novelty_math) — T10-bessel-bound — task 60

Verdict: checker's KNOWN upheld. This is a standard special-function fact with an elementary Laplace-type truncation. The paper makes no novelty claim and needs none.

## Independent checks
- Derivation (appendix.tex:210-214, with a_s := 3+s/2 from cyclic.tex:112 and r = 1/s < 1):
  - z<=1: psi <= z. ln(1+z^r) >= z^r/2 >= z/2 uses r<=1 and ln(1+x) >= x/2 on [0,1]. a_s >= 2 holds. OK.
  - z>=1: truncating at theta <= z^{-1/2} with cos >= 1 - theta^2/2 gives I_0(z) >= (1/pi) z^{-1/2} e^{z-1/2}. OK.
    Then ln pi + 1/2 = 1.645 < 3 ln 2 = 2.079. Also (s/2) ln(1+z^r) >= (s r/2) ln z = (1/2) ln z, since s r = 1. So psi <= a_s ln(1+z^r). OK.
- Numerics (scipy i0): I_0 / bound = 2.41 (z=1), 2.26 (z=2), 2.13 (z=5), 2.07 (z=100), tending to sqrt(pi/2) e^{1/2} = 2.066. The bound is valid on z>=1 and loose by a constant, which is harmless because only the order (1/2) ln z is used.

## Citations the checker confirmed, re-fetched this session
- https://dlmf.nist.gov/10.32 — 10.32.1: "I0(z) = 1/pi int_0^pi e^{±z cos th} dth = ..." CONFIRMED.
- https://dlmf.nist.gov/10.35 — 10.35.2 (Jacobi–Anger): "e^{z cos th} = I_0(z) + 2 sum_{k>=1} I_k(z) cos(k th)", for z, th in C. CONFIRMED.
- https://dlmf.nist.gov/10.25 — 10.25.2 power series (all terms >= 0 for z >= 0, nu = k integer >= 0), giving I_k(z) >= 0. CONFIRMED.
- https://dlmf.nist.gov/10.40 — 10.40.1: I_nu(z) ~ e^z/(2 pi z)^{1/2} sum (-1)^k a_k(nu)/z^k. CONFIRMED (the sharp constant 1/sqrt(2 pi)).
  - 10.40(iii) gives explicit error bounds only for K_nu, not I_nu. Do not cite DLMF for a *rigorous* I_0 lower bound. The paper's own two-line bound is the right way to go.
- https://dlmf.nist.gov/10.37 — I_nu(x) positive and increasing in x for nu >= 0. There is no e^x/sqrt(x) comparison there.
- https://dlmf.nist.gov/help/cite — the official BibTeX includes "Release 1.2.8 of 2026-09-15" in howpublished. The checker's bib entry omits the release. Add it: DLMF asks for the release, and the equation numbers are release-stable, but the release should still be stated.

## Adversarial novelty search (math lens)
- I_0(z) >= c e^z / sqrt z (z >= 1) is the standard Laplace-method lower bound. Sharper forms (monotonicity of sqrt(x) e^{-x} I_nu(x), Baricz/Laforgia–Natalini-type bounds) exist in the special-function inequality literature. I could not fetch a primary source this session: Springer paywall/redirect on Laforgia–Natalini JIA 2010:253035, arXiv API returned HTTP 429, and the WebSearch session budget was exhausted (200/200). So no extra citation is recommended. None is needed, since the proof is self-contained.
- The combination psi(z) <= a_s ln(1+z^r) is a bespoke scalar inequality with an ad hoc constant. It is not a named known result, and it does not need to be one.

## Recommendation
Keep the checker's change: cite DLMF Eq. 10.32.1, 10.35.2 (+10.25.2 for I_k >= 0), and optionally 10.40.1 for the sharp constant. Fix the bib entry to include "Release 1.2.8 of 2026-09-15" (or whichever release is consulted at camera-ready).

## Infra note (relevant to workflow slowness)
This verifier hit two limits. The WebSearch per-session budget was exhausted ("200 of 200 WebSearch calls"), and export.arxiv.org returned HTTP 429. Parallel verifiers are sharing and exhausting search/API quotas, and that likely stalls or slows the verify stage.
