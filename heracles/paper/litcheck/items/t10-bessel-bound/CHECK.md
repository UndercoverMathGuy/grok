# T10-bessel-bound (our_technique): verdict KNOWN (standard, elementary)

Location: sections/appendix.tex lines 198-214 ("A positive-Fourier inequality" + "A scalar bound").

## What the paper uses
1. I_0(z) = (1/pi) int_0^pi e^{z cos th} dth  -> DLMF 10.32.1 (verbatim: "I0(z)=1/pi int_0^pi e^{±z cos th} dth = 1/pi int_0^pi cosh(z cos th) dth").
2. Fourier coefficients of e^{z cos th} are I_k(z) >= 0 for z >= 0 -> DLMF 10.35.2 (Jacobi-Anger:
   "e^{z cos th} = I_0(z) + 2 sum_{k>=1} I_k(z) cos(k th)"); nonnegativity from the power series DLMF 10.25.2.
3. Lower bound I_0(z) >= e^{z-1/2}/(pi sqrt z) for z >= 1: truncate the integral to th <= z^{-1/2}, use cos th >= 1 - th^2/2.
   Elementary Laplace-method estimate; checked correct (proof is two lines; numerically ratio I_0 / bound ~ 2.07 as z -> inf,
   z=1: 1.266 vs 0.525). It is a factor ~ sqrt(pi/2)e^{1/2} ~ 2.07 weaker than the leading asymptotic e^z/sqrt(2 pi z)
   (DLMF 10.40.1, "I_nu(z) ~ e^z/(2 pi z)^{1/2} sum (-1)^k a_k(nu)/z^k"), which is fine since only psi(z) <= (1/2)ln z + O(1) is needed.
   The z <= 1 case (I_0 >= 1) and z >= 1 case together are valid for all z >= 0 (note the bound itself would FAIL for small z, e.g. z = 0.01, but the paper only uses it at z >= 1 — correct).
4. Sharper known inequalities exist (Baricz, Soni-type monotonicity of e^{-x} x^{-nu}... I_nu) — not needed; not fetched in full, so not cited.

## Novelty
None claimed/needed: standard special-function facts. The positive-Fourier averaging trick (only nonnegative resonant products survive) is also standard (cf. Alon-Roichman already cited nearby; the Chernoff/moment-generating style). Not the subject of this item.

## Recommendation
Add one citation to DLMF for 10.32.1 and 10.35.2 (and optionally 10.40.1 as a sanity remark on the constant). Suggested text edit at line 198:
"... $I_0(z)=\frac1\pi\int_0^\pi e^{z\cos\vartheta}d\vartheta$ \citep[10.32.1]{dlmf} ... the Fourier coefficients of $\vartheta\mapsto e^{z\cos\vartheta}$ are the modified Bessel values $I_k(z)\ge0$ \citep[10.35.2, 10.25.2]{dlmf}."
At line 214 optionally append: "(the sharp constant is $1/\sqrt{2\pi}$, \citealp[10.40.1]{dlmf}; we only need the order $\frac12\ln z$)."

Sources fetched: https://dlmf.nist.gov/10.32 , https://dlmf.nist.gov/10.35 , https://dlmf.nist.gov/10.40
