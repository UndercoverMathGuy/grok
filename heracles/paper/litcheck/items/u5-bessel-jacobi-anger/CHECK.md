# U5-bessel-jacobi-anger — CITATION_FOUND

Location: sections/appendix.tex line 198 (positive-Fourier inequality, app:scaling).
Statement: I_0(z) = (1/pi) int_0^pi e^{z cos th} dth; for z>=0 the Fourier coefficients of e^{z cos th} are I_k(z) >= 0.

Sources fetched (2026-09-25):
- DLMF §10.35 "Generating Function and Associated Series", https://dlmf.nist.gov/10.35
  Eq. 10.35.2: e^{z cos th} = I_0(z) + 2 sum_{k>=1} I_k(z) cos(k th)   (Jacobi-Anger, modified form)
- DLMF §10.32 integral representations, https://dlmf.nist.gov/10.32
  Eq. 10.32.1: I_0(z) = (1/pi) int_0^pi e^{±z cos th} dth  (matches the paper's definition exactly)
  Eq. 10.32.3: I_n(z) = (1/pi) int_0^pi e^{z cos th} cos(n th) dth
- DLMF §10.25, https://dlmf.nist.gov/10.25
  Eq. 10.25.2: I_nu(z) = (z/2)^nu sum_k (z^2/4)^k / (k! Gamma(nu+k+1)) -> every term >= 0 for z>=0, integer nu>=0, so I_k(z)>=0.

Recommended: "... are the modified Bessel values $I_k(z)\ge0$ \citep[Eqs.~10.32.1, 10.35.2, 10.25.2]{dlmf}."
Nothing in refs.bib currently cites DLMF; add entry below. Background fact, no novelty issue.

@misc{dlmf,
  key          = {DLMF},
  title        = {{NIST Digital Library of Mathematical Functions}},
  editor       = {Olver, F. W. J. and Olde Daalhuis, A. B. and Lozier, D. W. and Schneider, B. I. and Boisvert, R. F. and Clark, C. W. and Miller, B. R. and Saunders, B. V. and Cohl, H. S. and McClain, M. A.},
  howpublished = {\url{https://dlmf.nist.gov/}},
  note         = {Chapter 10 (Bessel Functions), \S\S 10.25, 10.32, 10.35}
}
(Release number/date not captured from the fetched page; add current release, e.g. "Release 1.2.x of <date>", after checking dlmf.nist.gov front page.)
