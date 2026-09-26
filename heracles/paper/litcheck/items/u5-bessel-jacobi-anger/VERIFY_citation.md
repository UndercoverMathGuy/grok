# VERIFY (citation lens): U5-bessel-jacobi-anger — task 93

Claim (sections/appendix.tex, "A positive-Fourier inequality", ~l.198): for z >= 0 the Fourier
coefficients of theta -> e^{z cos theta} are the modified Bessel values I_k(z) >= 0; I_0 defined as
(1/pi) int_0^pi e^{z cos theta} dtheta. refs.bib has no DLMF / A&S / Watson entry (grep confirmed).

## Pages fetched this session (2026-09-25)
- https://dlmf.nist.gov/10.35 — §10.35 "Generating Function and Associated Series".
  10.35.1: e^{(z/2)(t+1/t)} = sum_{m in Z} t^m I_m(z).  10.35.2: e^{z cos theta} = I_0(z) + 2 sum_{k>=1} I_k(z) cos(k theta).
  Page footer: Version 1.2.8, release date 2026-09-15.
- https://dlmf.nist.gov/10.32 — §10.32(i). 10.32.1: I_0(z) = (1/pi) int_0^pi e^{±z cos theta} dtheta (exactly the paper's definition).
  10.32.3: I_n(z) = (1/pi) int_0^pi e^{z cos theta} cos(n theta) dtheta.
- https://dlmf.nist.gov/10.25 — §10.25(ii). 10.25.2: I_nu(z) = (z/2)^nu sum_k (z^2/4)^k / (k! Gamma(nu+k+1)); every term >= 0 for z >= 0, integer nu >= 0, so I_k(z) >= 0.
- https://dlmf.nist.gov/help/cite — official citation: "NIST Digital Library of Mathematical Functions.
  https://dlmf.nist.gov/, Release 1.2.8 of 2026-09-15. F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier,
  B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, eds."
  Equations are cited as [DLMF, (x.y.z)] with permalinks https://dlmf.nist.gov/x.y.Ez.

## Verdict
Checker upheld. All three equation numbers exist and say what is claimed, in the right direction.
The editor list the checker could not confirm is confirmed verbatim by the /help/cite page (same 10
names, same order). Release to use: 1.2.8 of 2026-09-15.

Minor precision notes (no error):
- In cosine form (10.35.2) the coefficient of cos(k theta) is 2 I_k(z) for k >= 1; in complex-exponential
  form (10.35.1 with t = e^{i theta}) the coefficient of e^{ik theta} is I_k(z) with I_{-k} = I_k. The paper's
  "Fourier coefficients are I_k(z)" is correct in the exponential convention, which is the one the
  argument uses (frequencies summing to 0 mod p); either way all coefficients are >= 0, which is all
  the proof needs.
- The paper applies it with z = t y_j >= 0, inside the stated range.

## Recommended bib entry (official DLMF BibTeX, key renamed)
@misc{dlmf,
  key          = {{\relax DLMF}},
  title        = {{\it NIST Digital Library of Mathematical Functions}},
  howpublished = {\url{https://dlmf.nist.gov/}, Release 1.2.8 of 2026-09-15},
  url          = {https://dlmf.nist.gov/},
  note         = {F.~W.~J. Olver, A.~B. {Olde Daalhuis}, D.~W. Lozier, B.~I. Schneider, R.~F. Boisvert, C.~W. Clark, B.~R. Miller, B.~V. Saunders, H.~S. Cohl, and M.~A. McClain, eds.}
}
Cite as \citep[(10.32.1), (10.35.2), (10.25.2)]{dlmf}. Re-check the release number at camera-ready
(DLMF releases roughly quarterly).
