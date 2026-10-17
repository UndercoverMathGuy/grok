# One-dimensional mechanism (mechanism_note.pdf)

Field model of the 1-D shared-sign family: `oned.py` (efficiency of a parameter vector `(lam, logV, argV)`).
Exact LP for the weights at fixed slopes: `elim.py` (`lp`), backward elimination over a slope dictionary: `elim2.py`.
Slope search (CMA-ES around the LP): `slopes.py`; continuation in n with insertion/deletion moves: `contsl.py`.
Cross-feature (frequency 1+2 outputs) LP at p=5: `joint.py`, `jointcma.py`, `jtrend.py`.
Release of a 1-D solution into the full slot-symmetric class: `release.py`.  Mixed sign patterns: `mixed.py`.
Earlier exploratory variants: `oned3.py`, `oned5.py`, `rw.py`, `cont1d.py`.

Rigorous verification: `../certify_1d.py params.npy p n` (exact rational margin from the logits, interval ratio).
Certified parameters: `../oned_certified/p3_n*.json` (keys lam, logV, argV; field Phi(tau) = e^{i pi tau/p} sum V e^{lam tau}).
Typical run: `python slopes.py 3 11 2 6` then `python contsl.py 3 11 20 2 sl_p3_n11_m8.npy`.
