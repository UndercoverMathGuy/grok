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

## Feature-count proof support (Section 9 of mechanism_note)
- `prog.py` — programme Γ_s(p) by reweighted LP; support K*, binding set, coverage sums at binding directions.
- `rel.py` — counts short signed additive relations in programme supports (relevant to hypothesis (D) of the Riesz theorem).
- `collect.py` — collective-undercutting test: every programme feature realised by a two-neuron sub-network, translations optimised.
- `c1.py`, `c1q.py` — single-feature constant c1(n,s) (bonus over the interference-free price per unit of sign-flipped coefficient).
- `quad/` — quadratic and Hermitian-square product tests (Theorem `thm:quad`, Corollary `cor:quad`):
  `gmax.py` (worst-case skewness γ_max(n)), `ksharp.py` (k#(n,s) with γ_max), `ksharp2.py` (direction-resolved quadratic constant),
  `kflat.py` (best degree-2 Hermitian-square constant k♭(n,s)), `checkquad.py` (checks the factorisation identity E[wg] = RHS on random
  class-C fields under (D3), (I2)), `outride.py` (output-side riding compensation of a two-neuron cheap feature, n=2).
- `relations2/` — the relations case at n=2 (all interference on T={1,2}): oscillation/Parseval closure, sharp single-T
  payment for sign atoms (`c12.py`), phase-free water-filling criterion (`cl2f.py`, `cl2g.py`), one-sided constants of programme
  supports (`sigsupp.py`), write-up `result.tex`. Run from inside `relations2/`.
- `freehide.py` — cost-free compensation game for one cheap feature (n=2); `quadbonus.py` — sup of bonus*|A|/sum|beta_T|^2;
  `spectral_cost_check.py` — random check of the spectral cost inequality.
- `compensation/` — the single-cheap-feature compensation game at n=2 (protection curves, Chebotarev dichotomy, dense case,
  local bonus expansion, two-atom branch): write-up `result.tex`, per-feature verdicts `all15.txt`, `all12.txt`.
- `kkt/` — KKT multipliers of local maximisers (`kkt.py`), second-order check (`sosc.py`), the factorisation identity (`ident.py`);
  `p5n6_pol.npz` is the p=5, n=6, b=2 local maximiser with a cheap and a spurious feature. Run from inside `kkt/`.
- `relations3/` — efficiency–Sidon reduction, tilted-mean bound, relation-mass (RM) checks and large-p certificates (p up to 30011),
  large-deviation model; write-up `result.tex`, `numerics.tex`, `gap.tex`.
- `payment/` — proofs of the sharp n=2 single-feature payments: P12 (analytic) and c1(2,s)=1-3^{1-s} (interval-arithmetic
  certificate `t2_certify.py`, re-run output `t2_rerun.out`); random checks `check_chain.py`, `check_chain_t2.py`; write-up `ideaB_proofs.tex`.
- `falsify/` — counterexample searches at large p in the relations model (`relmodel.py`, drivers `run*.py`, summary `compile.py`),
  flatness and Parseval floor (`flat.py`, `parseval.py`), near-b=3 runs (`nearthree.py`, `dense2.py`), exact-margin checks (`netcheck.py`, `task3.py`),
  p=5 grouped-slot searches (`famp_g.py`, `run4.py`). Run from inside `falsify/`.
- `nearif/` — quadratic payments and near-IF rigidity at n=2 (Proposition `prop:quadpay`, Theorem `thm:nearif`):
  `cert_kq.py` (interval certificate for g(d) >= 0.9*(5+4r)/18*d^2, hence K_q = 5/(5+4r); output `cert_kq.out`),
  `quadcheck.py` (adversarial suprema of bonus*L/E and bonus*L/|D|^2; output `quadcheck.out`; the "claimed Kq" printed there is the second-order limit 9/(10+8r), the certified constant is 5/(5+4r)),
  `quadp12.py` (evidence that the relations-case quadratic constant is exactly c12, two aligned atoms; output `quadp12.out`),
  `nearif_check.py` (spot-check of the bound of Theorem `thm:nearif`(a) on 3000 random relations-case networks; output `nearif_check.out`).
