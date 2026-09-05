# Margin redo (2026-09-05): trained networks vs the provable class ceiling

Replaces the round-3 comparison in nudial_results.md ("AdamW beats direct
margin ascent 7-20x"), which is RETRACTED: cohorts D/H/J were stuck local
optima of a non-convex ascent from random init and say nothing about
optimality; round 8 (paired fixed-support) had already shown the optimum
is K*. No random-init margin ascent appears anywhere below.

Instrument: gamma = min over the full p^2 grid of (correct logit - best
wrong logit) / ||theta||_2^nu, float32 on GPU (TF32 off), exact minimum.
Ceiling = constructive phase-complete single-frequency circuits at each K
with optimal block balance (construct_gamma.json gamma_free; the f64
monotone polish cannot improve them, construct_polish.json). relu4: max
at K=4, 6.39e-6 (K=5: 6.21e-6). relu6: max at K=3, 6.22e-8 (K=4: 5.91e-8).
Polish = verified-monotone projected ascent on the norm sphere (softmin
direction, a step accepted only if the exact minimum margin rises by
> 1e-6 relative), 400 steps. Cohorts M (AdamW wd 1, 10k ep) and MG (GD
lr 0.3 wd 3e-4, full data, 20k ep), 32 seeds each, m=512, p=113. Script:
nudial/margin_redo.py -> notes/margin_redo.json, pod log margin_redo_pod.log.

Polish v2 (nudial/margin_redo.py, 2026-09-05 evening): four ascent
directions per step (softmin at sharpness 8/200/5000 x worst-decile
scale + hard-argmin gradient), 3-point line search, exact-min acceptance,
200 steps, float32. Positive control (notes/margin_control.json): from
AdamW nets perturbed by 10% relative noise (gamma -9.6%) it recovers 47%
of the loss in 150 steps; from 3% noise it ends ABOVE the unperturbed net
(+1.6%); f64 vs f32 agree (f64 +2.1% on the same 8 nets). The v1 polish
(single worst-decile softmin direction, margin_polish.py) accepted nothing
even from perturbed nets and is retired; v1 numbers in margin_redo_v1.json.

arm                 n   gamma median (IQR)      polish gain med (IQR, max)   /ceiling(K*) after  /ceiling(own K)  K mode (hist)
AdamW f0.5 relu4   32   4.96e-6 (4.73-5.08)     2.1% (1.7-2.5, 4.0)          0.793               0.854            5 {4:2,5:16,6:11,7:3}
AdamW f1.0 relu4   32   5.10e-6 (4.86-5.29)     0.8% (0.6-1.1, 1.8)          0.805               0.857            5 {4:4,5:16,6:11,7:1}
GD    f1.0 relu4   32   2.06e-6 (1.76-2.74)     5.4% (4.0-7.4, 16.7)         0.341               0.481            3 {1:2,2:7,3:18,4:5}
AdamW f0.5 relu6   32   4.74e-8 (3.91-4.93)     1.7% (1.3-1.9, 6.0)          0.773               0.830            4 {3:5,4:18,5:9}
AdamW f1.0 relu6   32   4.59e-8 (4.27-4.86)     0.6% (0.5-0.8, 1.4)          0.743               0.779            4 {3:6,4:25,5:1}
GD    f1.0 relu6   26*  2.82e-8 (2.56-3.03)     2.3% (1.1-3.6, 10.5)         0.462               0.579            2 {1:2,2:15,3:8,4:1}
(* 6/32 GD relu6 seeds diverged to NaN at lr 0.3 and are excluded.)

Reading:
1. AdamW+wd endpoints sit at 74-80% of the provable class ceiling
   (within 1.35x) and are NEAR-stationary: a verified ascent that passes
   its positive control gains only 0.6-2.1% (median) from them, and the
   committee does not change under polish (K histograms identical before
   and after). Not exact stationary points — "stationary" must not be
   used; "within 2% of a local maximum of the normalised margin" is the
   supportable sentence.
2. Plain GD+l2 endpoints sit at 32-45% of the ceiling, with fewer
   frequencies than K* (mode 3 at K*=4, mode 2 at K*=3) and a hierarchical
   profile (k_eff 1.5); they are not stationary (single accepted steps).
   GD nets gain 2-5% (max 17%) under the same polish: further from a
   local max. Same ordering as round 3 but against a fixed ceiling, not a
   broken maximiser: AdamW 1.7-2.4x GD in normalised margin.
3. The K*+1 AdamW committees are BELOW the K* construction (0.78) and
   below the K*+1 construction too (0.84): the +1 is not "the better
   optimum" (round-3 claim, withdrawn); it is a stationary point on the
   plateau that the dynamics parks on (P-N12, eps-crossover note).
4. Gap to ceiling is 20-26% — the same number as the plateau law's
   "within 26%": trained nets realise the plateau, not the peak.
