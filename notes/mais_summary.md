# MAIS registry pivot — interim ledger (2026-08-21)

Target list: https://github.com/lionellevine/MAIS (Lionel Levine, Cornell). 92 crisp,
ID'd open problems; resolutions posted as GitHub issues are recorded in days
(O60 killed by arXiv 2608.04451, Aug 5; O62 progress by "Opus and Iris Shi").
Problems on our turf: O52 (how many frequencies / sparse vs dense / growth in p),
O58 (is the key set a uniform k-set or multiplicatively structured), O61 (pilot
experiment, C5/S3, fully specified), O62/O64 (minimal width, marginal cost),
O59/O92 (two-neuron mod-5, one-neuron outcome law).

## Posed ensemble measurements (mais/selection_law.py)
One-hidden-layer f(a,b)(c)=sum_i relu(u_i(a)+v_i(b)) w_i(c), full table, CE + wd|theta|^2,
N(0, 1/p) init, full-batch Adam lr 1e-3 (O61 spec), 10k steps, 128 seeds, m=4p.
Key set K_.005, spectral count N_.05 from centered logits projected on chars of a+b-c.

| p | wd | acc=1 | |K| hist | N_.05 hist | mean top shares |
|---|----|-------|----------|-----------|-----------------|
| 11 | 1e-4 | 128/128 | 5:128 | 5:127 | .21 .20 .20 .18 .12 |
| 11 | 1e-3 | 128/128 | 5:128 | 5:127 | .22 .22 .20 .14 .12 |
| 13 | 1e-4 | 128/128 | 6:127 | 6:118 | .18 .17 .16 .15 .15 .12 |
| 13 | 1e-3 | 128/128 | 6:128 | 6:107 | — |
| 17 | 1e-4 | 128/128 | 8:128 | 8:98 | .14 .13 .13 .12 .12 .11 |
| 17 | 1e-3 | 128/128 | 8:127 | 8:66, 7:61 | — |
| 23 | 1e-4 | 128/128 | 11:116, 10:12 | 11:51, 10:65, 9:12 | .11 .10 .10 .09 .09 .09 |
wd=1e-2 collapses to chance at p>=11 (CE=log p) — dead config in posed ensemble.

=> DENSE regime: N_delta(p) ~ (p-1)/2, every frequency used, shares flat. The key-set
law is degenerate (always the full set) so O58 is vacuous at m=4p. Contrast: our
transformer cohorts (p=113/127/157, 500+ runs) are sparse, K~4 across 32x width.
Live question: what switches sparsity on (width ~p? wd? optimizer? attention?).
Queued: p=23 GD lr=1 (O52 is posed for gradient flow), p=47 16 seeds, quad p=23,
width dial m=p/2p at p=23, wd 3e-3.

## Option 2 (margin/search.py): non-Fourier max-margin counterexample — CLOSED, none
ReLU, 2-homogeneous, exact f64 margin; 96 random + 16 Fourier restarts per p.
p=7: rand .0135 < F .0151; p=11: .0061 < .0070; p=13: .0042 < .0048; p=17: .0027 ~ .0027;
p=23: .0015 > .0012 (random found a BETTER Fourier solution, 93% purity, all 11 freqs
used 5-12 neurons each, leakage on 2k only 16% vs 10% chance); p=11 m=132: .0087 > .0076.
All top solutions ~94% neuron-weighted single-frequency. Quad control failed to
converge (needs degree-3 normalization) — not run properly.

## Option 3 (chaos/run_chaos.py): init-perturbation sensitivity — INCONCLUSIVE
p=113 transformer, 2000 epochs (pre-grok at wd=1 — readout = amplitude menu, not
final committee). Seeds 101, 202: menu top-4 preserved through eps<=1e-2 in most
draws (one 22->28 swap at eps=1e-4), scrambled at eps=0.1. Needs 50k-epoch pod runs
for a real answer.

## UPDATE (same day, later): dials at p=23
- GD lr=1 (gradient-flow proxy), ReLU m=4p: dense (51/64 full set). Optimizer is not the switch.
- p=47 ReLU m=4p Adam (16 seeds): dense, |K| 22-23/23, N_.05 ~ 20. Growth linear, N ~ 0.9 (p-1)/2.
- **QUADRATIC m=4p Adam wd1e-4 (64 seeds): SPARSE, |K|=6-7 in 50/64, N_.05=6 in 40/64, and the
  key set is NOT a uniform k-set: conditioned on k=7, ALL 24 runs lie in ONE multiplicative
  orbit (of 30; chi2=696/29df); k=6: 24/26 in one orbit (of 42). The orbit contains additive
  INTERVALS of consecutive frequencies ((3..9), (5..11)) and their dilates. Not an AP in
  discrete log (so not O58's guessed geometric progression). => negative answer to O58 in
  the quadratic posed ensemble, pending replication (256 seeds, m=p/2p, p=47, GD queued).**
- ReLU m=p (=23) Adam: SPARSE (|K| 3-9, mode 6; resid 0.45), and key sets CONSISTENT WITH
  UNIFORM over orbits (occupied orbits ~ expected, chi2 ~ df at every k). Width is the
  sparsity switch for ReLU; exchangeability holds there (positive answer to O58 for ReLU at
  this width, n=64). Collision (sum-free) veto does NOT appear: rate = null at all k.
- Tool: mais/orbits.py (orbit-level uniformity test; dilation symmetry is the theorem-null).
- wd dial at ReLU m=4p, p=23: coupled 3e-3 dense (56/64 full); decoupled (AdamW-style) 0.1 dense (60/64), 1.0 dense (59/64). m=p at coupled 3e-3 never reaches acc=1 (invalid). Weight decay in any form is NOT a sparsity switch for the MLP.
- ReLU width dial p=23 (Adam, wd 1e-4): m=p -> |K| mode 6; m=2p -> mode 9; m=4p -> 11/11. Uniform over orbits at every width.

## REPLICATION (256 seeds, quad m=4p p=23 Adam wd1e-4): orbit concentration HOLDS
k=6: 95/100 in orbit of (1,2,4,5,7,10) (of 42; chi2 3701/41df); k=7: 82/85 in orbit of
(1,2,3,5,6,9,10) (of 30; chi2 2290/29df); k=5: 8/9 in orbit of (1,2,4,7,10); k=8: 7/11 in
orbit of (1,2,3,4,6,7,9,11). |K| hist 5:9 6:100 7:85 8:11 9:4 10:2 11:45. Same dominant
orbits as the 64-run pilot. Exact sets include plain intervals (4..8),(7..11),(4..11).
- Quad m=p (23 neurons, 128 seeds): sparse (|K| 4-10, mode 8, resid 0.31) but orbit occupancy ~ UNIFORM (chi2 ~ df at every k); the m=4p dominant orbits are still the modal ones but only mildly (k=7: 5/25). Concentration SHARPENS WITH WIDTH: lottery at ~2 neurons/frequency, deterministic at ~8. Also: within-orbit distribution uniform at m=4p (p=0.07/0.91); weight-space neuron census == logit key set 40/40.
- Quad m=2p (128 seeds): k=6: 50/61 in dominant orbit (1,2,4,5,7,10); k=7: 22/31 in (1,2,3,5,6,9,10). Width curve of concentration (dominant-orbit share, k=6): m=p ~12% (uniform-ish), 2p 82%, 4p 95%.
- **Quad p=47 m=4p (32 seeds, 10k steps): sparse (|K| 7-12 of 23) but NO orbit concentration (every run its own orbit) and the exact law (dropped set = dilate of {1..j}) holds 1/32. Weaker structure present: longest consecutive-multiples run in the dropped set above uniform null at most k (e.g. k=10: 10.0 vs 7.6; k=12: 8.6 vs 6.2). So the p=23 law is exact only at p=23 so far; graded elsewhere. Queued: p=13/17/31 (128 seeds), m=8p, and 30k-step convergence checks at p=23/47.**
