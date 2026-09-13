# Z results — Z_120 and Z_11 x Z_11 (2026-09-13, pod 8pbeu68dw8nykq L40S; 160 runs, ~$2.5)   [scores/ZR, Z1, ZL; analyze_Z.py]

Registered predictions: PLAN_Z.md (commit c0a7235, before any run). Record recipe gd lr 3, wd 3e-4, clip 1, frac 0.5,
m 512. "ceiling" = the exact cosine-committee construction at the ladder's set S*_K (kstar_ab.json), balanced and polished.
Counts = interface-ablation K_abl over irreps (2% floor). Every one of the 160 runs grokked except the fixed-support
supports noted below.

## ZR — recipe check (32 runs, 20k)
lr 3 vs lr 1 at wd 3e-4: 8/8 grok both; gamma_2 0.71 vs 0.65 of ceiling at Z_120 (0.68 vs 0.60 at Z_11^2). Record stands.

## Z1 — main (64 runs, 16 seeds per cell, 60k; 20k checkpoint = final in every cell)

| cell | K* (plateau) | K_abl histogram | mode | (a) mode in {K*, K*+1} | (b) gamma_2 med / ceiling (max) | clean | (c) equal-share coverage M(S)/M_K med (frac >= 0.74) |
|---|---|---|---|---|---|---|---|
| Z_120 nu 4 | 4 (3..8) | {4:3, 5:5, 6:4, 7:4} | 5 | holds | 0.72 (0.84) | 0.99 | 0.58 (0.12) |
| Z_120 nu 6 | 3 ({3, 4}) | {3:1, 4:3, 5:8, 6:1, 7:3} | 5 | FAILS (+2) | 0.70 (0.93) | 1.00 | 0.62 (0.12) |
| Z_11^2 nu 4 | 4 (4..8) | {4:2, 5:9, 6:3, 7:2} | 5 | holds | 0.67 (0.79) | 1.00 | 0.72 (0.38) |
| Z_11^2 nu 6 | 4 ({3, 4}) | {4:9, 5:6, 8:1} | 4 | holds (= K*) | 0.75 (0.88) | 1.00 | 0.69 (0.19) |

- (d) Z_120: every one of 32 committees (both nu) contains non-faithful frequencies (gcd > 1 with 120); the gcd
  histogram of counted frequencies at nu 4 is {1:20, 2:6, 3:18, 4:6, 5:8, 6:5, 8:4, 10:1, 12:1, 15:4, 20:3, 24:1, 40:2,
  60:10}. The 1-dim sign character (k = 60) sits in 10/16 committees at nu 4 and 5/16 at nu 6 — the cheap-patch role
  that sign plays on S5 / S4. Prediction (ii) holds outright.
- (e) Z_11^2 nu 6: mode 4 = K* with 9/16 at 4, 6 at 5, none at 3 — prediction (i) holds (every cyclic group of this
  size gives 3-4 here). Counted directions are pairwise non-parallel in 88-89% of pairs at both nu.
- (c) KILLED as registered: the counted sets' equal-share coverage is 0.58-0.72 of the optimum at their size, far below
  the 0.74 bar in every cell. The statistic was wrong, not the committees: the margin (b) says the committees are near-
  optimal, and the ladder's own fixed-support runs below show why — a network with 5-7 counted members carries them at
  unequal amplitudes, so it is not an equal-share 5-7 set. Not replaced post hoc.

## ZL — measured ladder (fixed-support GD at S*_K, 4 seeds, 60k) vs the constructed ceilings

| group | nu | K | constructed | measured best | ratio | K_abl of the fixed-support runs |
|---|---|---|---|---|---|---|
| Z_120 | 4 | 3 | 4.80e-6 | 4.53e-6 | 0.94 | 3, 3, 3, 3 |
| | | 4 | 5.85e-6 | 5.52e-6 | 0.94 | 4, 3, 4, 3 |
| | | 5 | 5.82e-6 | 4.28e-6 | 0.74 | 4, 3, 3, 3 |
| | | 6 | 5.09e-6 | 4.27e-6 | 0.84 | 4, 5, 3, 4 |
| Z_120 | 6 | 2 | 3.99e-8 | no grok 0/4 | - | - |
| | | 3 | 5.92e-8 | no grok 0/4 | - | - |
| | | 4 | 5.41e-8 | 5.08e-8 | 0.94 | 4, 4, 4, 4 |
| | | 5 | 4.31e-8 | 4.10e-8 | 0.95 | 4, 5, 5, 4 |
| Z_11^2 | 4 | 3 | 4.37e-6 | 4.11e-6 | 0.94 | 3, 2, 3, 2 |
| | | 4 | 5.83e-6 | 2.90e-6 | 0.50 | 3, 3, 2, 3 |
| | | 5 | 5.59e-6 | 4.04e-6 | 0.72 | 3, 3, 3, 3 |
| | | 6 | 4.96e-6 | 4.32e-6 | 0.87 | 4, 4, 4, 4 |
| Z_11^2 | 6 | 3 | 5.39e-8 | no grok 0/4 | - | - |
| | | 4 | 5.40e-8 | 5.24e-8 | 0.97 | 4, 4, 4, 4 |
| | | 5 | 4.14e-8 | 4.36e-8 | 1.05 | 4, 4, 4, 4 |
| | | 6 | 3.06e-8 | 5.11e-8 | 1.67 | 4, 6, 4, 5 |

- Where fixed-support GD realises the support, it reaches 0.94-0.97 of the constructed ceiling: the constructions are
  real ceilings (as at mod-add), and the ladder's ordering at nu 4 (peak at K 4) is reproduced.
- At nu 6 the K <= 3 supports do not train at all (0/4, negative margin at 60k) on either group, although the constructed
  K = 3 nets exist. The REACHABLE ladder at nu 6 therefore starts at K = 4 on both groups. That resolves the only count
  miss: Z_120 nu 6 predicted 3 from the construction, but the reachable argmax is 4 and the free runs sit at 4-5, the
  same one-step offset as every other cell in the programme. (Cf. mod-add nu 6: constructed K* 3, measured 4.)
- Fixed-support runs given MORE members than K* shed them: at Z_11^2 nu 6 the 5- and 6-supports end with exactly 4
  counted members in 11/12 runs and beat the constructed equal-share K 5 / K 6 nets (1.05x, 1.67x) by re-using the
  freedom as a better 4-committee. The count is decided by the margin even when the support allows more.
- The Z_11^2 nu-4 K = 4 support under-trained (0.50, 3 members): one specific optimal set is hard for GD; the free
  runs find others (mode 5, 0.67 of the ceiling).

## Verdict
1. Count: K* (or K* + 1) on both new groups at both degrees once the ladder is read as the GD-reachable ladder. The
   discriminating Z_11^2 nu-6 prediction (4, not the cyclic 3) lands exactly.
2. Choice of frequencies: Z_120 networks build committees out of divisor frequencies and the sign character, exactly
   as the coverage-weighted ladder says they should; a faithful-only committee never occurs. The registered equal-share
   coverage statistic was the wrong instrument for this and is retired; the margin instrument (0.67-0.75 of exact
   ceilings, best runs 0.79-0.93) is the right one.
3. Ceilings: exact constructions hold at 0.94-0.97 where GD can realise the support; at nu 6 the sparsest constructed
   committees are unreachable by GD, which is a property of the dynamics, not the ceiling.
4. Together with mod-add (5 primes x 4 degrees), S5, S4 and A5: the l_2 ladder predicts the feature count to within one
   on every group tested; the identity (which irreps / frequencies) is coverage-weighted but seed-chosen.
