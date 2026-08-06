# Review brief: set-level margin as the frequency-selection criterion

Companion to `runs/eviction/BRIEF.md` (the implant/eviction experiment). That
brief established the phenomena; this one proposes the explanation. Requested
from the reviewer: (1) attack the hypothesis and the statistics, (2) a serious
prior-work check — has anyone already done this ("already done before" audit).

## Setup (minimal recap)

1-layer transformer (d_model 128, 4 heads, d_mlp 512, ReLU), (a+b) mod 113,
full-batch AdamW lr 1e-3, **weight decay 1.0**, frac_train 0.3. Grokked models
implement the Nanda-style Fourier algorithm: logits ~ sum over a sparse
"committee" of frequencies k of cos(w_k(a+b-c)), w_k = 2*pi*k/113. Committee =
final key frequency set, detected by largest log-gap in sorted |phase-locked
coeff|. 14 independent-init runs available (shared data split, data_seed=0):
committees of size 3 (x3), 4 (x9), 5, 6.

Established phenomena the hypothesis must explain (see BRIEF.md for data):
- Broad "audition": ~8 frequencies grow well above background during
  memorization; final committee rank-order violates audition rank in both
  directions.
- Mass evictions time-locked just AFTER grokking (e.g. seed1: f5 rank #2 all
  through memorization, peaks 2484 at epoch 12.2k, crushed to ~0; f54 rank #6
  wins).
- From a mid-audition checkpoint (epoch 3000) the outcome is deterministic and
  robust (control and sham arms reproduce the original run exactly, including
  eviction timing); a surgically implanted phase-locked outsider frequency at
  amplitude parity with a destined winner decays from the first epoch.
- Six per-frequency observables at e3000 fail to separate destined winners
  from destined losers: logit amplitude, growth rate, phase-lock fraction, MLP
  energy share, committed-neuron count (zero for everyone at e3000),
  null-calibrated phase-coherence (z-scores: losers 8.8 vs winners 7.7 — if
  anything reversed).

## The hypothesis

Frequency selection is governed by an intrinsic, model-free, set-level
quantity: the **restricted min-margin** of the idealized solution.

For committee S with equal amplitudes, the logit for candidate answer c on
input (a,b) depends only on the miss x = a+b-c mod p:

    f_S(x) = sum_{k in S} cos(2*pi*k*x/p)

f_S(0) = |S| always. Define **margin(S) = |S| - max_{x != 0} f_S(x)**: the gap
between the correct answer's logit and the best wrong answer's logit. This is
a Diophantine property of the set — whether some x can drive all k*x mod p
near 0 simultaneously. It varies enormously across sets and is constant
through training (it is number theory, not weights).

Why the network should care: CE at scale A falls like exp(-A*margin), so the
amplitude (=> weight norm => weight-decay cost) needed for a given loss is
~1/margin. When training enters the wd-dominated consolidation phase
(post-grokking), margin-per-norm is what is optimized; members of low-margin
committees are evicted regardless of amplitude.

Claimed mechanism story: the audition phase is amplitude-greedy and
margin-blind (fit the masked data by any means); consolidation re-judges by
margin-per-norm; the purge implements the switch. This predicts/retrodicts:
determinism + perturbation-robustness (the ranking is a mathematical
constant), failure of all per-frequency observables (margin is a set function
with no per-frequency shadow), eviction of amplitude leaders, and why K=1
never occurs (single-frequency margin = 1 - cos(2*pi/113) = 0.00155 — 100%
accuracy but absurd norm cost).

## Evidence

### 1. Committees differ intrinsically (null distributions, 30k random sets)

| K | mean | sd | min | max | with a+b=c relation | without |
|---|------|----|----|-----|--------------------|---------|
| 3 | 0.417 | 0.191 | 0.022 | 0.845 | 0.164 | 0.427 |
| 4 | 0.946 | 0.344 | 0.046 | 1.884 | 0.623 | 0.999 |
| 5 | 1.590 | 0.465 | 0.101 | 2.775 | 1.298 | 1.727 |
| 6 | 2.287 | 0.552 | 0.285 | 3.697 | 2.082 | 2.541 |

~40x spread at K=4. Sum-closed sets are systematically bad — this also
retrodicts an independent earlier observation: 0/14 observed committees
contain an a+b=c relation vs 2.3 expected by chance.

### 2. Observed committees sit high (percentile vs size-matched null)

mainline {17,7,35,29} 93.8 | seed37 {43,26,39,37} 40.0 | seed42
{54,14,15,7,42} 76.7 | seed67 {42,12,49,28} 93.8 | seed69 {42,7,17} 93.9 |
seed72 {14,34,5,2} 97.1 | seed81 {15,25,2,14} 85.0 | seed1 {7,54,30,4} 97.8 |
seed27058 {14,49,52} 93.2 | seed51224 {7,38,4,50,2,17} 76.3 | seed55794
{14,6,1} 38.1 | seed63523 {41,16,23,52} 60.0 | seed71539 {7,1,30,34} 95.3 |
seed84679 {27,15,14,7} 39.4. Mean percentile **77.2** (uniform: 50).

Note: NOT at the optimum — three runs sit ~40th percentile. Claim is
margin-biased local search over nucleation-reachable sets, not global
optimization.

### 3. Counterfactual test (the sharp one)

For each of the 7 farmed runs (which have full spectra), compare margin of the
chosen committee vs the committee of the top-K frequencies by amplitude at
epoch 3000 (what rich-get-richer predicts):

- 3 runs: same set (no information).
- 4 runs where they differ: chosen margin > amplitude-favorite margin, **4/4**:
  seed1 1.582 vs 0.836; seed51224 2.715 vs 2.125; seed63523 1.052 vs 0.667;
  seed71539 1.506 vs 0.506. In each case the eviction victims are exactly the
  members whose presence made the amplitude-favorite set bad.

### 4. Statistics (Monte-Carlo unless noted)

- A. Mean percentile 77.2: p = 1.45e-4.
- B. Fisher-combined upper-tail evidence: p = 9.4e-4.
- C. 7/14 above 90th percentile: p = 1.4e-4 — POST-HOC threshold, transparency
  only, do not count as evidence.
- D. **Skeptic's null** (the important one): all 14 runs share one train mask
  and visibly reuse frequencies (7 appears in 7 committees, 14 in 6). Null:
  resample committees with frequencies drawn per the observed popularity
  (add-0.5 smoothing). Result: popularity-matched random committees average
  the 48.9th percentile (sd of the mean-statistic 7.7) — composition explains
  NONE of the enrichment; 0/20000 null worlds reach 77.2 (p < 5e-5). The
  effect is at the level of which frequencies go TOGETHER.
- E. Counterfactual sign test 4/4: p = 0.0625 alone (weak solo; independent
  arrow).
- Forking-paths discount: margin was the ~6th hypothesis tested after five
  failures; crude Bonferroni x6 leaves the main result at ~1e-3.

## Novelty claim (what we believe is new — REVIEWER: verify)

Closest prior work and where we believe the boundary is:

- **Morwani et al., ICLR 2024 (arXiv:2311.07568)** — margin maximization for
  modular addition. Their Theorem 7 (wide quadratic 1-hidden-layer MLP, full
  dataset, L_{2,3}, m >= 4(p-1)): max-margin solutions have single-frequency
  phase-locked neurons and use ALL (p-1)/2 frequencies; exact
  gamma* = sqrt(2/27)/(p^{1/2}(p-1)). Critically, their proof substitutes the
  uniform class-weighted (i.e. AVERAGE-over-wrong-answers) margin, under which
  all frequencies are exactly interchangeable — the worst-case sidelobe
  quantity our hypothesis lives on is averaged away by construction, and the
  full spectrum is the one configuration where min = average. They justify
  Fourier as a basis; they do not rank frequencies or subsets; no dynamics,
  no split, no finite-wd regime, nothing below width 4(p-1). (Their App. I.2
  notes subset-of-representations support for general non-cyclic groups — a
  precedent for "margin selects a subset", but not for Z_p, not Diophantine,
  no dynamics.)
- **He et al. 2026 (arXiv:2602.16849)** — two-layer MLP dynamics: per-neuron
  init lottery (magnitude + phase misalignment), full frequency
  diversification at the network level, no train/test split in the theory.
  Our eviction/counterfactual data directly contradicts the lottery at
  network level in our regime.
- **Nanda et al. 2023 (arXiv:2301.05217)** — the algorithm + progress
  measures incl. a cleanup phase; no committee selection question.
- **Varma et al. 2023 (arXiv:2309.02390), "Explaining grokking through
  circuit efficiency"** — efficiency competition between memorization and
  generalization circuits. Same economic logic one level coarser; we apply it
  WITHIN the generalizing solution, between frequency committees. Reviewer:
  check they don't do frequency-level efficiency anywhere.
- **Ding et al. (~2024), "Survival of the Fittest Representation" (modular
  addition case study)** — representation competition/death; closest prior to
  our eviction phenomenon. Reviewer: read closely — do they (i) document
  post-grokking eviction of high-amplitude frequencies, (ii) tie survival to
  any intrinsic set-level quantity?
- Mean-field / landscape lines (Tian 2024; Wang & Wang 2025; Kunin et al.
  2025; Gromov 2023; Zhong et al. 2023; Liu et al. 2022): to our knowledge
  none rank frequency subsets or predict which frequencies survive.

Specifically claimed as new: (i) the restricted min-margin varies ~40x across
frequency subsets of Z_113 via Diophantine structure and is computable a
priori; (ii) trained sparse committees statistically track this ranking
(77th pct, survives popularity-conditioned null); (iii) when amplitude
favorites and margin favorites diverge, dynamics side with margin via
mid-training eviction (4/4); (iv) the two-phase mechanism (amplitude-greedy
audition, margin-per-norm consolidation) with causal support (determinism,
robustness, implant rejection at amplitude parity).

Reviewer literature-audit asks, beyond the above list: search for (a) any
work computing margins/loss of RESTRICTED Fourier frequency sets for modular
addition; (b) any work on "which frequencies" selection beyond init-lottery
accounts; (c) exponential-sum / Weyl-sum literature applied to neural mod-p
tasks (the margin functional is a classical exponential-sum extremum — pure
math priors are fine, ML applications would be overlap); (d) any documented
mid-training eviction of high-amplitude Fourier components.

## Caveats (attack surface)

- n = 14 committees; counterfactual n = 4; all runs share data_seed=0. Test D
  rules out composition-level mask confounding, not every conceivable
  set-level mask correlation.
- Margin functional is equal-amplitude idealized. Real committees have ~2:1
  amplitude ratios; the amplitude-optimized restricted margin (max-min over
  amplitude allocations, a small LP) is the right functional and untested. The
  three ~40th-percentile committees may look better or worse under it.
- Committee detection = largest-log-gap heuristic on sorted |coeff|.
- Hypothesis was arrived at post hoc after five failed alternatives (discount
  applied above, but the clean fix is prediction, not correction).
- The hypothesis explains which sets survive; it does not yet explain the
  tie-break among good sets (freq 7's popularity under the fixed mask), nor
  the nucleation/reachability constraint that keeps runs off the optimum.
- 3/7 counterfactuals were SAME (amplitude favorite = final committee), so
  dynamics only demonstrably override amplitude when margin disagrees; a
  margin-blind reader could still say "amplitude usually wins" — the 4/4 is
  the rebuttal but it is n=4.

## Decisive tests proposed (not yet run)

1. **Pre-registered forecast**: fresh seeds with VARIED data_seed, spectra
   logged; at mid-audition, enumerate candidate committees and publish
   predicted final committee + predicted evictions from margin alone, before
   the runs finish. Kills forking-paths and the shared-mask caveat at once.
2. Amplitude-optimized margin functional; re-score the 14 committees.
3. Implant-into-context: implant an outsider that would COMPLETE a
   high-margin committee with existing auditioners (vs the f36 implant which
   completed nothing) — the hypothesis predicts context-dependent implant
   survival.
4. wd sweep: hypothesis predicts committee percentile rises with weight decay
   (stronger margin-per-norm pressure) and sparsity level shifts.

## Post-review addendum (attacks 1 and 3 executed)

Reviewer attack 1 (equal-amplitude functional) resolved in three steps:

1. Observed-amplitude rescoring (committees at their real amplitudes vs nulls
   carrying the same amplitude profile): mean percentile 99.0, min 94.8; the
   three "mediocre" committees jump to 94.8-99.4. HOWEVER —
2. The fair comparison (every set at its own LP-optimal allocation, the
   max-min zero-sum game solved exactly per set) deflates this back to mean
   percentile **77.0** — the 99 was an artifact of nulls wearing mismatched
   allocations. The set-level enrichment claim stands at ~77th pct
   (p ~ 1.5e-4 by the same uniform-percentile argument), NOT 99.
3. New independent finding: **allocation near-optimality.** Observed
   amplitude allocations achieve mean 94.0% (range 0.82-1.00) of their set's
   LP-optimal margin. Two-level structure: allocation (continuous) is
   near-optimal; selection (discrete membership) is only biased — under LP
   scoring, chosen committees rank 1/70, 5/56, 1/28, 8/70 among K-subsets of
   their own top-8 audition in four runs, but 38/56, 28/70, 36/70 in three.
   Membership is reachability/nucleation-constrained in a way allocation is
   not.

Reviewer attack 3 (crown counterexample) — predictions (i) and (ii) both
confirmed: the CE-completion harmonic committee (detected {54,5}; 5 = fold
(2*54), harmonically locked) scores 0.1st percentile at observed amplitudes
and **2.6th percentile even at its LP-optimal allocation** (LP relM 0.0062 vs
K=2 null mean 0.059); its weight norm is 1133 vs 827-944 for natural zoo
finals that reach 40x better CE (6.0e-6 vs ~1.6e-7). Two-economy summary
supported: completion recruits what is cheap to build (arithmetic closure of
installed structure); consolidation keeps what is cheap to run
(margin-per-norm). Prediction (iii) — metastability of the harmonic committee
under extended wd training — remains untested (requires a run).

Statistics caveat added by this addendum: the popularity-conditioned null
(test D) was run on equal-amplitude percentiles and has not been re-run under
LP scoring; expected similar but unverified. McCracken et al. 2505.18266
(approximate-CRT / O(log n) features) is flagged but not yet read in full;
the frequency-exchangeability question there is open.

## Addendum 2 (2026-07-28): the loss law, the 18.4-nat homeostat, and the revised economics

Free (analysis-only) session on existing data. Five results, two of which
supersede earlier framing.

### A. Zero-parameter loss law confirmed

Reading each model's cosine amplitudes a_k (in logit units) off its own
final logits (Fourier transform of the translation-averaged logit profile)
and predicting CE = sum_{x!=0} exp(-gap(x)), gap(x) = sum_k a_k(1-cos(w_k x)),
with NOTHING fitted: prediction matches the translation-symmetric part of
the measured CE to within a few percent for all 14 natural runs. Actual CE
sits a near-constant ~3x above the symmetric part (row-to-row phase noise;
13/14 runs in 2.5-5.6x, one at 17.8x). Post-grokking loss IS
exp(-A*relM), in actual nats, per model.

### B. The homeostat: A*relM is equalized at 18.4 +/- 0.4 nats

The planned slope(-1) regression of ln CE on the margin is unrunnable for
the best possible reason: there is no variance in the x-axis. All 14
natural committees, whose set-quality relM spans 3x (0.18-0.56), end at
minGap = A_tot*relM in [17.9, 19.5] nats. Weak committees compensate with
amplitude almost exactly ({14,6,1} carries A=101 where {7,54,30,4} needs
42). Training does not merely respond to margin; it equilibrates on it.
Trajectory version (spectra, calibrated to logit units): minGap(t) reaches
95% of its final plateau AT the grokking epoch in all 7 farmed runs —
margin arrival and grokking are the same event. New falsifiable
prediction: 18.4 is an equilibrium constant of the CE-vs-wd balance and
must shift under wd/lr changes.

### C. Amplitude is cheap; norm prices MEMBERS, not decibels (supersedes "margin-per-norm")

Weight norm anti-correlates with A_tot (r = -0.52) and instead tracks
committee size: K=3 runs ~850, K=4 ~905, K=5 969, K=6 1024. A weak
committee buys 2.4x the amplitude at identical norm. Consequence: at final
equilibrium all natural committees achieve the same CE at essentially the
same norm — the endpoint fitness landscape is nearly flat, so the 77th-pct
selection enrichment CANNOT be explained by endpoint costs. The selection
pressure must act during the transient (the post-grokking consolidation
window, where evictions in all 7 farmed runs cluster within ~1-3k epochs
of the grokking epoch). "Margin-per-norm pricing" as a static story is
retired; the live hypothesis is margin-per-amplitude during the climb,
plus a per-member infrastructure tax from wd.

### D. Statistics caveats closed; p=113 is typical

- Test D re-run under LP optimal-vs-optimal scoring: mean LP percentile
  77.3, uniform-null p = 8.5e-5, popularity-conditioned null mean 48.8
  (sd 7.7), 0/1500 null draws as extreme. The enrichment is significant
  under the honest scoring and is not frequency popularity in disguise.
- Margin-spread analytics for p in {59..251}: the ~4x amplitude-cost
  spread among K=4 sets at p=113 is generic and grows gently with p.
  Nothing special about 113.

### E. Failures and the monopoly's true pathology (report both plainly)

- **Seventh early-observable failure**: leave-one-out margin contribution
  per unit amplitude at e3000 (the theory's own currency) does NOT predict
  survivors (mean AUC 0.47, 7 runs, no signal). Selection remains
  illegible at audition time; whatever decides is decided during the
  grokking climb, consistent with B/C.
- **Monopoly run re-read**: its translation-symmetric profile ALSO obeys
  the homeostat (minGap_sym plateaus 18.1-18.7 by epoch ~1300 — the full
  harmonic tower rescues the average profile). Its 40x CE excess is
  row-phase noise: actual/symmetric CE ratio ~125-250 and RISING
  (natural runs: ~3), CE bottoms at 3.7e-6 near e3500 and degrades to
  6.0e-6 by e10000 while wd grinds norm 1662->1133. The run is visibly
  not at a fixed point — partial free support for prediction (iii)
  (metastability), which still wants the extended-training run for the
  eviction/repair endgame. Caveat: the act/sym ratio may partly reflect
  train/test row asymmetry; not yet decomposed.

### F. Cross-split replication (data-seed sweep, 2026-07-28)

Six fresh runs (data seeds 1 and 2 — new train/test masks — 3 OS-entropy
init seeds each, 30k epochs), landing in `runs/seed1/`, `runs/seed2/`. All
grokked to 100% test acc. Results on data never touched by any prior
analysis in this project:

- **Homeostat replicates**: minGap = A*relM in [18.13, 18.87] nats
  (mean 18.53) across all 6. Now 20/20 natural runs in [17.9, 19.6].
- **Enrichment replicates and strengthens**: LP optimal-vs-optimal
  percentiles 81.6-96.8, mean **89.8** (CLT p ~ 4e-4 for n=6 vs uniform;
  original 14 runs: 77.3). Possible cause of the higher mean: 30k epochs
  vs 20k (longer consolidation) — testable by extending old runs.
- **Norm prices members, again**: K=4 runs norm 888-903; the K=5 run 987
  (seed0's K=5: 969, K=6: 1024).
- **Loss law holds**: act/sym CE ratio ~3-7 for K=4 runs; the K=5 run at
  17.1 echoes seed0's K=6 outlier (17.8) — larger committees carry more
  row-phase noise. Unexplained regularity, now seen twice.
- **The mask shapes frequency identity**: mean shared members per
  committee pair is 0.62 within a split vs 0.32 across splits — and the
  across-split value equals the uniform-null 0.29. No universally favored
  frequencies exist; frequency popularity is entirely mask-specific
  (e.g. data seed 2 committees {18,38,15,11} and {18,26,11,38} share 3
  members across different inits). This kills any mask-independent
  per-frequency lottery story and sharpens test D: popularity itself is
  a mask effect, while the margin enrichment survives across masks.

**Canonical ensemble (final)**: `runs/seed0` was rebuilt with the same
uniform recipe (3 fresh init seeds, 30k epochs), giving the canonical
3x3 matrix. On the 9 runs alone: homeostat minGap mean 18.71, sd 0.36,
range [18.13, 19.36] (9/9); LP percentile mean 85.0 (uniform-null
p = 3.5e-5), min 62.3; norm 898 for K=4 (n=8), 987 for the K=5;
within-split overlap 0.89 vs across-split 0.26 (uniform-null 0.29).
Fresh seed0 committees re-picked the archive's mask favorites (42, 37,
14, 15, 2) from new inits, as the mask-menu story predicts. One outlier
to watch: seed0/seed37099 ({42,15,23,37}) has act/sym CE ratio 691
(CE 1.5e-5, still 100% acc) — a natural run with monopoly-style
row-phase noise, the first seen; worth checking whether extended
training repairs it.

## Data & scripts

- Runs/spectra: `runs/og_seed0/seed*/spectra.npz` (7 farmed runs,
  all-56-frequency trajectories every 50 epochs); old zoo committees from
  final checkpoints. (`runs/zoo` was renamed `runs/seed0` and then
  `runs/og_seed0` on 2026-07-28; the zoo is now organized by data seed:
  `runs/seed<data_seed>/seed<init_seed>`. The canonical ensemble going
  forward is the uniform-recipe matrix `runs/seed{0,1,2}/` — 30k epochs,
  3+ OS-entropy init seeds per data seed — with `og_seed0` retained as the
  exploratory-era archive and the monopoly/implant runs excluded from
  ensemble statistics as interventional, not observational, models.)
- Eviction experiment: `runs/eviction/` + `runs/eviction/BRIEF.md`.
- Analysis scripts (session scratchpad): `intrinsic_margin.py` (functional,
  nulls, percentiles, counterfactuals), `margin_significance.py` (tests A-E),
  `loser_census.py`, `coherence_test.py` / `coherence_v2.py` (the failed
  observable, incl. null calibration), `eviction_results.py`.
- Addendum-2 scripts: `ce_prediction.py` (zero-parameter loss law),
  `norm_price.py` (norm vs A_tot vs K), `testD_lp.py` (test D under LP),
  `primes_margin.py`, `homeostat_time.py` (minGap(t) + eviction timing),
  `currency_test.py` (failed predictor), `monopoly_drift.py`.
- Morwani PDF: local copy at `~/Downloads/2311.07568v2.pdf`.
