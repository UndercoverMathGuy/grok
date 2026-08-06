# Frequency selection in grokked modular addition: what is proved, what the theory says, what the paper is

Status: 2026-07-31 (revised after the adversarial re-analysis pass; attack
scripts `attack_*.py` live in this directory, results logged as T32-T37).
Sources: 24 spectra-logged runs across 9 data splits and 3 primes (+7
archive runs), the eviction/implant experiments, and the init surgery
(`runs/surgery/`, 5 arms). Evidence tiers are explicit; deflated claims are
listed so they stay dead. Major revisions this pass: the T23 natural-run
margin-prediction claim is RETRACTED (survivor-floor circularity, T32); P2
is narrowed (head-start vs growth-rate is unresolved, T35); the additive
depletion shadow is PROMOTED to P4's primary evidence with repair-stage
provenance (T34).

---

## 1. Proved (causal — intervention-grade evidence)

**P1. Committee identity is set by per-frequency energy noise in the
embedding at initialization.**
- Knockout (24 runs): predicting the final committee from epoch-0 weights
  gives AUC 0.70 (neuron-level readout). Replacing W_E with fresh random
  weights collapses it to 0.52 (chance); replacing MLP W_in changes nothing
  (0.71); attention partial (0.60); output side ~nothing.
- The init tilt is *pure chi-square noise*: relative sd of per-frequency
  W_E energy is 0.088, exactly the Gaussian-matrix prediction. Winners start
  at +0.55 sigma (~+5% energy) on average. There is no structure to find.
- Surgery (base run seed27058, committee {14,49,52}; control reproduced
  exactly):
  - **+2 sigma boost** (x1.2 energy, inside the natural noise range) on a
    dead outsider (7): adopted into the final committee. Membership flips at
    the natural fluctuation scale.
  - x2.25 boost: same outsider becomes the *dominant* member.
  - x0.5 suppress on the strongest incumbent (49): annihilated
    (peak coefficient 24,622 -> 172 -> 0).
  - Collision arm (boost 12 = fold(49+52), inviting a degenerate trio):
    12 adopted, incumbent 52 evicted — an engineered reconfiguration.

**P2 (narrowed 07-31). The carrier of the init signal is W_E energy — not
He et al.'s per-neuron alignment — and it is cashed out during epochs 0-50;
whether as a level head start or a growth-rate advantage is UNRESOLVED.**
Corr(init energy, log amplitude at epoch 50) = +0.33. Amplitude ordering
predicts the final committee at AUC 0.95 by epoch 50 and 0.985 by epoch 150
(memorization end); the ordering then persists to grok (rich-get-richer).
The signal is invisible in logits at init (corr +0.04, AUC 0.56) — so there
is *no head start in the computed function*; the tilt is expressed entirely
through the 0-50 window, for which no dense snapshots exist. Moreover init
energy correlates positively with log-amplitude growth in *every* measured
50-epoch bin (+0.09/+0.12/+0.20/+0.16 for 50-100 ... 150-200; T35), and the
two-snapshot "exponent" (corr +0.15) is attenuation-biased relative to the
level variable — a persistent-small-growth-advantage model fits the data as
well as a one-time head start. The old dichotomy claim is withdrawn; dense
0-50 spectra would settle it.

**P3. Menu closure.** The final committee is contained in the mid-audition
top-8 in 23/24 natural runs, and *every* recruit in every surgery arm
(including the K 3->5 rebuild after suppression) came from the original
menu. Nothing outside the audition ever enters a committee.

**P4 (restructured 07-31). Consolidation is a local repair operator whose
one proven signature is that it strips additive degeneracy; its trigger
variable is unknown.**
- PRIMARY EVIDENCE (promoted): committees avoid pairs whose sum or
  difference is also a member — 6 observed vs 23.2 expected. Correct
  inference (17/31 committees share a mask, so the old iid p = 0.0018 is
  invalid): mask-cluster bootstrap p = 0.0003; popularity-matched null
  p = 0.0048; robust to the committee detector (T33). Provenance is now
  repair-stage: on the same 24 runs the *blind* e3000 amplitude draws show
  NO depletion (20 obs vs 18.4 null) while the final committees are
  depleted (4 vs 18.4, p = 0.0035; T34) — the depletion demonstrably
  appears between audition and convergence, where consolidation acts.
- RETRACTED: "reconfiguration is predicted by the blind draw's margin
  percentile (p = 0.012)". Survivor-floor circularity: for loyal runs the
  blind draw *is* the final committee, so its percentile restates the floor.
  The claim's only new content — reconfigurers' blind draws below random —
  is not present (mean pctile 42.6 vs 50, p = 0.12-0.28, n = 14; T32), and
  p-127/seed3604/seed31429 reconfigured from a 99.6th-percentile draw.
  Repairs raised the margin percentile in 11/14 (not 12/14) cases, with two
  clear margin-lowering repairs.
- Engineered runs (DEMOTED to suggestive): both surgery-induced evictions
  chose the margin-best single swap among the available options (boost arm:
  0.386 vs 0.366 / 0.339 / 0.177; collision arm: 0.356 vs 0.327 / 0.262) —
  but that is 2 events with 3-4 options each, joint p ~ 0.08 under random
  choice. Needs the swap-event farm before "margin-guided" can be asserted.
- Never global: repaired committees rank only #8 and #11 of 35 menu subsets
  by margin; incumbents are kept whenever affordable.
- Graded, not a binary veto: the x2.25 boost evicted the harmonic relative
  (14 = 2*7) from a *feasible* set, while the x1.2 boost left the same four
  coexisting — eviction economics depend on the invader's amplitude, not
  feasibility alone. A weak-but-feasible hand is alternatively paid for with
  amplitude (seed55794 carries relM 0.115 at ~2.4x amplitude).
- Margin's surviving role: a *floor* (gates degenerate committees) and a
  measurement tool. It neither ranks survivors nor — on current evidence —
  triggers repair.

## 2. Proved (statistical — observational)

**S1. The train split determines frequency popularity, but through no simple
statistic of itself.** Within-split committee overlap 0.688 members vs 0.300
across splits (= uniform null 0.279; 31 runs). Corrected inference (07-31):
the original pair-level permutation test is deprecated (wrong exchangeable
unit, and each iteration drew two independent permutations); the run-level
mask-label permutation gives p = 0.0003, and p = 0.028 after excluding the
dominant dseed-0 mask (T36). The effect is heterogeneous across masks:
dseed-1's within overlap (0.333) is statistically nothing over the across
baseline, while dseed-2's is 1.333 — 0.688 is a population mean, not a
per-mask law. Yet every low-order
functional of the mask fails to predict membership (diagonal spectrum,
marginals, pair channels, harmonic channels: all null), and a first-order
symmetry argument shows the masked template Gram is exactly diagonal — the
mask *cannot* break frequency symmetry at first order.

**S2. The mask effect is architecture-specific.** A 2-layer quadratic MLP
(same task, same masks, same optimizer family, groks to 100%) shows *zero*
mask favoritism (within-mask committee consistency at chance), tied or
untied embeddings. Naive "He et al. transfers to any architecture" is false.

**S3. He et al.'s specific mechanism fails in this transformer.** Their
per-neuron input/output alignment variable carries no committee signal
(knockout P1); the correct variable is embedding energy, acting as a head
start (P2), not "best gradient fastest."

## 3. Dead (explicitly, so they stay dead)

- **The homeostat, the loss law, allocation optimality, and the
  amplitude-margin compensation are NOT findings.** Given the known circuit
  form (Nanda: logits ~ sum_k a_k cos(w_k(a+b-c))), CE's definition already
  *is* the "loss law"; allocation near-optimality is gradient descent
  minimizing its own objective; the 18.4-nat constancy restates same-loss
  convergence at fixed hyperparameters (corr(minGap, -log CE) = +0.82). LP
  margin accounting remains legitimate as a *measurement tool* (it scored
  the surgery repairs) but asserts nothing.
- All closed-form identity proxies: mask spectra (4 variants), LP-support,
  smallest-feasible-K, multiplicative/harmonic enrichment. Each tested,
  each null. Identity has no cheap formula because it is quenched noise (P1).
- "Margin ranks committees" (the old 77th-percentile story): the enrichment
  is the repair operator's floor shadow; margin gates and prices, never
  ranks survivors.
- **"Low blind-draw margin predicts reconfiguration" (T23, p = 0.012)** —
  retracted 07-31. The Mann-Whitney contrast is loyal-side floor selection
  in disguise; reconfigurers' blind draws are statistically random (T32).
  Any future repair-trigger claim must predict WHICH runs reconfigure from
  a variable measured before consolidation, and beat the additive-relation
  count (which, unlike margin, does separate blind from final — T34).
- **"There is no structure to find" in the init tilt** — softened: the
  chi-square match (relative sd 0.088 vs 0.088) checks one moment of the
  marginal distribution. Cross-frequency or higher-moment structure is
  untested. Claim only "consistent with pure noise on the moment checked."

## 4. The theory (current best statement)

> **Selection by dice and debt.** At initialization, the embedding assigns
> each frequency a chi-square-random energy; this is an amplitude head
> start. Exponential, competitive amplification during memorization turns
> the head-start ordering into a stable audition hierarchy by ~epoch 50 (the
> top-8 of which is the "menu"). The final committee is the amplitude
> top-K of the menu unless that hand is degenerate, in which case
> consolidation performs local single-member repairs, recruiting from the
> menu. What is *proven* about the repair operator is its composition
> signature: it strips additive relations (sum/difference pairs), which are
> absent from final committees but not from the blind draws they replace.
> Whether the trigger is the margin functional is OPEN — blind-draw margin
> does not predict which runs reconfigure (T32); the two engineered
> evictions chose the margin-best swap but at n = 2 that is suggestive
> only. For mild degeneracy the network can instead pay extra amplitude
> (the flat-CE bookkeeping). Identity is therefore lawless (quenched
> randomness, causally manipulable at the +/-2-sigma scale); the margin
> functional's proven role is a feasibility *floor*, not a ranking or (yet)
> a trigger.

Known incompleteness of the theory:
- The init tilt explains AUC 0.70 at epoch 0 vs 0.95 at epoch 50: the
  first ~50 epochs add information (data-split interplay, S1/S2) whose
  mechanism is unknown and architecture-specific — and the *form* of the
  tilt's expression (level head start vs persistent growth advantage) is
  itself unresolved inside that window (P2, T35).
- The repair operator's trigger variable is unknown (T32 killed the margin
  candidate); its only proven property is the additive-relation signature.
- The graded eviction threshold (evict vs coexist vs pay) has no
  quantitative form yet; two clean engineered data points constrain it.
- The phase-noise anomaly (actual/symmetric CE spanning 3-540x across runs)
  is unexplained by anything above.
- K (committee size) is described (menu + repair + suppression arm grew K)
  but not predicted.

## 5. The paper

**Working title:** *Dice and debt: a causal account of which Fourier
circuits a grokking transformer learns.*

**Question:** when a task admits many equivalent circuits (here: any sparse
set of ~56 frequencies solves modular addition), what selects the one a
network actually learns? This is the canonical mechanistic-interpretability
model organism, with the *algorithm* fully mapped since Nanda et al. — but
the *selection* of its parameters (the committee) unexplained.

**Contributions:**
1. **Causal localization of circuit identity to init noise in the
   embedding** (knockout + five-arm surgery, including membership flips at
   the natural noise scale and targeted annihilation of the dominant
   circuit). First intervention-grade account of circuit selection in a
   transformer; corrects the logit-level "init is unreadable" conclusion.
2. **Mechanism: the carrier is embedding energy, not gradient alignment.**
   Direct refutation of the transfer of He et al.'s per-neuron alignment
   variable to transformers (knockout T20). Scope limit stated honestly:
   whether the energy tilt acts as a level head start or a persistent
   growth advantage inside epochs 0-50 is left open (T35).
3. **The repair operator.** Consolidation characterized as local
   single-member repair whose proven signature is additive-relation
   stripping: absent from final committees (cluster-bootstrap p = 0.0003)
   but present at base rate in the blind draws they replace (p = 0.0035
   for the contrast, T34); demonstrated on demand via engineered
   invasions. The margin functional supplies the feasibility floor; its
   candidacy as the repair *trigger* is explicitly open (the natural-run
   prediction claim was retracted after adversarial re-analysis, T32).
4. **Negative-result backbone** (what makes 1-3 credible): systematic
   falsification of every closed-form identity proxy, and the surrogate
   non-transfer result showing mask favoritism is architecture-specific.
   Delineates the lawful part of selection (feasibility) from the lawless
   part (identity) — an epistemic template for "why this circuit and not
   another" questions in larger models.

**What the paper still needs before submission:**
- Surgery replication on 2-3 more base runs + a dose-response curve
  (adoption probability vs boost sigma, ~15 runs, ~3 h of compute) — with
  spectra logged every ~5 epochs for the first 300, which would also
  settle the P2 head-start/growth-rate question for free.
- The repair-threshold phenomenology formalized (evict/coexist boundary vs
  implant amplitude; the two existing arms bracket it).
- Swap-event mass production (the best-single-swap claim rests on 2
  engineered events, joint p ~ 0.08; the farm would harden or kill it).
- A repair-trigger variable that actually predicts which runs reconfigure
  (margin is dead for this, T32; the blind draw's additive-relation count
  is the natural next candidate, cf. T34).
- Literature positioning against He et al. (2-layer MLP init lottery),
  Morwani et al. (margin maximization), Ding et al. (representation
  competition), Varma et al. (circuit efficiency).

**Explicit scope limits:** one architecture, one task family, p in
{113,127,157}; the mask-popularity mechanism and phase-noise anomaly are
reported as open; no claim about the converged loss level (see §3).

---

## 6. Complete experimental log (adversarial-review package)

Everything lives in this directory (`findings/`): this document plus every
script behind the numbers below (bare script names in the tables refer to
files here). Run from the repo root: `uv run python findings/<script>.py`.
All run data is referenced by repo-relative path (`runs/...`). Shared utilities (run discovery,
committee detection, AUC) are in `findings/mask_lottery.py`; the margin
functionals (`relM_equal`, `lp_relM`, `homeostat`) are in
`scripts/margin_analysis.py`. Earlier-era experiments (implant/eviction,
homeostat harness, cross-prime farm) are documented in
`runs/eviction/BRIEF.md`, `runs/eviction/MARGIN_HYPOTHESIS.md`,
`runs/homeostat/BRIEF.md` (note: the homeostat brief's claims are deflated,
see section 3).

### Data inventory

- 24 spectra-logged grokked runs (the sample for every "n=24" test):
  `runs/og_seed0/seed{1,27058,51224,55794,63523,71539,84679}` (7, archive,
  data_seed 0), `runs/seed0/seed*`, `runs/seed1/seed*`, `runs/seed2/seed*`
  (9, data seeds 0/1/2), `runs/p-{113,127,157}/seed{2034,3604}/seed*` (8,
  fresh masks, three primes). Each has `config.json`, `spectra.npz`
  (401 snapshots x 56 freqs: phase-locked coeffs, energies, accs, every 50
  epochs), `checkpoints/epoch_*.safetensors` every 1000 epochs INCLUDING
  epoch 0, `metrics.json`.
- 7 additional committee-only runs (final checkpoints, no spectra):
  `runs/og_seed0/{mainline,seed37,seed42,seed67,seed69,seed72,seed81}` —
  used only in the n=31 composition censuses.
- Surgery runs: `runs/surgery/{control,collision,boost_strong,suppress,
  boost_subtle}` (12k epochs each, spectra every 100 epochs, checkpoints
  every 2000). Surgical checkpoints (modified epoch-0 weights) were written
  to the session scratchpad as `surg_<arm>.safetensors`; regenerate with
  `findings/surgery.py` (deterministic given the base run's epoch-0
  checkpoint).
- Excluded: `runs/og_seed0/seed11494` (never grokked, test acc 0.14);
  interventional runs (`runs/eviction/*`, distill runs) excluded from all
  ensemble statistics.

### Tests in chronological order (this investigation)

Mask-proxy sweep (all NULL; user-deprioritized direction afterwards):

| # | test | script | result | verdict |
|---|------|--------|--------|---------|
| T1 | mask diagonal spectrum abs(m_k) vs committee membership (perm. null within mask, 20k iters, 31 runs / 9 masks) | `mask_lottery.py` | member pctile 48.4 vs null 50.0+-3.7, p=0.67 | null |
| T2 | same vs audition top-8 at memorization end (24 runs) | `mask_lottery.py` | 50.4, p=0.44 | null |
| T3 | (a-b)-direction spectrum control | `mask_lottery.py` | 47.9, p=0.71 | null (as expected) |
| T4 | per-mask popularity Spearman(pop, abs(m_k)) | `mask_lottery.py` | rho -0.09/+0.08/+0.21, all ns | null |
| T5 | row/col marginal spectra M(k,0),(0,k) | `mask_lottery2.py` | 53.0, p=0.20 | null |
| T6 | pair-coupling channels abs(m_fold(j+-k))^2 for committee pairs | `mask_lottery2.py` | 60.0th pctile, two-sided p=0.054 | not credible after 5 looks; only nominal near-hit of the sweep |
| T7 | harmonic channels H2 (2k) / H3 (3k) / Htot, membership + audition + popularity | `mask_lottery3.py` | p=0.61/0.85/0.53 (membership); 0.57/0.86 (audition); pop rho ns | null — killed the theoretically forced channel |
| T8 | within/across-mask committee overlap (anchor fact) | `overlap_check.py` | 0.688 vs 0.300 (uniform null 0.279), perm p<2e-4 | mask determines popularity — S1 |

Surrogate (quadratic MLP) tests:

| #   | test                                                                 | script                               | result                                                                                | verdict                                          |
| -----| ----------------------------------------------------------------------| --------------------------------------| ---------------------------------------------------------------------------------------| --------------------------------------------------|
| T9  | memorization-stage favorites, 10 inits, dseed0 (Adam, no wd)         | `surrogate_flow.py`                  | within-toy top-8 overlap 1.09 vs chance 1.14                                          | toy audition = pure init lottery, no mask signal |
| T10 | wd probe for toy grokking                                            | (one-off probe, script not retained) | groks at wd=1.0 by step 2000                                                          | enabling result only                             |
| T11 | grokked-toy committees, 10 inits x 3 masks, vs transformer favorites | `surrogate_grok.py`                  | within-mask consistency at/below chance; same-mask excess +0.042 vs cross-mask +0.049 | no transfer — S2                                 |
| T12 | tied-embedding variant (shared V for a,b)                            | `surrogate_tied.py`                  | same-mask +0.004 vs cross -0.005                                                      | no transfer; tying doesn't rescue                |

Mask-free proxy tests (identity-from-menu):

| # | test | script | result | verdict |
|---|------|--------|--------|---------|
| T13 | LP-support rule (final = support of max-min LP over e3000 top-8 menu; thresholds 1%/5%, menus at mem-end and e3000; 24 runs) | `lp_support.py` | exact 0-1/24; Jaccard 0.45-0.52 vs amplitude-topK baseline 0.64-0.73; LP keeps 6-8 of 8 | falsified — sparsity is not convex geometry |
| T14 | multiplicative census: ratio-2 pairs (31 committees, 20k perm) | `mult_census.py` | 6 obs vs 7.18+-2.52 | null — no harmonic subsidy law |
| T15 | ratio-3 pairs | `mult_census.py` | 3 vs 7.17, p(depl)=0.063 | ns |
| T16 | additive pairs (sum/diff of a pair also in committee) | `mult_census.py` | 6 obs vs 23.19+-6.69, p(depleted)=0.0018 | CONFIRMED depletion — P4's composition shadow; iid p invalid (mask clusters) — corrected inference + provenance in T33/T34 |
| T17 | smallest-feasible-K rule (theta in {0.25,0.274,0.30}, all 2^8 menu subsets scored by lp_relM) | `floor_repair.py` | Jaccard 0.45-0.66 vs baseline 0.73; K-match at best 15/24 | falsified — model does not minimize K |

Init-lottery localization (the positive arc):

| # | test | script | result | verdict |
|---|------|--------|--------|---------|
| T18 | component x epoch AUC map at epoch 0 (5 stages, 24 runs, t-test across runs) | `ticket_map.py` | e0 AUC: mlp 0.704 (p<1e-4), align 0.702, emb 0.664, unemb 0.579, logit 0.561 | committee readable at init in weight space — overturns logit-level conclusion |
| T19 | combined rank score; ramp; per-run AUC vs reconfigurer status | `ticket_map2.py` | combined e0 0.725, e1000 0.991; reconfigurers 0.699 vs others 0.730, p=0.22 | init predicts; reconfigurers NOT low-signal runs |
| T20 | component knockout (randomize one component x3 draws, re-score; 24 runs) | `locate_ticket.py` | baseline 0.704/0.702; rand W_E 0.518/0.529; rand W_in 0.711/0.692; rand attn 0.602/0.610; rand W_out 0.694; rand W_U 0.659 | ticket located in W_E; He et al. per-neuron alignment ruled out — P1, S3 |
| T21 | gradient chain: growth exponent (log coeff e50->e150) | `locate_ticket.py` | AUC(exponent->final)=0.943; corr(init, exponent)=0.153 | exponent predicts, init doesn't set it |
| T22 | mediation: head start vs exponent | `mediation.py` | corr(init, log c50)=+0.329 = corr at c150; AUC(c50)=0.951, (c150)=0.985, (init)=0.704 | OVERTURNED as a dichotomy by T35 — carrier claim stands, head-start-vs-growth does not |
| T23 | veto/repair test: blind-draw (e3000 top-K, K=final K) margin pctile vs reconfiguration (relM_equal, 3000-draw nulls, 24 runs) | `veto_test.py` | reconf (n=14) mean pctile 42.5 vs loyal (n=10) 67.7, MW one-sided p=0.012; bottom-quartile draws 4/4 reconfigured, 0/10 loyal (Fisher p=0.094); repairs raise relM 12/14 | RETRACTED — survivor-floor circularity, see T32 |
| T24 | init W_E tilt statistics | `veto_test.py` | relative sd 0.088 vs chi2(2*128) prediction 0.088; committee-member mean z +0.546 (sd 0.470) | tilt = pure Gaussian-matrix noise — P1 |

Init surgery (causal; base `runs/og_seed0/seed27058`, predictions
pre-registered in the driver's output before any arm ran):

| # | arm | script | result | verdict vs prediction |
|---|-----|--------|--------|----------------------|
| T25 | control (untouched e0, fresh Adam, 12k epochs) | `surgery.py` | {14,49,52}, grok 3700 (orig 3500) | exact reproduction — determinism |
| T26 | collision: 12=fold(49+52) x2.25 energy | `surgery.py` | {12,14,29,49}; 12 peak 26,119 (control 94); 52 evicted; grok 3000 | prediction half-wrong (12 survived); mechanism-level right (degenerate trio broken) |
| T27 | collision margin ledger | `collision_ledger.py` | evict-52 repair LP 0.356 > swap-49 0.327 > keep-all 0.262; chosen #11/35 menu subsets; allocation 96.7% of LP optimum | margin-best single swap — P4 |
| T28 | boost-strong: 7 x2.25 | `surgery.py` | {7,29,49,52}; 7 peak 20,584 (control 208->0); 14 (=2*7) evicted; grok 2600 | adoption as predicted; eviction unpredicted |
| T29 | boost ledger | `boost_ledger.py` | evict-14 LP 0.386 > keep-all 0.366 > evict-49 0.339 > evict-52 0.177; chosen #8/35; allocation 95.0% of LP | margin-best swap again — P4 |
| T30 | suppress: 49 x0.5 energy | `surgery.py` | 49 peak 24,622 -> 172 -> 0; committee {14,26,29,52,53} (K 3->5, all recruits from menu); grok 5300 | annihilation as predicted; K growth unpredicted |
| T31 | boost-subtle: 7 x1.2 (=+2.3 sigma, natural scale) | `surgery.py` | {7,14,49,52}; 7 final 4,459; no eviction | adopted — natural-scale flip; also exposes graded-repair wrinkle vs T28 |

Adversarial re-analysis (2026-07-31; hostile re-runs of every load-bearing
positive claim; scripts in this directory):

| # | test | script | result | verdict |
|---|------|--------|--------|---------|
| T32 | T23 decomposition: reconfigurers' blind-draw pctile vs uniform 50 (the only non-circular content of T23) | `attack_veto.py` | reconf mean 42.6 (n=14): t p=0.28, Wilcoxon p=0.12, uniform-null p=0.17; loyal blind = final committee (floor, 67.7, p=0.026); counterexample seed31429 reconfigured at 99.6 pctile; repairs raise pctile 11/14; classification stable at e2000/e3000, K-free variant unchanged | T23 RETRACTED — MW p=0.012 is the floor restated |
| T33 | T16 dependence: mask-cluster bootstrap on per-committee excess; one-committee-per-mask; popularity-matched null; detector sweep (gap in top-8/12/16, fixed top-4) | `attack_census.py` | all 9 mask-means negative but one (n=1); cluster bootstrap p=0.0003; one-per-mask (n=9) p=0.088; popularity-matched p=0.0048; obs=6 under every detector | depletion survives correct inference — quote 0.0003, not 0.0018 |
| T34 | depletion provenance: additive count of blind e3000 top-K vs final, same 24 runs | `attack_census.py` | blind 20 obs vs 18.4 null (p=0.66, NO depletion); final 4 vs 18.4 (p=0.0035) | depletion arises between audition and convergence — repair-stage provenance for P4 |
| T35 | P2 window: interval-resolved corr(init W_E energy, log-amp growth); logit level at e0 | `attack_mediation.py` | corr(init, lc0)=+0.04, AUC(lc0)=0.56; corr(init, growth per bin)=+0.09/+0.12/+0.20/+0.16; AUC(init emb)=0.664 (reproduces T18) | no head start in the computed function; dichotomy unresolved — P2 narrowed |
| T36 | S1 inference: run-level mask-label permutation (correct unit); per-mask breakdown; excluding dseed 0 | `attack_overlap.py` | diff 0.388, run-level p=0.0003; ex-dseed0 diff 0.659, p=0.028; per-mask within 0.676/0.333/1.333/1.0/1.0 | S1 stands; original pair-permutation test deprecated |
| T37 | verification sweep: surgery arms vs claimed numbers; 8k/10k/12k drift; detector gap ambiguity (31 runs); menu-closure slack; epoch_00000 provenance | `attack_surgery.py` | every claimed peak/eviction/committee verified from raw spectra; zero drift 8k->12k; control groks 3700 vs original 3500 (same committee); detector fragile in 4/31; final within top-6 at e3000 in 22/24; train.py saves e0 before the first update | causal base verified; "exact reproduction" means committee, not trajectory; top-8 menu has slack |

Conjecture-seed probes (2026-07-31, post-review; NEW positives, not yet
hardened):

| # | test | script | result | verdict |
|---|------|--------|--------|---------|
| T38 | OV-circuit init readout: per-freq energy of sum_h W_O^h W_V^h W_E at epoch 0 (24 runs) | `conjecture_probes.py` | AUC 0.724 vs 0.5 (p=3e-8); raw emb 0.664; single stage, no forward pass | the ticket reads better through attention's OV than from W_E directly — new best single-stage init readout |
| T39 | replacement repair trigger: blind e3000 draw's additive-pair count vs reconfiguration | `conjecture_probes.py` | P(reconf given pair in blind) 10/12 vs 4/12 clean, Fisher p=0.018; non-circular check: reconf blinds enriched vs own random baseline (18 obs vs 11.6 exp), unlike margin (T32); grok-time corollary: reconf 7886 vs loyal 6255 epochs, MW p=0.084 (ns, but surgery agrees causally: clean hands grokked earlier) | additive degeneracy of the leader set is the live trigger candidate T23's margin failed to be — needs the same survivor-floor scrutiny at farm n before promotion |

Conjecture program (2026-07-31, same-day execution under the adversarial
standard; most conjectures FAILED — logged so they stay dead. Scripts:
`c3_kcost.py`, `c1_repair_rule.py`, `c4_c7_tests.py`, `c2c5_surgery.py`; new
runs in `runs/surgery2/`):

| # | conjecture / test | result | verdict |
|---|-------------------|--------|---------|
| T40 | C3: K = argmin of weight cost sum a^(2/3) s.t. margin gap, over menu subsets | argmin-K matches observed K in 4% (always-4 baseline 74%); cost is monotone under set inclusion, so ANY such functional trivially maxes K — conjecture ill-posed without a per-member fixed cost; K=3 committees rank 146/210 and 191/210 (counterexamples). Crumbs: realized allocations within ~3% of cost-optimal given S; 2/3-power ranks chosen sets slightly better than power-1 (paired p=0.0007, -4 pctile) | REFUTED / ill-posed |
| T41 | C1: pre-specified arithmetic repair rule (evict lowest-amp additive violator, recruit from menu avoiding violations) applied to blind draws | Jaccard 0.810 vs blind 0.730; exact 13/24 vs 10/24; evictee identity 10/10, recruit 9/10. BUT amplitude-churn null gets 0.766 and 8/10; incremental arithmetic edge = 3 runs (p~0.10); 4 reconfigs have violation-free blinds; 2 loyal runs falsely "repaired" | partial support, not decisive at n=24 — farm question |
| T42 | C4: full-path init readout beats the 0.725 combined score | PRIMARY (align+mlp+OV vs align+mlp+emb): +0.008, p=0.55 — failed. OV single-stage 0.724 (T38) stands but is subsumed by forward-pass scores. Init-readout ceiling stays ~0.73. OV-rank adoption rule also failed calibration on the known x1.2 arm | deflated |
| T43 | C7: phase-unlocked fraction explains the CE anomaly | anomaly ratio independently replicated (2.8-691x) but unexplained: rho(log ratio, unlocked)=0.20 p=0.34; vs reconf p=0.36 | unsupported; anomaly still open |
| T44 | C6: repair delays grokking | natural runs: reconf 7886 vs loyal 6255 epochs, MW p=0.084 (ns); additive-count corr rho 0.23 p=0.28. Causal hints only: boosted arms grok earlier than control (2600/3000/2700/3300 vs 3700), suppress later (5300) | directional, unproven |
| T45 | C2: triple-implant surgery — clean trio {5,37,22} out-survives degenerate trio {5,37,42=fold(5+37)}, both x1.5, thirds energy-matched (predictions pre-registered in `c2c5_log` before training) | REVERSED: deg arm kept {5,37} (committee {5,14,37,49}, grok 2700); clean arm kept only {5} (committee {5,29,49,52}, grok 3300). Both non-menu thirds died (peaks 135/804 -> 0) — x1.5 is below launch threshold for outsiders, so the arithmetic contrast was never reached. Unplanned finding: swapping one sub-threshold freq (42 vs 22) changed WHICH incumbent got evicted (52 vs 14) — committee assembly is chaotic under sub-threshold init perturbation. Menu closure and additive-cleanliness held in both finals | primary prediction FAILED; mechanism untested at this dose; new chaos observation |

C5 dose-response arms (x1.05, x1.10 on freq 7) were cancelled mid-run; the
pre-registered predictions (both NO-adopt; emb-rank rule, x1.10 at rank 9 =
boundary case) remain in the script/log for a future farm.

### Addendum 2026-08-02: carrier decomposition — P1/P2 REVISED

Full audit + numbers in `findings/CLAIMS_DOSSIER.md`; scripts
`tilt_carrier.py` (readout-level, 24 runs, stable crc32 seeds) and
`tilt_transplant.py` (causal; 6 runs in `runs/transplant/`, dose arms in
`runs/surgery2/dose_{105,110}`; predictions pre-registered in-script).

| # | test | result | verdict |
|---|------|--------|---------|
| T46 | W_E edit x readout: flat_energy (remove tilt, keep directions) | align AUC 0.637 vs baseline 0.702 (paired p=1.3e-4) | tilt carries only ~0.065 of the signal |
| T47 | scram_dir (keep exact energies, scramble directions) | 0.587 (p=1.3e-4); perm_energy 0.605 | directions carry MORE than energies |
| T48 | tilt transplant: donor energy spectrum onto same-mask recipient init, 6 ordered pairs, pre-registered P-TR1 (D-pull > R-retention) | D-unique adopted 2 vs R-unique kept 8 | **P-TR1 FAILED** — full-spectrum energy swap does not transfer identity |
| T49 | dose arms (freq 7, base seed27058): x1.05 (post-boost emb rank 18), x1.10 (rank 9, boundary) | x1.05 NOT adopted (auditions then evicted, peak 270 -> 0) as pre-registered; x1.10 weakly ADOPTED (f7 final 1477, ~9x background) — rank rule failed at its boundary case | full dose curve now monotone in final target amplitude: 0 / 0 / 1477 / 4459 / 19735 for x1.0/1.05/1.10/1.2/2.25; adoption threshold in (1.05, 1.10); use final amplitude, not detector membership, as the farm outcome |

REVISED P1/P2 statement: committee identity is set by W_E's random draw
(knockout unchanged); within the draw, the per-frequency energy tilt is a
*causal control knob* at concentrated +2-sigma doses (surgery, all arms
verified 08-02) but the *minority carrier* of natural identity — the
majority carrier is the draw's frequency-subspace geometry as transmitted
through attention's OV pathway (consistent with T38's OV readout 0.724 and
the W_in-knockout null). The orthWE flattening result is hereby explained
as over-determined (QR rewrites the dominant direction carrier, not just
the tilt) and should not be cited as a tilt-necessity proof.

### Addendum 2026-08-02b: megadataset re-review (78–80 selection runs)

Scan of all `runs/` configs: 122 dirs; 80 grokked spectra-logged
selection runs (24 natural + 56 intervention: orthWE 8, phase2 12+4,
eff 19, combined 2, surgery 5, surgery2 4, transplant 6), 9 masks, 26
init seeds, 3 primes. Scripts: `breadth_audit.py`, `ext_readout.py`,
`ext_twin.py`, `ext_floor.py`.

| # | test | result | verdict |
|---|------|--------|---------|
| T50 | epoch-0 readout by cohort: natural-normal (24), surgical-normal (15), orth-FLAT-energy (41) | align AUC 0.702 / 0.752 / **0.657 (p=1.9e-12)**; raw emb-energy readout in orth cohort 0.477 ≈ chance (flatness sanity check) | **carrier revision CONFIRMED on independent cohort**: with the energy tilt identically zero, init geometry still predicts the final committee |
| T51 | across-dynamics twins: same flat init trained under up to 10 dynamics variants (plain/noise×2/tilt/CVaR/wd-sweeps), cell 2034 | within-init-seed J 0.354 vs across-seed 0.113, permutation p < 1e-4 (37 runs, 154/460 pairs) | init geometry co-determines identity ROBUSTLY across training dynamics — the well-posed replacement for the orthWE necessity argument |
| T52 | LP-scored floor + clustered depletion on full zoo (dose_110 detector-corrected) | floor: **0/80 below 25th LP pctile** (min 31.8, mean 80.1) — incl. every intervention family; depletion: final 22 viol vs 66.1 exp, cluster bootstrap p=0.0003 (9 masks); blind 40 → final 22 | floor and depletion replicate at full breadth; caveat: the blind-dirty→final-clean provenance contrast is driven by the natural cohort (20→4; interventions 20→18) |

### Addendum 2026-08-02c: the ticket variable unified (T53–T54)

| # | test | script | result | verdict |
|---|------|--------|--------|---------|
| T53 | knockout WITHIN the orth-flat cohort (fresh QR frame / rand attn / rand W_in; 41 runs, 3 draws each) | `ext_orthknock.py` | baseline align 0.657; fresh-QR-frame 0.549 (p=1.2e-7), rand-attn 0.560 (p=5.3e-6), rand-W_in 0.653 (ns) | in flat runs the signal is the JOINT W_E-frame x attention geometry; W_in irrelevant in both cohorts |
| T54 | closed-form ticket T_k = sum_h \|\|W_O^h W_V^h W_E\|_k\|\|^2 at epoch 0 (no forward pass) | `ext_ovread.py` | AUC 0.724 natural (=T38), **0.661 orth-flat** (~= full forward readout 0.657) | ONE scalar per frequency carries essentially the whole init lottery in both regimes |

**Unified statement (supersedes the energy-vs-direction bookkeeping):** the
committee ticket is the per-frequency OV-transmitted embedding energy at
init, T_k ≈ E_k(W_E spectrum) x G_k(W_E-frame-to-OV alignment). Natural
init: both factors vary; surgery manipulates E_k (causal dose curve);
attn-knockout removes only G_k (0.70 -> 0.61) while W_E-knockout removes
both (-> chance). Orthogonal init: E_k ≡ const, the lottery runs on G_k
alone — QR-frame or attn knockout each kill it (T53), the closed form
still predicts (T54), and the same G_k ranking persists across 10 training
dynamics (T51 twins). The transplant failure (T48) is explained: copying
E_k without G_k barely moves the product. All seven results are one
variable seen from different angles.

### Addendum 2026-08-03: overnight suite (T55–T58) — logs in
`findings/logs/overnight-2026-08-02/`

| # | test | script / runs | result | verdict |
|---|------|--------|--------|---------|
| T55 | double-flat cohort completed to n=16 (orthogonal W_E + isometric attention; every run E_k and T_k flat to 1e-9) | `doubleflat.py` / `runs/doubleflat/` | ALL epoch-0 readouts at chance: emb 0.513, T_k 0.510, fwd_mlp 0.511 (p=0.81), fwd_align 0.564 (ns); knockouts all null; He per-neuron variable 0.565/0.581 (p=0.057/0.020 uncorrected over 6 probes — noise-level after correction). All 16 grok, menu-closed, K 3–5 | **P-DF1 CONFIRMED at n=16.** The 2026-08-02 run-1 "W_in floor" (fwd 0.94, W_in-knockout collapse) was a single-run fluke — RETRACTED. He-et-al reactivation NOT supported. Double-flat init = no readable init lottery; selection still works |
| T56 | alignment-knob causal test: rotate freq-7 W_E plane toward OV at FIXED energy, angles solved for T_7 gains 1.05/1.20/2.25 (energy err ≤ 4e-16) | `gk_rotate.py` / `runs/gkrotate/` | gain 1.05: rejected (peak 252→0); gain 1.20: adopted (final 1,582 ≈ energy-dose 1.10's 1,477); gain 2.25: dominant (19,849 ≈ energy's 19,735) with the SAME eviction (14) and SAME final committee {7,29,49,52} as the energy ×2.25 arm | **P-C1 CONFIRMED — T_k is the causal variable.** Energy and alignment knobs are interchangeable at matched transmitted gain, down to committee identity |
| T57 | cross-mask energy dose farm: 3 new bases (masks ds1/ds2/ds2034), targets 55/34/44 (all \|z\|≤0.02), doses 1.0–2.25; + seed27058 ×1.5 gap-fill (adopted, 4,450) | `dose_farm.py` / `runs/dosefarm/` | within-context monotone ✓; but adoption threshold is CONTEXT-DEPENDENT: (1.05,1.10] for 27058/f7, (1.50,2.25] for 21245/f55 and 33428/f44 (both evict an incumbent), >2.25 for 51376/f34 (peak 214→0 at max dose). Controls at 12k differ from 30k base committees on 2/3 bases (horizon effect, needs base-12k comparison before calling it violation) | **P-B2 (universal threshold) FALSIFIED.** Dose-response is real and monotone per context; the threshold varies ~an order of magnitude with mask/target context. State claim 3a as within-context dose control |
| T58 | collision farm: 4 engineered additive trios across 3 masks, t=fold(i±j) ×2.25 | `collision_farm.py` / `runs/collisionfarm/` | trio broken 4/4 (6/6 including T26/T28); t adopted 4/4 — incl. f35 on seed51376 where the plain dose target f34 failed at the same ×2.25 (context again); repairs LOCAL in 1/4 only (multi-member churn in 3/4, one 3-member reconfiguration); 3/4 finals additive-clean, 1/4 carries a harmonic pair {3,6} | **P-D1/P-D2 CONFIRMED at breadth** — repair-on-demand is mask-general; P-D3 (single-member locality) WEAKENED; P-D4 mostly holds |

Consolidated hierarchy statement (final form): the readable init lottery
has exactly TWO carriers — E_k (embedding energy) and G_k (embedding-to-OV
alignment), unified as T_k = E_k·G_k which is causally validated by knob
interchangeability (T56). Flatten E_k → the lottery runs on G_k (T50/T51,
AUC 0.66). Flatten both → NO probe reads the outcome (T55) yet selection
proceeds normally — identity becomes pure dynamical chaos, consistent with
T45. The hierarchy bottoms out at "unreadable," not at a third carrier.

### Adversarial-review notes (attack here first)

1. **Multiple comparisons.** This investigation ran ~17 hypothesis tests.
   The nulls are reported as nulls. The surviving positives: T18/T20
   (p<1e-4 across 24 independent runs, then confirmed by an orthogonal
   causal method); T16 (single pre-motivated statistic, corrected
   cluster-aware p=0.0003, provenance T34). T23 did NOT survive (T32).
   The weakest positives remain flagged: T6 (p=0.054 after 5 looks —
   treat as noise), T15 (p=0.063).
2. **Committee detection** is the largest-log-gap heuristic on sorted
   |phase-locked coeff| (`findings/mask_lottery.py:committee_from_coeffs`,
   cut searched in top 12). All composition statistics inherit it. The
   surgery conclusions do not (target trajectories are read directly).
3. **Researcher degrees of freedom in menus**: "menu" = top-8 at e3000 (or
   memorization end where stated). The 23/24 containment result justifies
   the choice post hoc; T13/T17 report both epochs.
4. **T23 circularity**: confirmed and fatal — not the mild K-flavor
   anticipated here, but the survivor-floor decomposition (T32). The
   K-free variant was also run (T32) and changes nothing.
5. **Surgery scope**: one base run, one dose per arm, 12k epochs (original
   ran 20k; committees at 12k could in principle still drift — the flat
   spectra tails argue not, but extended arms were not run). Replication on
   more base runs + a dose-response curve is listed as required work.
6. **AUC dependence structure**: stages within a run are correlated; all
   significance tests treat the run (n=24) as the unit, never the
   (run x frequency) pair.
7. **Surrogate mismatch**: the quadratic toy uses MSE (not CE), sigma=0.3
   init, N=256, and its own wd scale. "No transfer" is robust to the tied
   variant but has not been shown for a CE-trained toy.
8. **Init-tilt z (+0.546)** is descriptive, computed after the AUC results;
   its independent content is the chi-square match (0.088 vs 0.088), which
   is parameter-free.
9. **Session provenance**: exact console outputs for every test above were
   produced in one session (9b155e14); scripts are deterministic
   (fixed seeds) and re-runnable from the repo root.
