# Four-claim dossier: verification, critique, and lock-down designs

Status: 2026-08-02. Purpose: consolidate the evidence behind the four
headline claims for (a) a cold email to Neel Nanda and (b) the ICLR paper
(`findings/findings.md` §5 has the paper skeleton; this document is the
claim-by-claim audit that backs it). Every number below was RE-VERIFIED
today by re-running the named scripts (outputs in the session scratchpad;
all reproduced unless flagged).

The four claims as stated:

1. The model's preference over frequencies at init lives primarily in the
   init weights of W_E (sufficiency and necessity).
2. Orthonormalizing W_E gives varied frequencies (offered as a necessity
   proof — assessed below as weak, with replacement tests).
3. Modifying the frequency energy in W_E causes noticeable changes in the
   final post-grok committee (needs lock-down against the chaos
   phenomenon).
4. On tampering (or on a naturally bad draw), the model repairs its
   committee so it does not fall through a margin floor below which
   amplitude cannot compensate.

---

## Claim 1 — the init preference lives in W_E

**Status: STRONGEST claim. Intervention-grade on both sides. Ready for the
paper as-is; one decomposition gap now being closed.**

Evidence map (scripts in `findings/` unless noted; run from repo root):

| piece | script | verified numbers (2026-08-02 re-run) |
|---|---|---|
| Init readable at all (readout) | `ticket_map.py` (T18), `ticket_map2.py` (T19) | epoch-0 AUC: mlp 0.704, align 0.702, emb 0.664; combined 0.725; logits blind (0.561) |
| **Necessity** (readout knockout) | `locate_ticket.py` (T20) | randomize W_E → 0.506/0.537 ≈ chance; randomize MLP W_in → 0.707/0.700 (no change); attn partial 0.590/0.612; W_out 0.708, W_U 0.675 |
| **Sufficiency** (causal surgery) | `surgery.py` (T25–T31), verified from raw spectra by `attack_surgery.py` (T37) | ×1.2 energy boost (+2.3σ, inside natural noise) → adoption; ×2.25 → dominance (peak 20,584) + eviction of harmonic relative; ×0.5 suppress → annihilation (peak 24,622 → 172 → 0); control reproduces base exactly; committees drift-free 8k→12k |
| The tilt is quenched noise | `veto_test.py` (T24) | per-freq W_E energy relative sd 0.088 = chi²(2·128) prediction 0.088; winners start +0.55σ |
| He et al. mechanism ruled out | `locate_ticket.py` (T20/T21) | per-neuron alignment variable carries nothing beyond energy; corr(init, growth exponent 50→150) = 0.153 while AUC(exponent→final) = 0.943 |

Precision on what "necessity/sufficiency" means here (worth stating this
carefully in the paper and the email):

- *Necessity*: W_E is necessary for the **predictive signal** — knock it
  out and the epoch-0 committee readout collapses to chance while every
  other component knockout leaves it (nearly) intact.
- *Sufficiency*: editing **only W_E per-frequency energy** at epoch 0 is
  sufficient to **causally control membership** at the natural noise scale
  (±2σ), including targeted adoption, domination, and annihilation.
- What was NOT yet shown: that the *energy spectrum specifically* (rather
  than other structure inside W_E, e.g. frequency-subspace directions) is
  the carrier. The orthWE test (claim 2) tried and failed to isolate this.
  Two new tests close the gap — see claim 2's "replacement tests."

Reproducibility flag (fix before paper): `locate_ticket.py` seeds its
knockout RNG with Python's salted `hash(str(d))`, so exact knockout numbers
wobble by ~±0.01 between interpreter sessions (0.518/0.529 documented vs
0.506/0.537 today; conclusion unchanged). `tilt_carrier.py` uses stable
crc32 seeds; port that back.

---

## Claim 2 — the orthWE test, and why it under-delivers as a necessity proof

**Status: agreed — weak as necessity evidence. Keep as a descriptive
result; replace the necessity argument with the two new tests below.**

What exists: `grok/model.py:_embed_init` (QR-orthonormalize the same
Gaussian draw; per-freq energy exactly flat at 2.0000, rest of the init
stream seed-identical), 8 runs in `runs/orthWE/p-113/seed{2034,3604}`,
analysis `scripts/orthwe_analysis.py`. Verified today: 4/4 paired twins
(same init seed, tilt removed) pick near-disjoint committees — Jaccard
0.12, 0.14, 0.00, 0.00; all 8 orth runs grok to acc 1.000; committee
margins improve only mildly and non-significantly (mean lp-relM percentile
84.8 vs 79.8 baseline, MW p = 0.18).

Why it is a bad necessity proof (sharpening the user's instinct):

1. **Not surgical.** QR removes the per-frequency energy tilt AND all
   cross-token geometry at once (it forces W_E^T W_E = I, changing every
   angle between token embeddings). The intervention edits several
   variables; attributing the outcome to "the tilt" is under-determined.
2. **The outcome measure is uninformative under chaos.** T45
   (`c2c5_surgery.py`, `runs/surgery2/`) showed committee assembly is
   chaotic to *sub-threshold* init perturbations: swapping one
   energy-matched, arithmetically-equivalent dead frequency changed which
   incumbent got evicted. Given that, "the committee changed after
   flattening" is what ANY init perturbation produces. Jaccard ≈ 0 against
   the twin has almost no discriminating power for the tilt specifically.
3. **Quantification has no null.** There is no principled reference for
   "how much committee change proves the tilt mattered": same-seed
   determinism gives J = 1, any perturbation gives J ≈ cross-seed baseline
   (which within a mask is itself 0.1–0.5). The test can only land on one
   of two uninformative endpoints.
4. n = 4 pairs.

What the orthWE result IS good for (keep, reframed): the tilt is *not*
necessary for grokking, committee formation, or committee quality — flat
starts grok fine, form normal-looking committees, and selection stays
utility-blind (percentile does not jump). Identity moves; competence and
selection pressure don't. That is a real and quotable negative result. A
second real positive from that program: flat-start committees are partially
reproducible across dynamics variants (twin J 0.33–0.50 vs 0.00–0.14 for
tilt-removal) — the residual selector is the specific orth draw's phase
structure, which independently supports "init microstructure decides."

**Replacement tests (built and launched today):**

- `findings/tilt_carrier.py` (training-free, readout-level, n = 24 runs).
  Surgical Fourier-space edits of epoch-0 W_E feeding the same forward-pass
  readout as T20: *perm_energy* (permute the 56 per-freq energies, keep
  each frequency's subspace directions — destroys tilt information only),
  *scram_dir* (redraw directions, keep exact energies), *flat_energy*
  (equalize energies, keep directions — the surgical version of what
  orthWE wanted to be). Necessity of the energy spectrum = perm_energy and
  flat_energy collapse AUC to ~0.5; sufficiency = scram_dir stays at
  baseline ~0.70. This is well-posed because it predicts the *natural*
  run's committee — no chaotic re-training in the loop.
- `findings/tilt_transplant.py` (causal, 6 training runs). Among the three
  same-mask naturals `runs/seed0/seed{37099,55327,93077}`: for each ordered
  pair, rescale recipient R's W_E Fourier blocks to donor D's energy
  spectrum (R keeps directions + every other weight), train 12k epochs.
  Discriminating outcome with two *named* attractors: does the hybrid's
  committee contain more D-unique or R-unique members (pre-registered
  P-TR1: D > R pooled over 6 transplants)? This is the necessity+sufficiency
  test the orth design couldn't be — same-mask pairing controls popularity,
  and chaos is handled by scoring distributional pull, not identity.

---

## Claim 3 — energy edits change the final committee (lock-down)

**Status: TRUE but must be split in two before anyone hostile reads it.**

The chaos phenomenon that forces the split (T45, `runs/surgery2/`):
triple-implant arms boosted {5, 37, +third} ×1.5; both non-menu thirds died
(×1.5 is below launch threshold for outsiders), and the two arms — which
differ ONLY in an energy-matched, sub-threshold, dead-on-arrival third
frequency (42 vs 22) — produced different committees ({5,14,37,49} vs
{5,29,49,52}): the same two implants survived, but a *different incumbent
was evicted*. Sub-threshold init detail changes bystander composition.

The locked-down formulation:

- **(3a) Targeted effect — dose-controlled, deterministic, monotone.** For
  the boosted frequency ITSELF: ×1.0 → not adopted; ×1.2 → adopted
  (final 4,459); ×2.25 → dominant (19,735); ×0.5 on an incumbent →
  annihilated. Verified from raw spectra today (`attack_surgery.py`), zero
  committee drift 8k→12k. Dose arms ×1.05 / ×1.10 (pre-registered
  predictions: NOT adopted; ×1.10 is the rank-9 boundary case) were
  cancelled mid-run in July and are being completed today
  (`tilt_transplant.py`, landing in `runs/surgery2/dose_{105,110}`),
  giving a 5-point dose curve on one base run.
- **(3b) Bystander effects — real, but chaotic.** Boosts/suppressions
  reliably cause *some* recomposition of the rest (evictions, K-growth),
  and every recruit comes from the audition menu (menu closure, 23/24
  natural + all surgery arms), but WHICH bystander moves is not predictable
  from the intervention — treat as sensitive dependence, do not claim
  control. This is presentable as a *positive* finding: it independently
  confirms that identity is quenched randomness (P1) rather than the
  output of a computable rule.

What the paper still needs for 3a (unchanged from findings.md §5, now
sharper): dose–response replication on 2–3 more base runs with spectra
every ~5 epochs for the first 300 (settles P2's head-start-vs-growth-rate
question for free), and the adoption rule stated as "post-boost init-energy
rank ≤ menu size" so it is falsifiable per-arm.

Note on scope: the phase-2 result that *tilted ERM* (t = 5 loss tilt)
steers committees (+10.3 paired percentile, `scripts/phase2_analysis.py`,
verified today) belongs to the "selection pressure" program, NOT to claim
3 — it modifies the loss, not W_E. Keep them separate in the paper.

---

## Claim 4 — repair against a margin floor

**Status: the floor is solid; "repair exists" is solid; the *trigger* and
the wd part of the story need rewording. This is the claim to state most
carefully in the email.**

What survives adversarial review (all re-verified today):

| piece | script | verified numbers |
|---|---|---|
| Floor (survivor enrichment) | `attack_veto.py` (T32), `scripts/margin_analysis.py` | loyal survivors' committee percentile 67.7, vs uniform p = 0.026; 0/15 committees in bottom quartile (older margin analysis, P = 0.75^15 = 0.013); every menu contains catastrophic subsets (relM 0.006–0.06) that are never chosen |
| Repair exists, is local, strips additive degeneracy | `attack_census.py` (T33/T34) | blind e3000 draws: NO depletion (20 obs vs 18.4 null, p = 0.66); same-run finals: depleted (4 vs 18.4, p = 0.0035); cluster-bootstrap census p = 0.0003 |
| Repair on demand (causal) | `surgery.py` collision arm + `collision_ledger.py` | engineered additive trio {12,49,52} → single-member repair (52 evicted), chosen swap was margin-best of the options |
| Repair rule (predictive form) | `c1_repair_rule.py` (T41) | arithmetic repair of blind draws: Jaccard 0.810 vs blind 0.730, exact 13/24 vs 10/24; evictee identity 10/10, recruit 9/10; BUT amplitude-churn null reaches 0.766 (paired Wilcoxon p = 0.19) — suggestive, not closed |
| Amplitude compensation above the floor | homeostat material (`runs/homeostat/`), seed55794 | a weak-but-feasible hand (relM 0.115) is carried at ~2.4× amplitude instead of repaired — the floor is soft at its edge |

What does NOT survive (do not put in the email):

- **"Margin triggers repair"** — RETRACTED (T32). Reconfigurers' blind
  draws are statistically random (mean pctile 42.6 vs 50, p = 0.12–0.28);
  the old p = 0.012 was the survivor floor restated; a 99.6th-percentile
  draw reconfigured anyway. Live trigger candidate: additive-degeneracy
  count of the leader set (T39: Fisher p = 0.018, non-circular) — needs
  farm-scale replication before promotion.
- **The weight-decay wording of the floor.** The user's phrasing "a margin
  floor below which weight decay prevents amplitude from repairing" is the
  July story; it was overturned 2026-08-01 (`runs/epsfloor/`, 8 seeds):
  the ~18.4-nat equilibrium constant that set the floor's numeric value is
  **Adam's ε**, not wd — eps 1e-8 → 1e-10 moves the train-CE floor 1e-7 →
  8.3e-10 (8/8 seeds), while a wd anneal moved nothing. wd's actual roles
  (efficiency sweep, `scripts/eff_analysis.py`): audition clock + per-member
  norm tax; it prices K, not amplitude. The *structural* floor argument
  survives and is the durable part: CE demands minGap = A·relM ≳ ~19 nats;
  required amplitude scales as 1/relM, so below some relM the required
  amplitude is not reachable in the transient — committees below the floor
  are never kept. But the floor's numeric value (relM ≈ 0.27 from
  18.4/A_ceiling) matched observation once at n = 8 and the ceiling's own
  mechanism is unestablished. State the floor as *empirical* (bottom
  quartile never chosen, degenerate sets always repaired) with the
  amplitude-cost argument as *interpretation*, and keep ε/wd out of the
  headline.

Honest one-line version for the email: "committees that would be
Diophantine-degenerate get locally repaired between audition and
convergence (composition signature p ≈ 3e-4, on-demand reproduction via
engineered invasions); what triggers the repair is open — we killed our own
margin-trigger hypothesis on re-analysis."

---

## Extension results (2026-08-02)

### 1. Carrier decomposition (`tilt_carrier.py`, 24 runs) — DONE, surprising

Surgical Fourier-space edits of epoch-0 W_E, same forward-pass readout as
the T20 knockout (align AUC; baseline 0.702, chance 0.5):

| edit | keeps | destroys | align AUC | paired p vs baseline |
|---|---|---|---|---|
| flat_energy | directions | energy tilt | **0.637** | 1.3e-4 |
| perm_energy | directions (energies permuted) | tilt info | 0.605 | 3e-6 |
| scram_dir | exact energy spectrum | directions | **0.587** | 1.3e-4 |

**Neither pure story holds.** Removing the tilt entirely costs only ~0.065
of AUC (0.702 → 0.637): the frequency-subspace *directions* carry the
larger share of the epoch-0 signal. Keeping energies but scrambling
directions retains less (0.587). Interpretation consistent with T38 (OV
readout 0.724 > raw emb energy 0.664) and with the W_in-knockout null: the
ticket is the per-frequency **transmitted** energy — W_E's draw as
projected through the random attention (OV) pathway — of which the raw
energy tilt is one factor and W_E→OV alignment the other. Both factors
live on W_E's draw, which is why randomizing W_E kills everything (T20)
while randomizing W_in changes nothing and randomizing attention hurts
partially (0.59–0.61).

Consequences:
- P1/P2's wording "committee identity is set by per-frequency energy noise
  in the embedding" is too narrow as a statement about the *natural*
  signal; correct version: "set by the embedding's random draw; the
  per-frequency energy component is causally sufficient to control
  membership (surgery), but the draw's geometric alignment with the
  attention pathway carries at least as much of the predictive signal."
- The orthWE misreading is now mechanistically explained: QR rewrites the
  directions (the bigger carrier), not just the tilt — its total committee
  scrambling is over-determined.
- New readout ceiling question: does OV-projected energy (0.724) equal the
  directions+energies total, i.e. is the transmitted-energy variable THE
  sufficient statistic? Cheap follow-up: scram_dir while preserving each
  frequency's OV-transmitted energy.

### 2. Tilt transplant (`tilt_transplant.py`, `runs/transplant/`) — DONE,
**pre-registered primary prediction FAILED, and that is the finding**

Design: among the three same-mask naturals `runs/seed0/seed{37099,55327,
93077}`, each ordered pair (R ← D) got R's epoch-0 init with every W_E
frequency block rescaled to D's energy spectrum (R keeps directions and all
other weights); 12k epochs each.

| transplant | final committee | D-unique adopted | R-unique kept |
|---|---|---|---|
| 37099 ← tilt(55327) | {9,14,15,25,42} | 0 | 2 ({15,42}) |
| 37099 ← tilt(93077) | {15,25,42} | 0 | 1 |
| 55327 ← tilt(37099) | {1,7,44} | 0 | 2 ({1,7}) |
| 55327 ← tilt(93077) | {5,7,44} | 0 | 1 |
| 93077 ← tilt(37099) | {2,7,22,41,42} | 0 | 1 |
| 93077 ← tilt(55327) | {2,7,27,41,42} | 2 ({7,27}) | 1 |

**P-TR1 tally: donor-unique adopted 2 vs recipient-unique kept 8**
(prediction was D > R). Swapping the ENTIRE natural energy spectrum (±1σ
scale changes on all 56 frequencies) does not hand identity to the donor —
the recipient's direction structure (plus non-W_E init) wins. Committees
did move substantially off the recipient's natural committee (chaos, as
expected: only 8/14 R-unique members survived; several members from the
shared menu pool appeared), but with no directional pull toward the donor.

Combined reading of extensions 1+2 — the REVISED claim-1 statement:

> Committee identity is set by W_E's random draw at init (knockout:
> necessity of W_E for the epoch-0 signal). Within that draw, the
> per-frequency energy tilt is a **causal control knob** — a concentrated
> +2.3σ boost on one near-menu frequency flips membership, ×2.25 dominates,
> ×0.5 annihilates — but it is the **minority carrier of natural
> identity**: removing it costs only 0.065 AUC (0.702 → 0.637), and
> transplanting a full donor spectrum transfers essentially nothing (2 vs
> 8). The majority carrier is the draw's frequency-subspace geometry as
> transmitted through the random attention pathway (scram_dir 0.587;
> OV readout 0.724 — best single stage).

This resolves the apparent tension: surgery works because it is a large,
concentrated intervention at the adoption margin; natural selection among
frequencies is decided mostly by transmitted-energy geometry, of which the
raw tilt is one factor.

### 3. Dose arms (`runs/surgery2/dose_{105,110}`, base seed27058) — DONE

- ×1.05 (post-boost emb rank 18): **NOT adopted** (f7 auditions — in its
  own e3000 top-8 — then evicted, peak 270 → 0; committee {14,49,52}
  unchanged; grok 3300) — pre-registered P-D1 ✓.
- ×1.10 (post-boost emb rank 9, the boundary case; pre-registered NOT
  adopted): **weakly ADOPTED** — f7 final 1,477 (~9× above background) —
  the rank-≤-8 rule FAILED at its boundary case. Detector caveat: the
  formal committee {7,14,28,49,52} includes f28 at amplitude 172, which is
  background level (the control run's dead f7 sat at 208) — a largest-log-
  gap artifact of the kind T37 flagged; the substantive content is f7.

**The full dose curve on seed27058 freq 7, in the detector-free outcome
variable (target's final |coeff|):**

| dose (energy) | ×1.0 | ×1.05 | ×1.10 | ×1.2 | ×2.25 |
|---|---|---|---|---|---|
| f7 final | 0 | 0 | 1,477 | 4,459 | 19,735 |

Monotone, with the adoption threshold between ×1.05 and ×1.10 (≈ +0.6 to
+1.1σ of natural noise) on this base run. Two lock-down consequences for
claim 3a:
1. Use **final target amplitude** (continuous, detector-free) as the
   primary outcome in the dose-response farm, with binary adoption
   (~10× background) as the derived threshold — this sidesteps both the
   detector fragility and the chaos-in-bystanders problem.
2. The simple init-rank rule is falsified at the boundary; the farm should
   fit adoption threshold vs base-run context (target's audition rank,
   menu gap) rather than assert a fixed rank cutoff.

## Breadth audit (2026-08-02, `findings/breadth_audit.py`)

Full scan of `runs/`: 122 run directories with configs; **78 grokked,
spectra-logged, selection-relevant runs** (78/78 of those grokked) spanning
**9 train/test masks** ((p, data_seed) pairs), **26 init seeds**, **3
primes** (113 ×70, 127 ×2, 157 ×2), one task (add). 24 are natural
dynamics (the canonical ensemble); 54 are intervention families that still
run fresh selection: orthWE (8), phase2 noise/tilt (16), eff-A/B/D/E/G
(19), surgery (5), surgery2 (4), transplant (6). Excluded as non-selection:
distill/eviction/interp/onehot warm-starts; epsfloor has no spectra.

What each ensemble-level claim now rests on (quick-pass scoring:
relM_equal percentile, iid violation expectation — redo with LP + mask
clustering for the paper):

| claim | natural (n=24) | all selection runs (n=78) |
|---|---|---|
| Margin floor (≥ 25th pctile) | **24/24** (P = 0.75²⁴ ≈ 1e-3) | **77/78** after correcting the dose_110 detector artifact; the one true exception is eff-B/seed4242 (19.4th pctile) — a wd=2.5 arm, i.e. altered-optimizer, consistent with the floor being an optimizer equilibrium |
| Additive depletion | 4 obs vs 18.4 exp (p = 0.0035) | **23 obs vs 63.4 exp** — replicates at ~2.8× depletion across every family; informative exceptions: eff-A (loss-thrash arm) is *enriched* (6 vs 4.0) and sustained-noise arms are weakly depleted — the interventions that degrade committee quality also erode the repair signature |
| Menu closure (final ⊆ own e3000 top-8) | 23/24 | **74/78** (75/78 artifact-corrected); failures concentrate in eff-A (2/4) — same pattern |
| Adopted-without-eviction arms carry the {7,14} additive pair | — | boost_subtle and dose_110 both retain 14 = 2·7 (1 violation each): tolerance at low implant amplitude, eviction at ×2.25 — the graded-repair wrinkle, now seen at two doses |

Breadth of the *causal* work (the thin part, unchanged by this audit):
surgery = 1 base run (p=113, ds=0); transplant = 3 same-mask init draws.
Everything causal lives on one mask. The dose-response farm across base
runs/masks remains the single highest-value addition before submission.

### Megadataset re-review verdicts (2026-08-02b; T50–T52 in findings.md)

- **Claim 1 — UPGRADED.** The carrier revision passed its falsification
  test on an independent cohort: the 41 orth-init runs have *identically
  zero* energy tilt, yet the epoch-0 geometry readout predicts their final
  committees at align AUC 0.657 (p = 1.9e-12), while the raw energy readout
  sits at chance (0.477) exactly as construction demands
  (`ext_readout.py`). W_E-localization now rests on: knockout (24 runs) +
  surgery (causal) + flat-energy cohort (41 runs) + transplant
  falsification. Final statement: identity is seeded by W_E's init draw;
  geometry is the dominant natural carrier; energy is the causally
  manipulable minority component.
- **Claim 2 — REPLACED, and then UNIFIED (2026-08-02c, T53–T54).** The
  elegant form: the ticket is one closed-form scalar per frequency,
  **T_k = Σ_h ‖W_O^h W_V^h W_E|_k‖² (OV-transmitted embedding energy at
  init)**, which factorizes as E_k (W_E spectrum) × G_k (frame-to-OV
  alignment). It predicts committees at AUC 0.724 (natural) and 0.661
  (flat cohort, ≈ the full forward-pass readout) with no forward pass
  (`ext_ovread.py`). Orthonormalization is then a *surgical* intervention
  after all — it sets E_k ≡ const, so the lottery runs on G_k alone:
  confirmed by the orth-cohort knockout (fresh QR frame or attn
  randomization each collapse the signal, W_in never matters,
  `ext_orthknock.py`) and by the across-dynamics twins (same G_k ranking
  under 10 dynamics: J 0.354 vs 0.113, perm p < 1e-4, `ext_twin.py`).
  This one variable also explains the transplant failure (copying E_k
  without G_k barely moves the product) and the tilt_carrier split
  (flat_energy keeps G_k → 0.637; scram_dir keeps E_k → 0.587). Claim 2's
  justification stops being "QR scrambled things and committees changed"
  and becomes "QR deletes one factor of the ticket; the other factor
  still runs the same lottery."
- **Claim 3 — unchanged by breadth** (the zoo adds no new causal arms):
  target-level dose-response solid on one base run; bystander chaos;
  farm still required. The megadataset DOES show every surgical/transplant
  run's committee remains menu-closed and above-floor — the interventions
  never produced a pathological committee.
- **Claim 4 — STRENGTHENED with one honest asterisk.** LP-scored floor:
  **0/80 runs below the 25th percentile** (min 31.8) — the two quick-pass
  exceptions were scoring artifacts (dose_110 detector; eff-B rises under
  LP allocation). Depletion replicates at full breadth (22 vs 66.1,
  cluster p = 0.0003). Asterisk: the repair-stage *provenance* contrast
  (dirty blind draw → clean final) is a natural-dynamics phenomenon
  (20→4); intervention families' auditions are already clean (20→18), so
  state provenance on the natural cohort only.

## Mapping to the paper (findings.md §5) and the email

- Claim 1 → Contribution 1+2 (causal localization; carrier = energy, not
  gradient alignment). Lead with this.
- Claim 2 (reframed) + tilt_carrier/transplant → the carrier-isolation
  paragraph inside Contribution 1; orthWE becomes supporting evidence that
  identity ≠ competence.
- Claim 3a → the dose–response figure; 3b (chaos) → folded into the
  "identity is quenched noise" discussion as independent confirmation.
- Claim 4 → Contribution 3 (repair operator with composition signature;
  trigger explicitly open) — the honesty here is a feature for a Nanda
  audience.
