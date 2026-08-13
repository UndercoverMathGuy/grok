# Induction-head lottery — pilot

Pivot target (2026-08-13): port the lottery → readout → surgery program from
mod-add frequency committees to **induction-head identity in a 2-layer
attention-only LM**. This pilot answers whether the program's premise holds
before any compilation work: is circuit identity a seed lottery here, and is
it readable from the init?

Files: `train_pilot.py` (self-contained torch trainer, 8 init-seeds × 3
data-seeds = 24 runs), `analyze_pilot.py` (lottery stats + weights-only init
readouts). Full setup docstrings in the scripts.

## Pre-registered pilot questions & predictions (committed before training)

- **PQ0 — concentration.** The induction circuit is sparse: the top L1 head
  carries the majority of induction-score mass and ablation delta-CE.
  *Prediction:* mean top-1 share ≥ 0.5; induction-score winner and ablation
  winner agree in ≥ 80% of runs. *If distributed instead (share ≲ 1/4), the
  lottery question is ill-posed in this task setup and the pilot fails.*
- **PQ1 — lottery.** Winner identity varies across init seeds: ≥ 3 distinct
  winning heads over 8 init seeds. Same-init twins (different data order)
  agree more than cross-init pairs — the init, not SGD noise, carries the
  identity. *Prediction:* twin agreement ≥ 2× cross-init agreement.
- **PQ2 — init readout.** The winner is predictable from the init weights
  alone by Elhage-style composition scores (K-composition from L0 OV into L1
  QK, plus L1 OV copying score). *Prediction:* top-1 hit rate beats chance
  (1/8) with binomial p < 0.05; pooled AUC ≥ 0.65. This is the T_k analogue;
  it does NOT need to be strong for the pivot to proceed — PQ0+PQ1 are the
  gate, PQ2 sets the difficulty of the readout problem.

Decision rule: PQ0 and PQ1 both land → proceed to the surgery phase (edit
composition scores at init to pick the winner — the compilation claim).
PQ0 fails → try LM-ish data (mixed random/repeat) before abandoning.
PQ1 fails (identity ~fixed across seeds) → the lottery framing dies here,
report honestly.

Chance CE ln(64) ≈ 4.16; induction solves the second half exactly, so
trained probe CE ≪ 1 is the sanity bar for "the circuit formed at all."

## Pilot v1 result (2026-08-13, task=copy) — PQ0 FAILED, task-design flaw

24 runs, 5090, 114s. conc_ind 0.131 vs 0.125 uniform → induction-score mass
statistically uniform over heads; no winner variable exists. Diagnosis: the
fixed-lag repeat (always T/2) is solvable by a purely positional head, so
content-matching induction is never forced and the pattern smears. Genuine
surviving signals: ablation concentration 0.29 (effective committee ~3-4
heads); same-init twin agreement 0.33 vs cross-init 0.14 (2.3×, clears the
prereg ratio through a noisy label). PQ2 unscoreable against a noise label.
Data: `runs_induction/pilot/` (pod), `notes/pivot/induction_pilot.json`.

## Pilot v2 (task=induction): variable-offset repeats — prereg

Fix: zipfian background (α=1), one segment (len 8–16) repeated at two
random offsets per sequence; lag varies per sequence so only content-based
prefix matching predicts the second copy; loss restricted to inducible
positions; induction score measured on the per-sequence correct
(query → key) edges. Same PQ0/PQ1/PQ2 predictions and decision rule as v1,
measured against BOTH winner definitions (induction-score argmax and
ablation-ΔCE argmax); ablation is primary if they disagree. Additional
prediction PQ0b: ablation concentration rises vs v1 (task now has a niche —
winner-take-most). If PQ0 fails again on ablation concentration, treat the
"induction committee" (top heads by ablation) as the identity variable,
mod-add style, before abandoning.

## Pilot v2 result (2026-08-13) + wd-sweep prereg

v2 (24 runs, 178s): task fix worked (delayed transition t_form 450-600, CE
0.242 vs oracle floor 0.221, all-8-head winner diversity, twin/cross winner
agreement 0.458/0.155 = 3.0×). But NO crisp identity variable: ablation
concentration 0.173 ~ uniform; gap-detected committees bimodal {1,7} =
detector artifact on flat profiles; set-level membership AUC ≈ popularity
prior ≈ 0.49. Diagnosis: no scarcity — wd 0.01 makes redundancy free, so
induction smears over heads (cf. Singh et al. many-to-many redundancy).
Mod-add lesson: committee crispness is wd economics (the K* capacity law).

**wd sweep (pre-registered before training):** wd ∈ {0.01, 0.1, 0.3, 1.0},
same init seeds each cell, 6 init × 2 data seeds per cell.
- P-W1: ablation top-1 concentration rises monotonically with wd; k_eff
  (participation ratio of ablation deltas) falls.
- P-W2: where k_eff reaches ~1-2, winner-level twin agreement rises above
  v2's 0.458 (crisper variable => more init-determined).
- P-W3 (cross-system law): k_eff(wd) falls with wd as mod-add K* does —
  same membership-tax economics in an unrelated system.
Failure semantics: if high wd kills induction (CE stays at unigram floor)
before sharpening it, report the window honestly; if k_eff stays ~flat
across a 100× wd range, the redundancy is not economic and the crisp-
identity premise fails here → fall back to the mod-add compiler paper.

## Uniqueness check (2026-08-13)

Closest work, all verified NOT to cover selection-from-init:
- **rLLC head specialization** (Timaeus, arXiv:2410.02984): 2-layer attn-only,
  documents head differentiation during training; seed appendix checks pattern
  generality only, NOT head identity. Predicting roles from step-0 weights,
  init interventions, and "why this head rather than another" are all absent —
  the paper "emphasizes developmental dynamics over developmental determinism."
  Same relation to us as Li₂ had in mod-add: they map formation, we add selection.
- **Induction-head theory** (Bietti et al. 2306.00802; provable dynamics
  2409.10559; emergence 2511.01033): minimal architectures, typically one head
  per layer — the identity lottery is erased by construction.
- **Singh et al.** (what needs to go right / transient ICL): finds multiple
  redundant induction heads, many-to-many prev-token wiring — background for
  PQ0 (committee-like outcome plausible), no init prediction.
- **Seed-induced subspace uniqueness** (2511.01023): seeds → non-transferable
  attention subspaces (subliminal-transfer context); supporting evidence that
  init geometry is seed-individual, no circuit-identity claims.
- **Frozen-QK result** (2506.01115): induction circuits form even with frozen
  random attention weights — indirect evidence that init QK/OV geometry
  constrains which circuit assembles (the readout should partly work).

Nobody found who (1) quantifies winner-identity variability across seeds,
(2) predicts the winner from init weights, or (3) dictates it by init editing.
All three lanes open.
