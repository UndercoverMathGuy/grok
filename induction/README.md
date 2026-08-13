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
