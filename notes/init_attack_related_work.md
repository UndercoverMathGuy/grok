# Init-edit attack & init-determinism — related-work audit (2026-08-14)

Condensed from a 3-agent uniqueness sweep (read-side, write-side, 2025-26
recency; ~45 searches, ~18 abstracts fetched). Originally run for the
(since-vetoed) dominoes/shortcut direction, but the write-side and
supply-chain findings are the related-work spine for the stealth arm and
the attack-surface framing of the compiler/dial paper.

## Mandatory carves (cite and distinguish, or reviewers will)

- **Grosse/Trost, "On the Security Relevance of Initial Weights"**
  (1902.03020): stat-preserving init permutation that CRIPPLES training
  (dead neurons, accuracy denial). Owns "invisible init edit with a
  training-outcome effect". Carve: denial vs SELECTION — ours picks which
  valid circuit wins, dose-controlled, without hurting the task.
- **FAB, "Finetuning-Activated Backdoors in LLMs"** (2505.16567, ICLR
  2026): meta-learned checkpoint, benign until victim fine-tunes, then a
  planted BEHAVIOR emerges. Owns the "weights predetermine fine-tuning"
  threat frame. Carve: behavior-implant via heavy optimization vs
  behavior-free, statistically invisible edit that biases which legitimate
  solution SGD selects. Also cite Grond 2501.05928 (parameter-space
  stealth for backdoor injection) on the invisibility axis.
- **Goldwasser et al., "Planting Undetectable Backdoors"** (2204.06974,
  FOCS): cryptographic undetectability vocabulary for the threat model;
  still an input-triggered behavior, not a learning disposition.
- **He et al. 2602.16849** (Fourier features / lottery / grokking): init
  magnitude+alignment predicts winning frequencies, mod-add, observational.
  Already carved in the 2026-08 uniqueness audit: they predict, we dictate;
  re-verify no companion paper has added the causal intervention before
  submission.
- **Butterfly Effect** (Kwok et al., ICML 2025, 2506.13234): tiny init
  perturbations divert training to different basins — untargeted, no
  outcome variable. Motivation citation; our novelty is targeted,
  outcome-chosen flips.
- **Omnigrok** (2210.01117): init weight NORM toggles memorize/generalize
  regime — a visible macroscopic knob on timing, not solution selection.

## Read-side landscape (init-determinism of feature/circuit choice)

- Hermann, Mobahi, Fel, Mozer (ICLR 2024, 2310.16228) + Hermann & Lampinen
  (2020): predictivity x availability governs feature use; decodability at
  init; population-level only, no per-seed prediction, no editing.
- Lim, Kim, Moon (NeurIPS 2025, 2602.03066): shortcut features are top NTK
  eigenfunctions — formalizes availability as init kernel alignment.
  Population-level mechanism; our per-seed readout would be "their metric,
  deployed per-seed" — a reason the read claim alone is weak in real-data
  settings.
- Per-seed strategy variance is established prior art: McCoy et al.
  "BERTs of a feather" (1911.02969, 100 seeds, HANS 0-66%), Juneja et al.
  linear-connectivity basins (2205.12411), D'Amour underspecification
  (2011.03395). A lottery OBSERVATION is not novel anywhere; prediction
  from init and dictation are the open lanes.
- Lin (2510.00468) muses eNTK-at-init could predict feature choice
  (mod-add Fourier features) without executing it — cite and differentiate.

## Verdict (as of 2026-08-14)

All three surfaces OPEN for "statistically-invisible, behavior-free init
edit that causally selects which competing solution training adopts, with
dose-response". The vetoed dominoes direction would have needed these
carves; the mod-add/dial stealth arm needs them too, plus an explicitly
pre-committed detector family for the invisibility claim (per-tensor KS on
weight marginals, norm/moment checks, chi-square envelope) so "invisible"
is operationalized, not asserted. Residual scoop risks to re-check at
submission: Lim et al. camera-ready follow-ups; any new paper adding a
causal init intervention to the He et al. 2602.16849 line.