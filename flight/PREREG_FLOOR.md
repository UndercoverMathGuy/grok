# Replica noise floor — adjudicating "under-modelled" vs "physical ceiling"

Registered 2026-08-20, before any run exists. Question: the flight-time
law's out-of-sample residual is ×1.54 median (R²(log) 0.73, train = test).
Is that residual (a) a missing observable (under-modelling) or (b) a
noise/chaos floor no forecaster of early observables can beat?

## Design

8 identical-config groups: cells (f0.25,wd1), (f0.3,wd1), (f0.3,wd3),
(f0.4,wd1) × 2 fresh seeds (families `floornoise_*`, META_SEED 20260820).
Per group:
- 4 **identical replicas** (same data_seed, init_seed; separate concurrent
  training launches so kernel scheduling can differ) → hardware /
  nondeterminism floor ρ_id.
- 4 **micro-perturbed replicas** (init + 1e-6 × per-tensor-std Gaussian,
  perturb seeds 7000+j) → chaos floor at forecast-relevant scale ρ_pert.

Statistic: within-group max/min t_gen ratio (sustained test ≥ 0.9,
30k budget); ρ = median over the 8 groups per arm. Reference: the law's
per-seed residual ×1.54.

## Registered interpretation

- **ρ_pert ≥ 1.4** → the ceiling is physical: infinitesimally different
  inits diverge in t_gen by about the law's residual, so no early-window
  forecaster (of any model class) can beat ×~1.5. "Under-modelled" is
  refuted at the forecast-relevant scale; the predictability ceiling is a
  finding.
- **ρ_pert ≤ 1.1** → headroom is real: t_gen is stable to tiny init
  changes, so the ×1.5 residual must live in observables we have not
  used. The observable hunt (junk composition, margin trajectory, Adam
  state) is justified.
- **1.1 < ρ_pert < 1.4** → partial: report the split between floor and
  headroom quantitatively (residual² ≈ floor² + headroom² in log space).
- ρ_id ≈ 1.0 exactly (bitwise-identical replicas) means training is
  deterministic on this stack; the verdict then rests entirely on ρ_pert.
- Censored runs (no gen by 30k) are reported; a group with censored
  members contributes its ratio bound and is flagged, not dropped.
