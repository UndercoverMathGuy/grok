# T2-D v2 — hybrid forecaster, fresh seeds

Registered 2026-08-20, after v1 scored 4/15 (P-F1 FAIL) and before any v2
run exists. Fresh seeds (families `t2dv2_*`, META_SEED 20260820, crc32).
Same cells, budget (30k), forecast epoch (500), warm-resume protocol,
r\* = 10^-2.25, floor = 132, gate = e·r\*, ×1.5 band, and scoring rules as
v1 (`flight/PREREG_T2.md`). Only the forecaster changes.

## Diagnosis driving the change (from v1's own runs)

1. **Committee misdetection dominated**: 14/15 largest-gap committees at
   epoch 500 under-counted K (the largest energy gap sits inside the
   still-separating committee), so junk share was computed with committee
   members counted as junk.
2. **λ non-stationarity**: the [t_mem, 500] slope underestimates the
   flight-average rate on slow cells (the known cleanup acceleration);
   window extension alone does not converge (checked to 2000 ep).

## The v2 recipe (all constants frozen from prior data)

- **Slow cells** {(f0.2, wd3), (f0.25, wd1), (f0.3, wd1)}:
  committee = top-4 energies at ep 500 (K≈4 is the task-intrinsic
  capacity result); λ = slope on [t_mem, 500] × γ(cell), with γ the
  median λ_full/λ_500 from degrok + degrok2 add113 ONLY:
  γ(0.2,3) = 0.65, γ(0.25,1) = 2.23, γ(0.3,1) = 1.44.
- **Fast cells** {(f0.3, wd3), (f0.4, wd1)}: the v1 recipe verbatim
  (largest-gap committee, raw λ) — it was already calibrated there
  (3/3 at ×1.01–1.10 on f0.3 wd3).
- Floor gate unchanged, evaluated with the respective committee.
- **Shadow ablation**: the v1 recipe is also computed on the same
  segment-1 data and committed alongside, so v1-vs-v2 is a same-cohort
  comparison, not a cross-cohort one.

Retrodictive score of this exact recipe on the v1 cohort: 10/15
(reported for context; the bars below bind on the FRESH cohort only).

## Bars

- **P-F1v2 (primary):** ≥ 9/15 fresh forecasts within ×1.5 (CENSORED
  scored as in v1). For reference, v1's registered bar was 11/15 and it
  scored 4/15.
- **P-F2v2:** Spearman ≥ 0.8 over numeric pairs.
- **P-F5 (ablation):** v2 hits > v1-shadow hits on the same fresh runs.
- **Known risk, disclosed:** f0.2/wd3 is erratic (historical γ < 1, one
  v1 run missed ×5 both-directions); the cell stays in as hard mode.
