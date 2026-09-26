# PLAN — empirical programme for the paper

Created 2026-09-25 16:31 IST. The paper deadline is 2026-09-26 17:30 IST. The rules in NORMS.md are
binding.

## Notation

| Symbol | Meaning |
|---|---|
| p | modulus (odd prime) |
| n | number of operands |
| ν = n+1 | homogeneity degree |
| q | penalty exponent (the paper's b); 2 ≤ q ≤ ν |
| s = ν/q | the exponent that the theory says decides everything |
| Γ_s(p), y\* | the share programme and its maximiser |
| K\* | the number of frequencies y\* uses |
| B_p = 2(2pν)^{−ν/2} | normalisation constant |

- **Class optimum** (product networks, interference-free class): γ_IF = ν^{−(ν−q)/q} · B_p · Γ_s(p).
- **Universal ceiling** (all networks): B_p · Γ_s(p) for product networks; n!·B_p·Γ_s(p) for power
  networks.

The primary claim is that **the ansatz is valid**: trained networks allocate amplitude across
frequencies as the share programme predicts, and s = ν/q alone decides the outcome.

## Experiments

| id | claim | model | what varies | key predictions |
|---|---|---|---|---|
| E0 | core library, validation suite, frozen training recipes | — | — | tests pass; constructed networks hit the exact optimum |
| E1 | ansatz: count and shares follow the programme; s alone decides | product | ν ∈ {3,4,5} × s-grid (q = ν/s) × several p × seeds | K = K\*; amplitude shares ∝ y\* (modulo relabelling k → uk); γ ≤ ceilings |
| E1b | equivalent s: different (ν, q) with the same s behave the same | product | subset of E1 with s shared across ν | within-s spread ≪ between-s spread |
| E3 | q = ν is dense, every time | product (ν = 3,4,5) + power (ν = 3 attained; ν = 4,5 labelled "bound maximiser") | ν × p × seeds | all (p−1)/2 frequencies used, with equal amplitudes |
| E4 | ablation bound (CONSEQUENCES §3): deleting frequency j costs ≥ 1−(1−w_j)^s of margin | product | reuses E1 networks + constructed controls | loss vs budget share lies above the curve (a theorem for the clean allocation; trained networks are a measurement) |
| E5 | bottlenecks (§7): K offsets tie at the worst margin, and their equations fix the amplitude ratios | product | reuses E1 networks + constructed controls | number of near-tied offsets ≈ K; predicted ratios match |
| E6 | dilation equivariance (§9): relabelling the init by x → ux relabels the learned frequencies; marginals uniform | product | paired runs; many seeds at one cell | exact frequency map; P(j active) = E[K]/h |
| E7 | plateaus (§1): amplitude ratios are piecewise constant in q, jumping at predicted switch points | product | fine q-grid at fixed p, ν | ratios match the predicted vertex per q; jumps at the predicted s |
| E8 | log-p growth (§0): K\*(p) = Θ(log p) | product | exact K\*(p) over primes; trained K across p at fixed s | trained K vs K\*(p); K\*/log p bounded (exact range is small p only; stated honestly) |
| AUDIT | independent audit of all of the above | — | — | prereg compliance, reruns of a sample, check every number |

p ranges (amended 16:55 IST): p is NOT restricted to small values. K* grows only like log p, so the
q/ν sweeps need large p to have room to move.
- ν = 3: p up to about 113.
- ν = 4: p up to about 59.
- ν = 5: p up to about 31–43, whatever fits the 2 h per-run cap.
- Predictions are EXACT where the programme can be solved exactly. Beyond that they are best-found
  solutions from a validated solver (cross-checked against the exact solver wherever the two
  overlap), and are labelled as such.

Excluded by decision: the identity/certificate comparison (CONSEQUENCES §5–6), and any
ReLU/transformer/Adam work.

## Directory layout

```
heracles-emp/
  NORMS.md  PLAN.md  README.md  LOG.md  pyproject.toml
  core/            library (programme, models, train, measure, metrics, construct, runner) + THEORY_SPEC.md + RECIPES.md
  tests/           validation suite (must pass before any evaluation)
  experiments/E*/  PREREG.md, predictions.json, configs/, runs/, RUNS.csv, analysis/, REPORT.md, LOG.md, reproduce.sh
```
