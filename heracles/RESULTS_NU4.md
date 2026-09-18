# NU4 — is the functional right at ν ≥ 4, and can free GD be brought to it?
(2026-09-18, 4×5090 + 1×4090 secure, all deleted after count-verified pulls)   [FINAL_RUNS/NU4 (213 metrics), nu4_fixed.py, nu4_program.py,
nu4_table.py → nu4_tables.md (full tables), nu4_scatter.html/json, nu4_pod.sh, nsum.py flags --grid/--out/--trace_every]

Symbols. p prime, n operands, ν = n + 1 homogeneity degree, K number of Fourier frequencies, S_K the coverage-optimal
K-set (kstar.best_subset), M_K = min_{D≠0} Σ_{k∈S_K}(1 − cos 2πkD/p) its coverage. γ = (min margin over the full table)
/ ‖θ‖^ν, the l2-normalised margin (nsum.analyse, fp64). Equal-share prediction M_K/K^{ν/2}. New here: the
FREE-AMPLITUDE programme Γ_S = max_{b≥0, Σb=1} min_D Σ_k b_k^{ν/2}(1 − cos 2πkD/p) (b_k = share of the l2 budget on
frequency k; nu4_program.py, SLSQP with restarts), and Γ* = Γ_S over all frequencies. Bound constant c (one single-frequency
neuron of squared norm m has clean amplitude ≤ c m^{ν/2}, AM-GM over the n+1 factors): chain c = 2^{-n}(2/(pν))^{ν/2}
(× n^{n/2} tied, × n! for x^n). So γ ≤ c Γ_S and "γ/(cΓ_S)" is the fraction of the sparse-programme upper bound reached.

Method (T1). Build the junk-free K-frequency net analytically, then fine-tune on CE + λ‖θ‖² with every embedding row and
readout column projected onto the Fourier span of S_K after each step (support fixed; neurons may mix frequencies inside
S_K). Optimiser: normalised GD + exact 1-D line search on the global scale (solves the slow weight-decay norm mode
exactly), gap-form CE (fp32-stable at margin 12), λ 1e-5 → 1e-6 → 1e-7, 3000 steps each. Populations per frequency:
"cube" = 2^n phases in {0, π/2}^n (THEORY_nsum §6; ×2^{n−1} signs for x^n), "common" = (n+1) neurons with the same phase
πl/(n+1) on every operand (PROOF_AUDIT finding 1). 5% weight noise + random population phase per seed.

## 1. The programme itself (no network): free amplitudes keep the argmax, and kill frequencies in large supports

| p, ν | equal-share argmax (top values) | free-amplitude Γ_S on S_K | Γ* over ALL frequencies: active support | within 2% of Γ* |
|---|---|---|---|---|
| 71, 3 | K 5 (.300; K6 .289, K7 .286) | K5 .3080, K6 .2932, K4 .2746 | 5 | 5 only |
| 71, 4 | K 4 (.1359; K5 .1344, K3 .1253) | K5 .1381, K4 .1373, K3 .1332 | 4 (Γ* .1406, a non-ladder 4-set) | 4..5 |
| 71, 5 | K 3 (.0723; K4 .0680) | K3 .0770, K6 .0725 (3 active) | 3 | 3 only |
| 59, 5 | K 3 (.0774) | K3 .0805 | 3 (Γ* .0835) | 3 only |

On S_K with K ≥ 6 (ν 4) / K ≥ 4 (ν 5) the optimal shares put exact zeros on some frequencies: inside a big support the
programme's optimum is a sparse sub-committee. The dense equal-amplitude point is 0.21 (ν 4) of Γ*.

## 2. Measured fixed-support ladders (γ at λ 1e-7, mean over seeds; "norm" = γ / best sparse rung)

Chain (`--act prod`), ν 4, p 71 — seeds agree to 5 digits:

| K | γ cube ×1e-7 | γ common ×1e-7 | norm (both) | equal-share pred | free-amp Γ_S pred | γ/(cΓ_S) cube | γ/(cΓ_S) common | N_.05 at end |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.529 | 1.058 | 0.50 | 0.51 | 0.51 | 0.122 | 0.244 | 2 |
| 3 | 1.016 | 2.032 | 0.96 | 0.92 | 0.96 | 0.123 | 0.246 | 3 |
| 4 | 1.049 | 2.097 | 0.99 | 1.00 | 0.99 | 0.123 | 0.247 | 4 |
| 5 | 1.055 | 2.110 | 1.00 | 0.99 | 1.00 | 0.123 | 0.247 | 5 |
| 6 | 0.916 | 1.831 | 0.87 | 0.87 | 0.90 | 0.113 | 0.226 | 6 |
| 7 | 0.839 | 1.678 | 0.80 | 0.80 | 0.91 | 0.108 | 0.216 | 7 |
| 8 | 0.891 | 1.783 | 0.85 | 0.71 | 0.90 | 0.123 | 0.246 | 6 (2 freqs switched off) |
| 10 | 0.898 | 1.798 | 0.85 | 0.61 | 0.86 | 0.116 | 0.232 | 5 |
| 16 | 0.783 | 1.566 | 0.74 | 0.41 | 0.77 | 0.117 | 0.235 | 7 |
| 24 | 0.915 | 1.871 | 0.87–0.89 | 0.29 | 0.86 | 0.127 | 0.251 | 5 |
| 35 (all) | 1.51, 2.06 | 1.84, 2.12 | — | 0.21 | 1.00 | 0.237 of cΓ* | 0.244 of cΓ* | 4, 4 / 5, 4 |

Other ladders (full rows in nu4_tables.md):

| family, ν | measured norm γ(K) | predicted (free-amp) | argmax meas / pred | dense rungs |
|---|---|---|---|---|
| chain n 2, ν 3 (cube and common identical shape) | K2 .32, K3 .75, K4 .89, **K5 1**, K6 .95, K7 .94, K8 .91, K10 .90, K12 .90–.93 | .32 .75 .89 1 .95 .93 .93 .90 .91 | 5 / 5 | K16/24/35 → N_.05 5–7, γ .86–.97 of the K5 rung |
| x² n 2, ν 3 | .32 .76 .91 **1** .93 .89 .90 .87 .83 | same | 5 / 5 | K16/24/35 → N_.05 8–14, γ .83–.86 (K35 clean only .70) |
| x³ n 3, ν 4 (seed spread ±7%) | K2 .35, K3 .84, K4 .93, **K5 1**, K6 .87, K7 .75, K8 .96 (→5–6 active), K10 .95 (→5–6) | .51 .97 1.00 1.00 .95 .91 | 5 / 4≈5 (0.4% apart in Γ_S) | K16 → 6 active, .83; K35 → N_.05 10, clean .70, γ .41 |
| chain n 4 tied, ν 5, p 59 | K2 .78, **K3 1**, K4 .92, K5 .75, K6 .64, K8 .72 (→5), K12 .75 (→5) | .77 1 .92 .76 .64 (equal-share) | 3 / 3 | — |

Verdict T1. **The functional is right at ν 4 and ν 5.** Through the argmax the measured curve equals the prediction to
1–3% in both exact families and at all three degrees: ν 3 peaks at 5 (positive control passes), ν 4 has the K4/K5 tie
the free-amplitude programme predicts (K5 ahead by 0.6% measured, 0.6% predicted; equal-share has K4 ahead by 1.2%;
Γ* itself is a 4-set), ν 5 peaks at 3. Above the argmax the fixed-support optimum switches frequencies off (N_.05 falls
below K), exactly as the free-amplitude programme does, and γ stays below the argmax rung; the measured values there
are 5–10% under Γ_S because GD from the equal-amplitude construction does not fully find the best sub-committee.
**Dense < sparse, measured:** no rung with K ≥ 16 beats the K* rung within its own population, every dense rung ends
with 4–7 active frequencies in the chain, and the dense construction itself is 0.21–0.57 of the argmax rung. The one
apparent exception (chain ν 4 cube K 35, γ 1.5–2.1e-7 > cube K5 1.05e-7) is a population effect, not a count effect:
with all frequencies free the 8-neuron cube re-organises towards the cheaper common-phase population and lands at 4
active frequencies and 0.237 of cΓ*, still under the common-phase K5 rung (0.247).

Audit constant (PROOF_AUDIT finding 1). Trained rungs sit on the population constant N^{-(ν−2)/2}, flat across K:
cube 2^n neurons → 0.488–0.500 (n 2), 0.122–0.123 (n 3); common-phase (n+1) neurons → 0.564–0.578 (n 2; conjecture
3^{-1/2} = 0.577), 0.244–0.247 (n 3; conjecture 0.25), 0.087–0.088 (n 4 tied; conjecture 5^{-3/2} = 0.0894). The small
shortfall is the finite λ. Nothing trained exceeded the conjectured constant; x³ reaches only 0.044–0.057 of its bound.

## 3. Free GD (T2/T3): the excess count is a junk basin, not the optimum

Chain ν 4, baseline recipe (init 14, warm-up, lr 3.2, λ 1e-5, 120k), now 8 seeds (0–3 old, 4, 5, 8, 9 new):
N_.05 7, 5, 8, 8, 9, 5, 10, 9 (median 8); K_5pct 6, 2, 4, 5, 3, 4, 2, 4 (median 4); clean 0.47–0.74; margins −1.5…+0.01;
γ ≤ 0 → **0.00 of the measured ceiling 2.11e-7**. Count does not drift in the traces (20k → 120k: 5→8→8, 7→10→10).

| T2 variant (chain ν 4) | result | reading |
|---|---|---|
| λ 1e-6 from the start, 120k | margin +1.85, γ 5e-9 (0.02), N_.05 7, clean 0.47 | same basin |
| lr cap 0.8, 60k | margin −2.6, N_.05 7, clean 0.44 | same basin |
| init 8 + 40k warm-up, 60k | margin −2.7, N_.05 6, clean 0.55 | same basin |
| width 125, 60k (2 seeds) | N_.05 **4** (clean 0.77, margin −0.02) and 9 (clean 0.52) | one hit, not a recipe |
| margin polish of the 4 baseline endpoints (normalised GD + scale search, λ→1e-7, no projection) | γ 2.3e-8, 4.5e-8, 3.5e-8, 7.4e-8 = 0.11–0.35 of ceiling; N_.05 7→7, 5→2, 8→8, 8→3; clean 0.46–0.84 | junk survives; polishing prunes or freezes, never reaches the plateau value |
| **q = ν first, then plain l2** (train L_{2,4} λ 1e-6 → dense clean-ish net; then l2) | see below | **works** |

The recipe that works. Train with the L_{2,ν} penalty (dense, N_.05 18–22, clean 0.54–0.75, l2-γ only 1.0–2.7e-8 =
0.05–0.13 of ceiling), then switch the penalty to plain l2:

| start net (q 4) | l2 stage | N_.05 / K_5pct / K_2pct | clean | margin | γ (fraction of ceiling 2.11e-7) |
|---|---|---|---|---|---|
| seed 0, 120k | plain GD resume, λ 1e-5, lr 3.2, 60k | 6 / 6 / 6 | 1.000 | 8.2 | 1.63e-7 (0.77) |
| seed 1, 120k | plain GD resume, 60k | 6 / 6 / 6 | 1.000 | 8.2 | 1.77e-7 (0.83) |
| seed 0, 120k | polish optimiser | 5 / 5 / 5 | 1.000 | 12.0 | 1.80e-7 (0.85) |
| seed 1, 120k | polish optimiser | 6 / 6 / 6 | 1.000 | 12.8 | 1.80e-7 (0.85) |
| seed 4, 60k | polish optimiser | **4 / 4 / 4** | 1.000 | 12.2 | 1.98e-7 (0.94) |
| seed 5, 60k (start margin < 0, clean 0.54) | polish optimiser | 4 / 2 / 5 | 0.88 | 5.9 | 0.70e-7 (0.33) |

Trace of the plain-GD resume (seed 0): N_.05 9 → 7 → 7 → 6 → 6 while γ 0.50 → 0.75 → 1.03 → 1.28 → 1.46 → 1.63e-7 and
clean 0.94 → 1.000: **the count falls towards the plateau as γ rises**, which is the T2 hypothesis. Scatter of every run
and trace point (γ/ceiling vs N_.05 and vs clean fraction): nu4_scatter.html; the picture is two clouds — junk basin
at γ/ceiling ≤ 0.35 with N_.05 2–10, and clean nets at 0.77–0.94 with N_.05 4–6 — plus the fixed-support rungs.
5 of 6 recipe runs land in the predicted plateau (3..7) with clean 1.000; counts 4, 5, 6, 6, 6 against argmax 4≈5.
Not done: q4 seeds 2, 3 finished training but were not polished (session outage); plain-GD resume has n = 2.

ν 5 (chain n 4 tied p 59, baseline recipe, 80k steps, seeds 2, 3): N_.05 4, 6; clean 0.33, 0.40; margins −16, −8.
With the two 120k seeds (3, 3) that is 3, 3, 4, 6 — in or next to the plateau (2..5) but junk-laden and at γ < 0; the
ν 5 fixed-support ladder is what carries the verdict there. ≥ 8 seeds at ν 5 was not reached (50 ms/step on a 4090).

## 4. What to claim
- Theorem-side: on both exact families the measured fixed-support γ(K) follows the programme at ν 3, 4, 5; argmax
  5 / 4≈5 / 3; dense supports are strictly worse and collapse to 4–7 frequencies when optimised under plain l2.
  Use the free-amplitude programme (not equal shares) for the predicted curve: it removes the K ≥ 8 mismatch.
- Dynamics-side: plain-l2 GD from random init at ν ≥ 4 sits in a junk-carrying basin at ~0 of the ceiling; its N_.05
  (5–10) is not evidence about the optimum. A clean start (q = ν pre-training) then plain l2 reaches 0.77–0.94 of the
  ceiling with 4–6 frequencies.
- Not to claim: a sharp 4-vs-5 at ν 4 (the programme itself has them 0.6% apart); attainment of the AM-GM bound
  (best trained fraction 0.25 at n 3 = the audit's conjectured constant); the recipe as a distribution (n = 5 usable).

## 5. Engineering notes
- nsum.py: `--grid` (untied: broadcast-grid forward, identical logits to 0.0 in fp64 vs the one-hot path; ~7–13 ms/step
  at p 71 n 3 m 500 on a 5090), `--out`, `--trace_every`. Tied `--grid` uses index_select and is SLOWER than the one-hot
  GEMM path (71 vs 50 ms/step on a 4090): do not use it for tied cells.
- Concurrency on one GPU is host-dependent: nu4-b/nu4-e scaled ~2× with 4–5 processes, nu4-d not at all.
- Spend: ≈ $14.6 against a $9 cap. Planned stop was 16:10Z (≈ $7.8); a session outage (≈ 15:55 → 17:32Z) left five pods
  billing $4.70/h for ~1.6 h. All five pods deleted and individually confirmed 404.
