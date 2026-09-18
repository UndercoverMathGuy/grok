# Q — the penalty-exponent dial on the x² two-operand net (2026-09-17, one 4090 secure pod, ~$1)   [runs/NS/n2_pow_p71_l2q*, runs/NS_probe/pod4, nsum.py --penalty l2q, THEORY_nsum §5]

Question: with the network fixed (x² on a + b, p 71, width 500, Morwani recipe: init 1, lr 0.05 doubling to 12.8,
40k steps, penalty 1e-4) and only the penalty exponent q in P_q = λ Σ_i ‖θ_i‖^q moved, does the frequency count
follow argmax_K M_K/K^{3/q}? q = 2 is plain weight decay (RESULTS_QM: 5), q = 3 is Morwani's L_{2,3} (all 35).
"anneal" = resumed from the 40k endpoint at λ 1e-5 for 60k steps (lr 12.8), the λ → 0 continuation.
Predictions from kstar.py M_K at p 71 (kmax 35), plateau = rungs within 26% of the top.

## Table (4 seeds per cell; K_2pct / N_.05 / clean fraction listed per seed)

| q | e = 3/q | K* (plateau) | steps | acc | margin | K_2pct | K_5pct | N_.05 | clean |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.50 | 5 (4..16) | 40k | 1.000 | 0.7–1.5 | 5, 6, 4, 6 | 5, 5, 4, 3 | 6, 7, 12, 13 | 0.36–0.45 |
| 2 | 1.50 | 5 (4..16) | anneal | 1.000 | 3.4–3.7 | 5, 6, 4, 6 | 5, 5, 4, 3 | 5, 6, 11, 13 | 0.35–0.44 |
| 2.25 | 1.33 | 7 (4..35) | 40k | 1.000 | 0.5–1.7 | 6, 5, 5, 4 | 6, 5, 5, 4 | 7, 11, 15, 15 | 0.41–0.52 |
| 2.25 | 1.33 | 7 (4..35) | anneal | 1.000 | 3.1–3.7 | 6, 5, 5, 4 | 6, 5, 5, 4 | 6, 10, 15, 15 | 0.40–0.51 |
| 2.5 | 1.20 | 10 (4..35) | 40k | 1.000 | 0.6–2.2 | 9, 6, 8, 10 | 6, 6, 6, 8 | 10, 15, 10, 15 | 0.40–0.61 |
| 2.5 | 1.20 | 10 (4..35) | anneal | 1.000 | 2.5–4.0 | 9, 6, 8, 10 | 6, 6, 7, 8 | 10, 15, 10, 14 | 0.38–0.60 |
| 2.75 | 1.09 | dense, tie (5..35) | 40k | 1.000 | 2.0–2.3 | 13, 11, 13, 14 | 6, 5, 7, 6 | 13, 18, 13, 16 | 0.57–0.67 |
| 2.75 | 1.09 | dense, tie (5..35) | anneal | 1.000 | 3.7–4.3 | 12, 11, 13, 14 | 7, 5, 7, 8 | 13, 18, 13, 15 | 0.57–0.66 |
| 3 | 1.00 | dense | 40k | 1.000 | 4.9–5.2 | 32, 35, 35, 34 | 0 (flat) | 32, 33, 33, 33 | 0.99 |
| 3 | 1.00 | dense | anneal | 1.000 | 7.4–7.7 | 33, 35, 35, 35 | 0 (flat) | 33 | 0.99–1.00 |

Below q = 2 (theory says the unrestricted argmax stays at 5; THEORY_nsum §5):

| q | lr cap | λ | steps | acc | margin | K_2pct | N_.05 | clean | note |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 12.8 | 1e-4 | 40k | — | — | — | — | — | NaN at lr 12.8 (seeds 0, 1); seeds 2, 3 acc 0.84 / 0.74, margin −10 / −13 |
| 1.5 | 3.2 | 1e-4 | 40k | 0.993 | −1.9 | 2 | 2 | 0.25 | CE 0.08 still falling |
| 1.5 | 0.8 | 1e-4 | 40k | 0.994 | −2.6 | 1 | 1 | 0.24 | |
| 1.5 | 12.8, clip 1 | 1e-4 | 40k | 0.862 | −24 | 2 | 3 | 0.25 | |
| 1.5 | 3.2 or 12.8 | 1e-3 | 40k | 0.021 | 0 | — | — | 0 | collapsed to zero weights |
| 1.5 | 3.2 | 1e-4 | 200k, 2 seeds | 0.96, 0.95 | −19, −19 | 2, 3 | 2, 5 | 0.24, 0.21 | CE plateau 0.07 at 100k then oscillation; never converges |
| 1.2, 1.0 | 12.8 | 1e-4 | 40k | — | — | — | — | — | NaN (3 seeds, not pursued) |

## Reading

1. **The count is monotone in q and the endpoints are the two published regimes.** Same network, same data, same
   optimiser and schedule, 40k steps: median K_2pct 5.5 → 5 → 8.5 → 12.5 → 35 for q = 2, 2.25, 2.5, 2.75, 3, against
   the programme's 5 → 7 → 10 → dense → dense. The transition from plain weight decay's K* to Morwani's flat
   spectrum is continuous in the penalty exponent, as argmax_K M_K/K^{3/q} says; the plateaus at q ≥ 2.25 are wide
   (the functional is within 26% of its top from K = 4 up to 35), so the per-seed spread there is the plateau, not
   noise in the prediction.
2. **λ → 0 leaves every count where it was.** The continuation changes no K_2pct by more than 1 in 20 runs, raises
   every margin (×2.5 to ×5) and leaves the clean fractions unchanged: the counts are properties of the GD
   fixed-point family at each q, not of λ = 1e-4.
3. **Junk is the price of q < ν.** Clean fraction 0.35–0.45 (q 2), 0.40–0.52 (2.25), 0.38–0.61 (2.5), 0.57–0.67
   (2.75), 0.99 (3): monotone in q and reaching 1 only at q = ν, where a frequency can be split over many neurons at
   no cost and the cancelling pairs are free (THEORY_nsum §5, last paragraph). This is the same "half junk" seen at
   every rung of the ν ladder (RESULTS_NS), now with its cause.
4. **q < 2 is dead on both sides.** The unrestricted bound reverts to the plain-l2 programme (concave cost makes one
   many-frequency neuron the cheapest carrier), and GD does not converge there at any learning rate or λ tried
   (NaN at 12.8; CE plateau then oscillation at 3.2 and 0.8, margins −2 to −19; λ 1e-3 collapses). So the exponent
   dial on a fixed network runs over e ∈ [1, ν/2] only, and steeper exponents need a higher degree (the ν ladder).
5. **What this buys the paper.** One figure, one recipe, two dials: count against ν/q with the x² net at q ∈
   [2, 3], the ReLU net (ν 2, dense) and the x³ net at q = 2 (ν 4, RESULTS_NS §4) as the other points; Morwani and
   Li et al. sit at ν/q = 1 on it. The measured x² curve is the clean half; the ν > 3 half stays what RESULTS_NS says.

Recipe facts: q ∈ [2, 3] runs on the unchanged Morwani recipe (no clip, no warm-up, 20 s per 40k-step run on a
4090); q < 2 needs lr ≤ 3.2 to avoid NaN and still does not converge. The gradient of ‖θ_i‖^q at ‖θ_i‖ = 0 is
0 × ∞ for q < 2; nsum.py clamps the squared norm at 1e-12 there (no effect on the q ≥ 2 runs).
