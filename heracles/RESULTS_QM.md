# QM — Morwani's quadratic recipe with the penalty as the only dial (2026-09-16, pod 8xjo75gioo7xey 4090, 40 runs, ~$0.60)   [runs/QM, quad_morwani.py]

Motivation: the unrestricted sparse-program theorem (notes/l2_theorem.md §7) is for the quadratic one-hidden-layer
net whose degree comes from the activation (U, V, W each Frobenius-priced). Our earlier quadratic runs (P-N14, twin
distillation) used a train/test split and memorised. Morwani et al. (ICLR 2024, App. B.1) trained the SAME quadratic
architecture on the FULL dataset — no generalisation asked — under L_{2,3} and found all frequencies used (their
Fig 7b). Their recipe, copied verbatim, with plain l2 swapped in, is the cleanest test of the ansatz on the one
architecture where the theorem is unrestricted.

Recipe (theirs): p = 71, width m = 500, x^2 activation, no biases, input one-hot(a) (+) one-hot(b), full dataset
(all p^2 pairs), full-batch GD, 40 000 steps, lr 0.05 doubled at steps 1e3..10e3 (their final 51.2), penalty 1e-4.
Ours: identical, plus (i) a cap on the doubling schedule (plain l2 diverges at 51.2; both caps 12.8 and 3.2 run),
(ii) init scale x4 at p = 113 (at default init plain l2 never leaves the zero solution: the l2 pull is linear in the
weights while the quadratic net's loss gradient near zero is quadratic; L_{2,3}'s pull vanishes faster and escapes).
Penalties: l2 = lam sum(theta^2); l23 = lam sum_i (||u_i||^2 + ||v_i||^2 + ||w_i||^2)^{3/2}, lam = 1e-4 for both.
Count instrument here = number of frequencies carrying >= 2% (and >= 5%) of the clean-logit Fourier energy, plus the
Morwani-style histogram (frequencies holding >= 1% of neuron mass). NOT the validated interface-ablation count.

| p | penalty | lr cap | init | acc | K at 2% (4 seeds) | K at 5% | #freqs with neurons | K* (plateau >= 0.95) |
|---|---|---|---|---|---|---|---|---|
| 71 | l2 | 12.8 | 1 | 1.000 | 5, 6, 4, 6 | 5, 5, 3, 2 | 13, 12, 17, 18 | 5 (5..7) |
| 71 | l2 | 3.2 | 1 | 1.000 | 5, 5, 4, 6 | 5, 5, 3, 2 | 12, 11, 15, 17 | 5 (5..7) |
| 71 | l23 | 12.8 | 1 | 1.000 | 32, 35, 35, 34 | 0, 0, 0, 0 | 35, 35, 35, 35 | (theirs: all 35) |
| 71 | l23 | 3.2 | 1 | 1.000 | 27, 28, 27, 27 | 0, 0, 0, 0 | 32, 35, 35, 35 | |
| 113 | l2 | 12.8 | 1 | 0.098 | collapsed to zero weights (4/4) | | | 6 (5..8) |
| 113 | l2 | 12.8 | 4 | 1.000 | 9, 7, 8, 8 | 8, 6, 7, 6 | 24, 26, 24, 29 | 6 (5..8) |
| 113 | l2 | 3.2 | 4 | 1.000 | 9, 7, 8, 8 | 8, 6, 7, 6 | 25, 26, 28, 30 | 6 (5..8) |
| 113 | l23 | 12.8 | 1 | 1.000 | 25, 26, 26, 29 | 0, 0, 1, 0 | 46, 45, 48, 45 | (theirs: all 56) |
| 113 | l23 | 3.2 | 1 | 1.000 | 17, 17, 19, 18 | 7, 6, 8, 9 | 42, 41, 40, 42 | under-trained (margin 0.6-1.4 vs 2.8-3.2) |

Reading.
1. Same architecture, same data, same optimiser and schedule: plain l2 gives 4-6 frequencies at p 71 (K* = 5) and
   7-9 at p 113 (K* = 6, plateau 5-8, rung 9 at 0.93), L_{2,3} gives 27-35 of 35 and 25-29 of 56 with every
   frequency populated by neurons. The penalty alone moves the count from the ladder's K* to "all of them". This is
   the Morwani L_{2,nu}-vs-l2 dial the reviewers asked for, on the architecture where the sparse program is a theorem.
2. Every run, both penalties: single-frequency neurons (0.95-1.00 of each neuron's energy in its top frequency).
3. p 113 l2 committees are not equal-share: seed 1 amplitudes 1.00, 0.95, 0.59, 0.58, 0.58, 0.55, 0.34, 0.19, ... —
   two large, four medium, a tail. The 5% count (6-8) is the better read of the committee size; the 2% count picks up
   the tail. Interface-ablation count (nudial/ablation_count.py) still to be run on these checkpoints.
4. The l2 count does not depend on the lr cap (identical seeds at 12.8 and 3.2). p 71 l2 seeds 2-3 have low margin
   and only 2-3 frequencies at 5% at 40k steps: under-converged, longer training needed before quoting per-seed numbers.
5. Escape from zero is the one recipe difference between the penalties and it is understood (linear vs superlinear
   pull near zero). Report it as such; it is not a count effect.

Status for the paper: quadratic-on-full-data is a live demo of the ansatz on the theorem's own architecture, with
Morwani's norm as the control. Needed before quoting: 8 seeds, ablation count, longer training for the low-margin
seeds, and the normalised-margin ceiling comparison (c0 3^{-3/2} Gamma*(p) in the note's normalisation).
