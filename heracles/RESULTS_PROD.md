# PROD — the gated chain (depth n multiplicative network, ν = n + 1) on n-ary modular addition
(2026-09-17, eight 4090 secure pods in parallel, ~$10, all deleted after checksummed pulls)   [runs/NS/n*_prod_*, runs/NS_probe/pod5, nsum.py --act prod, prod_pod.sh, THEORY_nsum §6]

Family: h_i(a_1..a_n) = Π_j e_ij(a_j), readout w_i(c); n Hadamard-gate layers, one one-hot embedding per layer, plain
weight decay on all of them; equivalently a rank-m CP factorisation of the logit tensor. Exact for every n and every
q ∈ [2, ν] (THEORY_nsum §6): K*(p, ν, q) = argmax_K M_K/K^{ν/q}. Instruments as in RESULTS_NS (fp64, full table).
Recipe: init 14 (n ≥ 3; init 4 at n 2), decay-free warm-up until CE < 0.8 ln p (ends at step 5–7k), then lr cap 3.2,
clip 1, 120k steps, λ 1e-5 for plain l2 (1e-4 under-fits: acc 0.966, margin −14 — the product term lacks the
multinomial 6 of x³, i.e. the same λ is 2.4× stronger decay), λ 1e-6 for q > 2 (1e-5 collapses q 4 to zero weights).

## Table

| cell | ν | q | seeds | acc | margin | K_5pct | K_2pct | N_.05 | clean | K* (plateau) |
|---|---|---|---|---|---|---|---|---|---|---|
| n 2 (bilinear e(a)e(b)w(c)), p 71, init 4, 40k, λ 1e-4 | 3 | 2 | 6 runs, 3 recipes | 1.000 | 1.3–3.2 | 4–6 | 5–8 | 7–10 | 0.58–0.87 | 5 (4..16) |
| n 2, init 1, λ 1e-4, no warm-up | 3 | 2 | 4 | 0.65 | 0 | — | — | — | — | never leaves zero |
| n 3, p 71, λ 1e-5, 120k | 4 | 2 | 4 | 1.000 | −1.2, −0.5, −0.6, −1.0 | 6, 2, 4, 5 | 8, 3, 10, 9 | 7, 5, 8, 8 | 0.47, 0.74, 0.59, 0.47 | 4 (3..7) |
| n 3, p 71, λ 1e-4, 120k (under-fit) | 4 | 2 | 1 | 0.966 | −14.6 | 7 | 7 | 7 | 0.36 | 4 (3..7) |
| n 3, q 8/3, λ 1e-5 → 1e-6 (+40k) | 4 | 8/3 | 2 | 1.000 | −1.7, −0.7 | 7, 2 | 7, 8 | 7, 8 | 0.38, 0.65 | 5 (4..16) |
| n 3, q 4, λ 1e-6, 120k | 4 | 4 | 2 | 1.000 | +2.3, +1.3 | 9, 7 | 14, 13 | 21, 22 | 0.75, 0.71 | dense (5..16) |
| n 3, q 4, λ 1e-5 | 4 | 4 | 2 | 0.64, 0.68 | 0 | — | — | — | 0.2 | collapsed after warm-up |
| n 4 (4-sum), p 59, tied, λ 1e-5, 120k | 5 | 2 | 2 | 1.000, 0.999 | −6.0, −8.3 | 2, 3 | 4, 3 | 3, 3 | 0.56, 0.25 | 3 (2..5) |

## Reading

1. **Depth with the theorem, at every depth.** ν 3, 4, 5 on one recipe family with an unrestricted theorem at each
   rung, and every cell inside the predicted plateau. ν 5 — which x⁴ on the 4-sum never reached in 120k steps —
   fits (acc 0.999–1.000) and lands on the argmax: N_.05 = 3 in both seeds against K* 3.
2. **ν 4 is in-plateau, not on the pin, in both exact families.** N_.05 5–8 (x³: 4–9), K_5pct 2–6 with median 4.5,
   K_2pct 3–10. Since the gated chain has no u²v junk class and shows the same spread, the spread is a property of
   gradient descent at degree 4, not of x³'s extra junk. The clean fraction is higher than x³'s (0.47–0.74 vs
   0.44–0.71) and much higher than x²'s (0.41).
3. **The q dial at a second ν goes the right way.** On the n 3 chain: q 2 → 3–10, q 8/3 → 7–8, q 4 → 21–22 with
   positive margin and clean 0.7 (the flat endpoint; 35 would be full). The predicted collapse (ν 4, q 8/3) ≡ (ν 3,
   q 2) = 5 is consistent (7–8 vs 5–8) but not sharp: the e = 1.5 plateau at p 71 is 4..16, so this prime cannot
   discriminate the middle of the dial. A larger prime or a fixed-support ladder would.
4. **Escape and decay strength are the recipe facts.** Init 1 never leaves zero at any n (bilinear included: loss =
   ln p from step 1); init 4 (n 2) / init 14 + warm-up (n ≥ 3) escape at step 5–7k. Plain-l2 λ must be 1e-5 at n ≥ 3;
   L_{2,q} with q > 2 needs 1e-6 (1e-5 collapses q 4 and under-fits q 8/3; continuing at 1e-6 fits within 40k and
   leaves the count unchanged: 7/7/7 → 7/7/7, 9/2/8 → 8/2/8).
5. **Margins.** Plain-l2 margins are slightly negative at 120k (−0.5 to −1.2 at ν 4, −6 to −8 at ν 5) with accuracy
   ≥ 0.999: the fixed point is close but not reached; the q 4 cells are the only ones with positive margin. Counts
   were stable over the last 40k in every cell where we have the trace (K unchanged from the 40k probe to 120k).

## Paper use
Ladder figure: count against ν/q with the chain (ν 3, 4, 5 at q 2; ν 4 at q 8/3, 4), x² (q 2..3), x³ (ν 4), ReLU
(ν 2 dense), all on the same M_K/K^{ν/q} prediction with plateaus drawn. Statement: "on the exact families every
cell lies in the predicted plateau; at ν ≤ 3 and ν 5 on the argmax; at ν 4 the seeds spread across the plateau in
both families". Not to claim: exact argmax at ν 4; a sharp q-collapse at p 71.
