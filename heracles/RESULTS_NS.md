# NS — the exact ν ladder on n-operand sums (2026-09-16/18, three 4090 pods, ~$15, all deleted after verified pulls)   [runs/NS, runs/NS_probe, nsum.py, nsum_pod.sh, THEORY_nsum.md]

Question: does the count K*(p, ν) = argmax_K M_K/K^{ν/2} hold on the family where it is an unrestricted theorem at
every degree — x^n activation on the n-operand sum a_1 + … + a_n ≡ c (mod p), one hidden layer, no biases, full table,
full-batch GD, plain weight decay — with ReLU on a + b (Levine's MAIS-O52 network) as the ν = 2 dense anchor?
Instruments (all fp64 on the full table): margin; γ = margin/‖θ‖^ν; clean projection g(D) over D = Σa − c and its
count at 5% / 2% energy; Levine's spectral weights b_k² and N_.05; clean fraction Σ b_k² (share of the centred logit
energy on the characters of Σa − c); per-embedding single-frequency share; neuron histogram. Predictions from
kstar.py (q = 2) at p 71: ν 3 → 5 (plateau 4..10), ν 4 → 4 (3..7), ν 5 → 3 (3..5); ν 2 → dense.

## Table (p = 71, width 500, λ = 1e-4 unless stated; "anneal" = resumed at λ 1e-5 for 60k steps)

| cell | ν | seeds | acc | margin | K_5pct | N_.05 | clean | 1-freq share | K* (plateau) |
|---|---|---|---|---|---|---|---|---|---|
| ReLU, a+b, 40k (O52 net) | 2 | 2 | 1.000 | 5.6, 5.8 | — | 32, 33 of 35 | 0.92, 0.93 | 0.93 | dense |
| x², a+b, Morwani recipe 40k (RESULTS_QM) | 3 | 4 | 1.000 | 1.4–1.5 | 5, 5, 3, 2 | — | 0.41–0.45 | 0.95–1.00 | 5 (4..10) |
| x², a+b, anneal λ 1e-5 from the above | 3 | 2 | 1.000 | 4.1, 3.4 | 5, 5 | 5, 6 | 0.44, 0.40 | 0.97, 0.95 | 5 (4..10) |
| x², a+b, Morwani L_{2,3} (RESULTS_QM) | 3 | 4 | 1.000 | 4.9–5.1 | 0 (flat) | all 35 | 0.99 | 0.95–1.00 | dense under L_{2,3} |
| x³, a+b+c, untied, 120k (fixed point) | 4 | 4 | 0.999–1.000 | −1.3, −0.3, −1.4, −0.5 | 9, 2, 6, 5 | 9, 5, 8, 5 | 0.45, 0.71, 0.53, 0.61 | 0.94–0.98 | 4 (3..7) |
| x³, a+b+c, untied, anneal λ 1e-5 (+60k) | 4 | 4 | 1.000 | −0.4, 0.6, −1.1, 0.5 | 7, 3, 6, 5 | 9, 4, 7, 5 | 0.44, 0.71, 0.56, 0.61 | 0.93–0.97 | 4 (3..7) |
| x³, a+b+c, tied (exact multiset table), 120k | 4 | 2 | 1.000 | 0.34, 0.41 | 9, 10 | 13, 12 | 0.42, 0.43 | 0.97, 0.98 | 4 (3..7) |
| x³, a+b+c, L_{2,4} foil λ 1e-5 / 1e-6, 40k | 4 | 1+1 | 1.000 | −0.3 / 0.9 | 7 / 7 | 11 / 16 | 0.67 / 0.64 | 0.93 / 0.87 | dense under L_{2,4} |
| x⁴, 4-sum, tied, p 31, 120k, lr 0.8 | 5 | 1 | 0.94 | −14 | 7 | 11 of 15 | 0.31 | 0.92 | 3 (2..4) |
| ReLU, a+b+c, 40k | 2 | 2 | 0.83 | −7 | — | 23 of 35 | 0.24 | 0.18 | dense (width-limited fit) |

Junk-free constructions for x³ at p 71 (64 neurons per frequency, sign patterns cancel every non-clean monomial
exactly): γ = 4.0, 7.3, 7.9, 7.8, 6.8 ×1e-8 at K = 2..6 — argmax K = 4 as the formula says, but the tied trained nets
reach γ 7.1–8.9e-8 with 12–13 frequencies and 43% junk, so this construction is not the ceiling at ν 4.

## Reading

1. **ν 2 is dense and clean, on our own recipe.** The O52 network (ReLU, a + b, plain weight decay) uses 32–33 of 35
   frequencies at N_.05, every frequency populated, clean fraction 0.92, single-frequency neurons. Same recipe, same
   prime, x² gives 5. The dense/sparse switch is the degree, as the ν = 2 theorem (THEORY_nsum §2) says.
2. **ν 3 holds and is λ-stable.** Morwani's recipe with plain l2 gives K* = 5 at p 71 (RESULTS_QM); resuming the
   converged nets at λ 1e-5 for 60k steps leaves the clean count at 5 (5 and 6 at N_.05), grows the margin with the
   norm, and leaves the clean fraction at 0.40–0.44.
3. **Every plain-l2 net is about half junk.** Decomposing the logit tables into Fourier classes: the x² nets carry
   0.41–0.51 of their energy on the clean characters and 0.41–0.46 on the two-operand non-clean classes (the
   uncancelled uv cross terms); Morwani's L_{2,3} nets on the same architecture are clean to 0.99. The junk survives
   the λ → 0 continuation, so it is a property of the GD fixed-point family, not a finite-λ artefact. The theorem
   prices the junk-free optimum; GD finds a junk-carrying solution whose CLEAN committee has the predicted size
   (and, from the heracles margin redo, 0.6–0.8 of the constructed ceiling).
4. **ν 4 is open on the dynamics side.** The x³ endpoints are true GD fixed points at λ 1e-4 (loss constant to four
   digits over the last 20k steps) with accuracy 0.999, margin ≈ 0, 45–71% clean, single-frequency neurons, and a
   clean-projection count that varies 5–13 across seeds and tying, against a predicted 4 (plateau 3..7). The λ → 0
   continuation keeps each seed's structure (N_.05 9, 5, 8, 5 → 9, 4, 7, 5; clean fractions unchanged), lifts three of
   four margins above zero, and leaves three of four seeds inside the plateau and the median at 6 = K* + 2. The junk is
   the two-operand class (u²v cross terms, 33–47%), which needs 8-neuron sign-pattern cancellation at degree 4 where
   degree 3 needs 2 (cos² + sin² is a constant). The naive junk-free K = 4 construction is beaten on normalised
   margin by the trained tied nets, so we cannot say from margins whether the ν 4 optimum is the sparse committee.
   Claim in the paper: theorem at every degree, count verified at ν 3, GD at ν 4 reaches junk-carrying fixed points
   with N_.05 in 4..9 (median 6, predicted 4, plateau 3..7) that are stable under λ → 0; whether the ν 4 optimum is
   the sparse committee is undecided by margins (open).
5. **ν 5 is out of reach for GD here.** x⁴ on the 4-sum needs a decay-free warm-up to leave zero at all (no init
   escapes with decay on at p 59/71), then sits in a dense junk-heavy state (14 of 15 frequencies within 4×, clean
   0.31, negative margin) through 120k steps at p 31 at every λ and lr tried; λ 1e-3 underfits to four tiny clean
   frequencies. Theorem intact, no measurement.
6. **The foil goes the predicted way.** L_{2,4} at λ 1e-6 populates 24 frequencies and reaches N_.05 16 (vs 5–13
   for l2) with positive margin; at λ 1e-4 it collapses after escape (its pull is degree 4 like the loss gradient,
   unlike L_{2,3} at degree 3), which is why the foil needs the warm-up and a smaller λ.

## Recipe facts (all in nsum.py; see THEORY_nsum.md for the architecture)
- Escape from zero: the gradient/decay ratio at init scales as s^{ν−2}, but a large init only makes the random logits
  large; the first thousands of steps shrink them and decay then eats the norm. What works: x² init 1–4 (QM);
  x³ init 14 plus a decay-free warm-up until train CE < 0.8 ln p, lr up to 12.8 (clipped at 1) during the warm-up,
  then lr cap 3.2 (12.8 diverges or oscillates at degree 4); x⁴ needs the warm-up and lr ≤ 0.8.
- λ = 1e-4 everywhere; 1e-3 underfits x⁴ and collapses x³ after escape.
- Tied embeddings + multiset table are an exact reduction (verified 1e-14): same loss, gradient, margin, spectrum.
- 40k steps is under-converged at degree 4 (CE 0.07 still falling); the fixed point is reached by ~100k.
- The ReLU three-operand net plateaus at accuracy 0.83 (width-limited); the ν 2 anchor is the two-operand net.
