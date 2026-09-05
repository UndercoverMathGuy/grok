# The +1 reworked — C8/C9 inverted (2026-08-24)

Follow-up to `notes/adversarial_review_nudial_2026-08-24.md`. New tools:
`nudial/nu_eff.py` (auditable ν_eff), `nudial/construct_gamma.py` (leak-free
constructive γ(K)), `nudial/construct_check.py` (gold-check + polish),
`nudial/margin_polish.py` (verified-monotone ascent). New artifacts:
`notes/nu_eff_replication.json`, `notes/construct_gamma.json`,
`notes/construct_polish.json`.

## 1. ν_eff is now auditable — and replicated

`nudial/nu_eff.py` recomputes the scaling test from the raw `runs_dial`
checkpoints (148 grokked of 192, same count as the pack). Result vs the
committed `nu_eff_dial.json`:

- `all_nu_logit_med`: per-run corr **0.9999**, max |diff| 0.007; median 6.61.
  C6's headline number is real and now regenerable.
- margin variant: corr 0.85, median 5.99 vs committed 6.13 — their margin
  definition differed in detail; same conclusion (≈6).
- New facts the old JSON lacked: **bias share of ‖θ‖² ≈ 2×10⁻⁴** (the MLP
  biases exist but are immaterial — homogeneity is effectively exact), and
  **strong t-dependence**: ν_eff = 7.56 at t=0.8, 6.61 at t=1, 5.97 at
  t=1.25. "6 + ε_softmax" is a statement about the derivative at t=1 only;
  say so in the paper.

## 2. The construction that settles C9

`construct_gamma.py` builds K-committee networks in the exact cohort-A
architectures (legal parameter settings of the trained nets, m=512,
d=r=128): per frequency, an n×n **product grid** of phase pairs
(φᵢ, ψⱼ); preactivation cos(ka+φ)+cos(kb+ψ); readout **cos(kc+φ+ψ)** (sign
matters — the docket §2.7's "cos(kc−φ−ψ)" is a slip: it phase-averages the
signal to zero; verified numerically). On a product grid every ReLU harmonic
(m,n)≠(1,1) with m−1 or n−1 ≢ 0 (mod n) cancels **exactly** — the leak that
§2.7/§8 treat as unavoidable is a *choice*; the number of phase pairs is
norm-free (embedding and unembed scales are shared per frequency). Block
scales are balanced in closed form
(γ = g / (B^{B/2}·√∏ᵢcᵢ) for B scale groups). Free per-frequency amplitudes
polished by coordinate ascent. Gold-checked: `construct_check.py` pushes the
tensors through `train_nudial.forward` and reproduces γ to all printed
digits.

γ(K), optimal V1 supports, p=113 (gamma_free = amplitude-polished):

| K | relu-4 | relu-6 | relu-5 (eq amps) |
|---|---|---|---|
| 1 | 8.1e-8 | 3.0e-9 | – |
| 2 | 2.6e-6 | 4.7e-8 | 3.5e-7 |
| 3 | 5.0e-6 | **6.2e-8** | 5.7e-7 |
| 4 | **6.4e-6** | 5.9e-8 | **6.2e-7** |
| 5 | 6.2e-6 | 4.6e-8 | 5.4e-7 |
| 6 | 5.6e-6 | 3.5e-8 | 4.6e-7 |
| 7 | 5.0e-6 | 2.7e-8 | – |

- **Argmax at K\* in every family** (4 / 3 / 4) — the ansatz does not
  undercount.
- **The K\* construction beats every one of the 576 trained runs per
  family** (A+G, m=512): trained max 6.07e-6 (relu-4), 6.11e-8 (relu-6),
  5.67e-7 (relu-5). The docket's registered falsifier §6.6 ("a stronger
  maximiser at K=K\* beating AdamW's γ kills C9's interpretation") is HIT.
- Tied-phase (diagonal-grid) constructions are 10–40× worse — this is why
  the pack's Morwani-style hand constructions and D/H/J ascent looked so
  weak, and why trained nets seemed unbeatable.
- The monotone polisher cannot improve the construction (0% accepted
  steps): at least a local max, and unlike J it provably never regresses.

## 3. The pack's own data agrees: γ stratified by trained K

Never computed in the pack — γ medians by K_gap over trained A+G m=512 runs:

| K | relu-4 med (n) | relu-6 med (n) | relu-5 med (n) |
|---|---|---|---|
| 3 | −1.1e-6 (5) | **5.00e-8 (135)** | **5.18e-7 (7)** |
| 4 | **4.78e-6 (52)** | 4.54e-8 (289) | 4.79e-7 (39) |
| 5 | 4.62e-6 (227) | 3.53e-8 (97) | 4.35e-7 (39) |
| 6 | 4.41e-6 (174) | 1.17e-8 (9) | 3.81e-7 (9) |
| 7 | 3.89e-6 (48) | 9.2e-9 (2) | 1.6e-7 (2) |

Within trained runs, γ **peaks at K\* (or below) and falls monotonically in
K**. The best individual trained runs are K=K\* nets (relu-4: K=4 at
6.07e-6; relu-6: K=3 at 6.11e-8). The docket's §6.3 prediction ("measured
γ(K) peaks at K\*+1") is refuted by data it already contains.

## 4. The reworked story

- **C9 is dead, inverted.** "AdamW-trained K\*+1 nets have the highest
  measured γ of anything produced" was an artifact of weak baselines
  (tied-phase constructions; a non-monotone ascent that regressed 32/32 warm
  starts). The true ordering: construction(K\*) > trained(K\*) >
  trained(K\*+1) > trained(K\*+2) > … > D/H/J.
- **C8 splits.** The one-hot-quad cross-talk term survives (not retested
  here). The relu "+1 = leak cancellation" half dies: the leak is
  eliminable at zero norm cost, the optimum is at K\*, and democratic
  amplitudes at K\*+1 are *not* better than K\* — trained K\*+1 nets sit
  ~3–15% below trained K\* nets and ~25% below the construction.
- **The +1 is a dynamics/reachability bias, full stop.** Trained mode =
  K\*+1 (relu-4/6) or K\* (relu-5) while the γ ceiling is at K\*. This
  revives the Round-3 memorisation-remnant/reachability reading — it was
  "killed" only by C9's margin argument, which is now void. Consistent with
  F (small init → K<K\*, lower γ; the seed barrier) and with
  memorisation-selects-which. C11's tilt γ_t>0 becomes a clean dynamics
  parameter, not an architecture margin term.
- **C5 unchanged** (the dial is real). **C3 strengthened** (K\* is now
  confirmed as the realised-architecture optimum, not just the ansatz
  optimum). **C6's measurement replicated**, with the t-dependence caveat.

## 5. What still needs a pod

- p=1009 dial (C4's slope) — unchanged.
- Fixed-support AdamW training (§6.3) — now *optional* for the +1 verdict
  (settled above), still nice for P(K) calibration.
- Quad-3: a product-grid-style search for a sparse one-hot-quad solution
  beating 6.1e-6 (C8's surviving half is still lower-bound-vs-lower-bound).
- Re-train a few cohort cells with weight saving: **no cohort weights
  survive** (pods deleted; tarballs contain spectra/config/metrics only), so
  no weight-space claim about A–J can ever be re-examined. Add to W-list.

## 6. Corrections to the docket this forces

- §2.7 readout sign: cos(kc+φ+ψ).
- §2.5 killed-candidates table: remove "finite-temp/memorisation-remnant —
  dead" (the kills used C9); reinstate reachability as the live explanation.
- §6.3/§6.6: both falsifiers are now resolved (peak at K\*; construction
  beats AdamW).
- W3 upgraded from "in principle" to "realised".
- New W: cohort weights unrecoverable.

## 7. Second wave (same day): C8 dies entirely — the optimum has NO architecture term

- **Quad-3 one-hot, 3×3 product grids** (the quadratic activation generates only
  |q|≤3 phase multipliers, so a 3-grid cancels every non-signal term; quad DOES
  pay √n for phase diversity — no shared embedding to amortise — but 3× is
  affordable): γ(K) = 5.2e-6 / 1.26e-5 / 1.83e-5 / **2.01e-5** / 1.95e-5 /
  1.78e-5 for K=2/3/4/6/8/12; leak-free dense = 1.02e-5.
  **Peak at K=6 = K*(3); 3.3× above cohort D's dense 6.1e-6; sparse/dense
  = 2.0×, matching the ansatz ratio exactly.** P-N7's quad kill was a
  tied-phase artifact. Trained quad density (k_eff 44–51) is a reachability
  failure of magnitude 3.3×, not an optimum.
- **One-hot relu-4 (0,2)**: γ(K) peaks at K=4 (6.12e-6), within 4% of the
  embedded (1,1) construction — the E-cohort placement effect (k_eff 8.84 vs
  4.19) is also dynamics (that cell groks 42/96 — bad reachability), not a
  different optimum.

Unified claim this forces: **the ℓ2 max-margin optimum is K*(ν,p) in every
cell tested (relu-4/5/6 embedded, relu-4 one-hot, quad-3 one-hot) —
architecture enters the optimum ONLY through ν.** All architecture-specific
committee phenomena (the +1, quad density, one-hot density, placement) live
in the dynamics layer: how far the trained endpoint falls short, from +1
(embedded relu) to +dense/3.3× (one-hot quad). C8's cross-talk mechanism is
real but it is a statement about which basins standard training reaches, not
about the optimum. ν's status (C6, path rule) is unchanged and is now the
single architecture input to the optimum layer.

## 8. Scope guardrails (added after review discussion — do not drift)

Risk: the "dynamics layer" degenerating into an unfalsifiable init-lottery
catch-all (the road the identity program already mapped: AUC 0.7–0.9,
residual = chaos). Rules:

1. This paper's dynamics claim is about the COUNT, not the identity. The
   count is regular (mode K*+1 in 96/96 cells, width- and wd(1–3)-invariant,
   near-ties landing on landscape flats) — a distributional law, not a
   lottery. WHICH frequencies = the identity program, separate paper.
2. γ_t stays phenomenological: a measured constant per (architecture,
   init-scale). No decomposition into init microstructure, no epoch-0
   readouts, no W_E anatomy here.
3. Claims are about P(K) over seeds, never per-seed.
4. Falsifiers that pin layer 2 down (register them): γ_t K-independent
   within a family (histogram shape constraint); sign flips with approach
   side (F: confirmed); dispersion widens with p (flat-top width); seeded-K*
   holds under training. ~2 parameters vs ~30 cell observations.
5. The causal experiments (seeded-optimum, ablate-at-transition) manipulate
   the committee, not the init — they close the story without opening init
   structure. If seeded quad-3 sparse collapses to dense, the explanation
   moves to loss-geometry instability, not init — follow the data there.
6. Phrasing: "one-sided tilt from eviction-only dynamics; sources of the
   tilt deliberately not decomposed." Not "init decides."
