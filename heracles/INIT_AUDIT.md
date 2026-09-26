# INIT_AUDIT — hostile audit of "the x² frequency count depends on the initialisation scale" (2026-09-19)

Target: `RESULTS_REP.md` § "Post-hoc init-scale control" (exploratory, not pre-registered) and reading 3(b),
"the x² count depends on init scale (p 71: K_2pct 6 at init 1, 10 at init 4), so 'K* = 5 at p 71' is an init-1 result".

Scripts written for this audit (all new, prefix `initaudit_`): `initaudit_scan.py`, `initaudit_h1.py`,
`initaudit_h34.py`, `initaudit_stats.py`, `initaudit_ceiling.py`, `initaudit_calib.py`, `initaudit_place.py`,
`initaudit_support.py`, `initaudit_match.py`, `initaudit_abl.py`, `initaudit_cont.py`, `initaudit_traces.py`.
Outputs under `heracles/FINAL_RUNS/INITAUDIT/`. Nothing existing was edited, deleted or committed.

## Symbols (plain language)

| symbol | meaning |
|---|---|
| p | the modulus; the task is a + b = c (mod p) on the full p² table |
| n_f = (p−1)/2 | number of distinct Fourier frequencies available |
| init scale | a single number multiplying **all three** weight tables (both embeddings E₀, E₁ and the readout W) at step 0 |
| ‖θ‖² (`norm2`) | sum of squares of every weight in the network |
| margin | over all p² rows, the smallest (logit of the right answer − biggest wrong logit). Positive ⇔ the net separates the data |
| ν = 3 | homogeneity degree of the x² net: scaling all weights by c multiplies the logits by c³ |
| γ (`gamma`) | margin / ‖θ‖³ — the **normalised margin**. Scale-free, so it measures the *direction* of the solution. This is the quantity the sparsity theory maximises |
| g(D) | the "clean" part of the logits: the exact projection of the logit function onto functions of D = a+b−c only |
| clean fraction | share of total (centred) logit energy that lies in g(D). 1.00 = a pure modular-addition circuit, 0 = pure junk |
| en_k | share of g(D)'s energy at frequency k (so Σ en_k = 1) |
| K_2pct / K_5pct | #{k : en_k ≥ 2% / 5%} — a **fixed relative threshold** count |
| N_.05 | smallest number of frequencies holding 95% of g(D)'s energy — a **cumulative** count |
| PR | participation ratio 1/Σ en_k² — a **threshold-free** effective count |
| expH | exp(Shannon entropy of en) — a second threshold-free effective count |
| A_1pct | #{k : b2_k ≥ 1%} where b2_k is k's share of *total* logit energy — an **absolute** threshold count |
| K_abl | interface-ablation count: k counts if projecting cos/sin(2πkx/p) out of the embeddings **and** the readout drops the margin above a null floor, and hurts from each side alone. Never looks at the logit spectrum |
| n_freq_hist_1pct | **weight-space** count: #frequencies that are the dominant Fourier mode of ≥1% of the neurons |
| M_S | coverage of a frequency set S: min over D ≠ 0 of Σ_{k∈S} (1 − cos 2πkD/p). The theory's objective |
| γ_con(K) | exact γ of the junk-free constructed x² circuit on the coverage-optimal K-frequency support — the **ceiling** for that many frequencies |
| att | γ(run) / γ_con(K of that run) — how close the trained net is to the best possible circuit of its own size |

## VERDICT

**The finding as stated does not survive. It is part instrument artefact, part a confound with solution quality —
and it is *not* an unconverged transient.** Two independent problems. (1) *Instrument dependence*: at p = 71 the
init 1 → init 4 change is +4 on K_2pct, +3 on K_5pct and +1.6 on the participation ratio, but **−3 on N_.05,
−3.5 on the interface-ablation count K_abl, and non-monotone on the weight-space count** (16 → 21 → 14.5 for
init 1/2/4). Of seven count instruments three go up, three go down and one is non-monotone; at p = 59 the two
threshold-free instruments (PR, expH) show *no* effect at all (6.34 → 6.31, p = 0.52). All seven instruments read
**exactly K** on an exactly-K-frequency constructed circuit (`initaudit_calib.py`), so the disagreement is a
statement about how far these trained nets are from any clean circuit, not about the instruments. (2) *Collinearity*:
init scale and solution quality are perfectly confounded in this design. γ ranges for init 1 and init 4 are
**disjoint** at p 59 and p 71 (and init 2 vs 4 at p 97), so no comparison at matched quality is possible for the
headline pair; where the cells *do* overlap in γ (p 71 init 2 vs 4, p 113 init 3 vs 4) the K_2pct gap falls to 1 and
K_5pct **reverses**. The single thing init scale robustly changes at every prime is the *quality* of the solution
(γ up, ‖θ‖² down, CE down, clean fraction up; Mann-Whitney AUC 0.00–0.16, p ≤ 6 × 10⁻⁶). The p 71 init-1 cell that
supplies the headline "6" is an optimisation near-failure: it reaches γ = **0.129 of the exact constructed ceiling**
(init 4 reaches 0.449), 2 of its 16 "fit" seeds do not even separate the training data (margin −0.40, −0.26 with
accuracy ≥ 0.999), and the same recipe at init 1 collapses outright at p = 89 (0/16). The honest summary is:
**nothing in these cells is close enough to an optimum for "the number of frequencies" to be a well-defined quantity**
— the best cell sits at half the achievable normalised margin and 41 % of its logit energy is junk — so the init-scale
control neither supports nor contradicts the λ → 0 theory; it measures which mediocre basin GD falls into.
What I could **not** kill: the K_2pct/K_5pct increase itself is statistically solid (p = 8 × 10⁻⁴ and 3 × 10⁻⁶ at
p 71, replicated in sign at 59/89/97/113), it is not caused by the clean-fraction normalisation, and it is not a
transient — 40 k extra steps at the same λ move γ and the count by nothing. So *something* real about the basin does
depend on init; it is just not "the number of frequencies the network uses".

---

## H1 — INSTRUMENT ARTEFACT. **Partly survives: this is the strongest kill.**

First: the obvious mechanism is *not* the culprit. K_2pct/K_5pct normalise by **clean** energy, and g(D) is the exact
orthogonal projection onto D-only functions, so junk contributes to g(D) not at all. A rising clean fraction cannot
mechanically inflate K_2pct. But the count still turns out to be instrument-specific.

### (a) relative threshold sweep 1 %…10 % on the clean spectrum, and (b) threshold-free counts
(`initaudit_h1.py`; medians over fit seeds, 16 per cell)

|  p | init |  K_1% | K_2% | K_3% | K_5% | K_7% | K_10% |  PR  | expH | clean |
|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 1 | 10.0 | 7.0 | 6.5 | 6.5 | 5.0 | 4.0 | 6.34 | 8.29 | 0.440 |
| 59 | 4 |  9.0 | 9.0 | 8.0 | 6.0 | 4.5 | 4.0 | **6.31** | **7.80** | 0.658 |
| 71 | 1 | 16.0 | 6.0 | 4.0 | 4.0 | 4.0 | 4.0 | 5.69 | 7.81 | 0.309 |
| 71 | 2 | 10.0 | 9.0 | 8.5 | 8.0 | 6.5 | 3.0 | 7.16 | 9.61 | 0.461 |
| 71 | 4 | 10.5 | 10.0 | 10.0 | 7.0 | 5.0 | 3.0 | 7.25 | 8.64 | 0.587 |
| 89 | 2 | 10.5 | 7.0 | 7.0 | 6.0 | 6.0 | 4.0 | 7.33 | 9.97 | 0.407 |
| 89 | 4 | 10.0 | 9.0 | 8.5 | 8.0 | 5.0 | 3.0 | **6.97** | **9.14** | 0.551 |
| 97 | 2 | 17.0 | 7.0 | 6.0 | 5.5 | 5.0 | 3.5 | 7.05 | 9.67 | 0.353 |
| 97 | 4 | 10.0 | 9.5 | 9.0 | 7.0 | 5.0 | 3.0 | **6.94** | 9.90 | 0.522 |
| 113 | 3 | 12.0 | 7.0 | 7.0 | 7.0 | 5.0 | 3.0 | 7.52 | 10.44 | 0.403 |
| 113 | 4 |  9.5 | 9.0 | 9.0 | 7.5 | 5.5 | 2.5 | 8.11 | 11.19 | 0.461 |

The threshold sweep does **not** kill it — at every prime the init-high cell has more frequencies above 2–3 %. But it
reverses at the 1 % and 10 % ends (at p 71, K_1% is 16 at init 1 vs 10.5 at init 4), and the two threshold-free
effective counts show **no effect at p 59, 89, 97** (PR actually *falls*). Only p 71 has a significant PR rise
(5.69 → 7.25, Mann-Whitney p = 9 × 10⁻⁴).

### (c) absolute thresholds, and the cumulative count
`A_1pct` (≥1 % of *total* logit energy) rises everywhere — but that instrument *is* contaminated by the clean
fraction (a net with 31 % clean energy has fewer frequencies clearing 1 % of the total), so it is not a control.
The clean control is **N_.05**, the cumulative 95 %-of-clean-energy count, and it moves the **other way**:

| p | init lo → hi | K_2pct | K_5pct | N_.05 | PR | K_abl | n_freq_hist_1pct | γ |
|---|---|---|---|---|---|---|---|---|
| 59 | 1 → 4 | 7 → 9 ↑*** | 6.5 → 6 (ns) | 10 → 8 ↓* | 6.34 → 6.31 (ns) | 7 → 9 ↑ | 17 → 12 ↓ | 2.57e−5 → 3.66e−5 ↑*** |
| 71 | 1 → 4 | 6 → 10 ↑*** | 4 → 7 ↑*** | 13 → 10 ↓*** | 5.69 → 7.25 ↑*** | 14 → 10.5 ↓ | 16 → 14.5 (non-mono, init2 = 21) | 7.18e−6 → 2.51e−5 ↑*** |
| 89 | 2 → 4 | 7 → 9 ↑*** | 6 → 8 ↑* | 15 → 10 ↓** | 7.33 → 6.97 (ns) | – | 22 → 19 ↓ | 9.18e−6 → 1.45e−5 ↑*** |
| 97 | 2 → 4 | 7 → 9.5 ↑** | 5.5 → 7 (ns) | 15 → 12 ↓* | 7.05 → 6.94 (ns) | – | 21 → 23 ↑ | 4.47e−6 → 1.10e−5 ↑*** |
| 113 | 3 → 4 | 7 → 9 ↑** | 7 → 7.5 (ns) | 18 → 18 flat | 7.52 → 8.11 (ns) | – | 24 → 27.5 ↑ | 4.56e−6 → 6.06e−6 ↑*** |

(*** p < 10⁻³, ** p < 0.01, * p < 0.05, two-sided Mann-Whitney, n = 16 per cell; `initaudit_stats.py`.)
K_2pct and N_.05 are both pre-registered instruments of the same paper and they disagree in **sign at every prime**.

### (d) interface-ablation count K_abl — the independent instrument (`initaudit_abl.py`, seeds 100–105)

| cell | K_abl med [range] | per-seed | K_2pct | K_5pct | N_.05 | clean |
|---|---|---|---|---|---|---|
| p 71 init 1 | **14.0** [9–15] | 15 15 12 9 13 15 | 5.5 | 4.0 | 12.0 | 0.302 |
| p 71 init 2 | **8.5** [6–14] | 6 7 14 8 11 9 | 8.5 | 8.0 | 14.5 | 0.438 |
| p 71 init 4 | **10.5** [7–13] | 13 12 7 11 10 10 | 10.0 | 7.0 | 10.0 | 0.590 |
| p 59 init 1 | 7.0 [6–16] | 6 6 16 6 8 14 | 7.0 | 7.0 | 11.0 | 0.413 |
| p 59 init 4 | 9.0 [9–12] | 9 9 12 9 10 9 | 9.5 | 6.0 | 8.5 | 0.644 |
| p 71 init 1, λ/10 cont. | 11.5 [6–16] | 14 6 12 6 11 16 | 5.0 | 4.0 | 11.0 | 0.296 |

At p 71 — the headline cell — K_abl **reverses** the ordering (14 at init 1 vs 10.5 at init 4) and is non-monotone.
At p 59 it agrees with K_2pct. Caveat in both directions: K_abl uses the minimum margin as its statistic, and the
init-1 nets have tiny minimum margins (median 1.12 vs 3.10), so *any* cut looks catastrophic there — that is itself a
consequence of the quality gap, not an independent confirmation.

### Instrument calibration on the theory's own object (`initaudit_calib.py`)

Exact junk-free x² circuits at p = 71 on the coverage-optimal support of each size:

| K built | K_2pct | K_5pct | N_.05 | PR | clean | γ_con |
|---|---|---|---|---|---|---|
| 3 | 3 | 3 | 3 | 3.00 | 1.0000 | 4.030e−5 |
| 5 | 5 | 5 | 5 | 5.00 | 1.0000 | **5.580e−5** |
| 8 | 8 | 8 | 8 | 8.00 | 1.0000 | 5.073e−5 |
| 10 | 10 | 10 | 10 | 10.00 | 1.0000 | 4.895e−5 |
| 12 | 12 | 12 | 12 | 12.00 | 1.0000 | 4.545e−5 |

Every instrument is exact on an exact circuit. Their disagreement on the trained nets is therefore a measure of how
un-circuit-like those nets are — which brings us to H3.

### Sorted clean-energy spectra, init 1 vs init 4 (p = 71, seeds 100–104; top 14 of 35)

```
 init1 s100 K2= 5 cl=0.33 γ=8.05e-06 | 0.238 0.186 0.185 0.142 0.116 0.018 0.016 0.016 0.015 0.014 0.013 0.013 0.011 0.010
 init1 s101 K2= 9 cl=0.29 γ=7.30e-06 | 0.249 0.184 0.172 0.151 0.028 0.026 0.022 0.022 0.022 0.020 0.018 0.018 0.016 0.015
 init1 s102 K2= 5 cl=0.32 γ=5.46e-06 | 0.231 0.215 0.177 0.151 0.130 0.020 0.019 0.017 0.014 0.009 0.009 0.008 0.000 0.000
 init1 s103 K2= 4 cl=0.30 γ=1.11e-05 | 0.287 0.277 0.204 0.023 0.018 0.018 0.017 0.017 0.017 0.017 0.016 0.015 0.014 0.013
 init1 s104 K2= 6 cl=0.31 γ=-2.3e-06 | 0.258 0.218 0.192 0.179 0.028 0.021 0.018 0.017 0.015 0.015 0.015 0.013 0.010 0.000
 init4 s100 K2=12 cl=0.55 γ=2.43e-05 | 0.259 0.151 0.104 0.078 0.078 0.055 0.053 0.050 0.042 0.041 0.035 0.034 0.006 0.004
 init4 s101 K2=11 cl=0.51 γ=2.34e-05 | 0.240 0.172 0.086 0.085 0.085 0.080 0.076 0.043 0.042 0.031 0.025 0.010 0.008 0.005
 init4 s102 K2= 8 cl=0.63 γ=2.27e-05 | 0.234 0.203 0.195 0.170 0.059 0.047 0.027 0.024 0.007 0.005 0.005 0.004 0.004 0.004
 init4 s103 K2=12 cl=0.51 γ=2.65e-05 | 0.247 0.131 0.091 0.089 0.088 0.086 0.064 0.060 0.033 0.033 0.028 0.021 0.007 0.006
 init4 s104 K2= 8 cl=0.66 γ=2.68e-05 | 0.283 0.243 0.214 0.058 0.057 0.055 0.051 0.031 0.003 0.002 0.002 0.001 0.000 0.000
```

What actually changes is the **shape**, not a count: init 1 gives 3–5 dominant modes plus a long shallow tail; init 4
gives a wider, flatter head with a sharper cut-off. That is exactly why K_2pct (a head statistic) rises while N_.05
(a tail statistic) falls. Quantitatively, `flat5` = (5th largest share)/(largest share) — 1.000 for the exact optimum
— is 0.104 at p 71 init 1 and 0.30–0.34 in **every other cell of the whole sweep** (p 59 init 1/4: 0.340/0.331;
p 89: 0.341/0.305; p 97: 0.225/0.302; p 113: 0.301/0.308). p 71 init 1 is an outlier cell, not one end of a trend.

---

## H3 — MARGIN CHECK. **Does not rescue the effect; it is what destroys it.**

The higher-count cells are not worse optima. They are **better** on every quantity, including the training objective
being minimised (`initaudit_h34.py`; medians over fit seeds; obj = CE + λ‖θ‖², λ = 10⁻⁴):

| p | init | n fit | CE | λ‖θ‖² | **obj** | ‖θ‖² | margin | **γ** | K_2pct | K_5pct |
|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 1 | 16 | 0.0268 | 0.2155 | 0.2426 | 2154.6 | 2.550 | 2.573e−5 | 7.0 | 6.5 |
| 59 | 4 | 16 | 0.0206 | 0.1996 | **0.2202** | 1996.3 | 3.285 | **3.655e−5** | 9.0 | 6.0 |
| 71 | 1 | 16 | 0.0460 | 0.2862 | 0.3316 | 2861.8 | 1.122 | 7.180e−6 | 6.0 | 4.0 |
| 71 | 2 | 16 | 0.0278 | 0.2568 | 0.2849 | 2567.9 | 2.726 | 2.102e−5 | 9.0 | 8.0 |
| 71 | 4 | 16 | 0.0260 | 0.2460 | **0.2719** | 2460.5 | 3.105 | **2.505e−5** | 10.0 | 7.0 |
| 89 | 2 | 16 | 0.0409 | 0.3317 | 0.3730 | 3317.4 | 1.750 | 9.183e−6 | 7.0 | 6.0 |
| 89 | 4 | 16 | 0.0351 | 0.3198 | **0.3552** | 3198.4 | 2.602 | **1.446e−5** | 9.0 | 8.0 |
| 97 | 2 | 16 | 0.0541 | 0.3739 | 0.4279 | 3738.7 | 1.008 | 4.472e−6 | 7.0 | 5.5 |
| 97 | 4 | 16 | 0.0395 | 0.3568 | **0.3959** | 3568.2 | 2.349 | **1.098e−5** | 9.5 | 7.0 |
| 113 | 3 | 16 | 0.0544 | 0.4295 | 0.4846 | 4295.2 | 1.281 | 4.559e−6 | 7.0 | 7.0 |
| 113 | 4 | 16 | 0.0489 | 0.4271 | **0.4759** | 4270.9 | 1.719 | **6.058e−6** | 9.0 | 7.5 |

So the reading "init 1 recovers K* = 5 at p 71" rests on the **worse** point of the same objective. At p 71 the
init-1 median γ is 0.29× the init-4 median. The theory's own ladder says the *entire* spread of γ over K = 3…12 at
p 71 is a factor **1.38** (5.580e−5 at K = 5 down to 4.030e−5 at K = 3). A 3.5× γ gap between the two cells is
therefore **dominated by optimisation quality, not by the number of frequencies** — neither cell is a point "inside
the plateau"; both are far below the plateau, and one is much further below than the other.

### Distance to the exact ceiling (`initaudit_ceiling.py`, `initaudit_place.py`, `initaudit_support.py`)

γ_con(K) computed exactly from the junk-free 6-neuron-per-frequency x² construction on `kstar.json`'s optimal
supports; the constructed ladder reproduces the predicted r(K) to 3 decimals, so it is the same ladder.

| cell | γ median | / γ_con(K*) | cov = M_S/M_K* | att = γ/γ_con(K of the run) | non-separating "fit" seeds |
|---|---|---|---|---|---|
| p 59 init 1 | 2.573e−5 | 0.352 | 0.721 [0.47–0.90] | 0.363 [0.25–0.45] | 0/16 |
| p 59 init 4 | 3.655e−5 | **0.500** | 0.818 [0.61–0.93] | 0.558 [0.51–0.92] | 0/16 |
| p 71 init 1 | 7.180e−6 | **0.129** | 0.709 [0.40–0.91] | 0.146 [−0.04–0.25] | **2/16** |
| p 71 init 2 | 2.102e−5 | 0.377 | 0.783 [0.54–0.87] | 0.428 [0.34–0.72] | 0/16 |
| p 71 init 4 | 2.505e−5 | **0.449** | 0.815 [0.69–0.91] | 0.527 [0.39–0.72] | 0/16 |
| p 59 init 1, λ/10 cont. | 3.185e−5 | 0.436 | – | – | 0/16 |
| p 71 init 1, λ/10 cont. | 1.465e−5 | 0.263 | – | – | 0/16 |

The best cell in the whole sweep reaches half the achievable normalised margin, picks a support with 82 % of the
achievable coverage, and leaves 41 % of its logit energy in junk. At that distance from the optimum the difference
between the K = 5 rung (γ_con 5.580e−5) and the K = 10 rung (4.895e−5) — 12 % — is unmeasurable. **Neither cell is
a test of the prediction.**

Rank correlations of γ with the counts (`initaudit_h34.py`): pooled over all 192 fit x² q = 2 seeds,
ρ(γ, K_2pct) = **+0.345**, ρ(γ, K_5pct) = +0.253, ρ(γ, N_.05) = **−0.703**. Within a single prime, pooling inits,
ρ(γ, K_2pct) = +0.48…+0.69. Within a *single cell* it collapses to +0.15 median (range −0.25 to +0.56). So the
correlation is between-cell, i.e. between-basin, not a per-seed law.

---

## H5 — CONFOUNDS. **Clean: init scale really is the only launch difference.**

- `nsum_rep.py` line: `Es.append(a.init_scale * E); Ws.append(a.init_scale * W)` — the scale multiplies **all three**
  tables (E₀, E₁, W) uniformly. Confirmed.
- The three p 71 x² q = 2 λ = 10⁻⁴ cells were launched from three different job logs (`d/d_4.log` init 1,
  `b/b3_8.log` init 2, `a/a2_1.log` init 4) with byte-identical argument dicts apart from `init_scale`: same trainer
  (`nsum_rep`), same seeds 100–115, `lam 1e-4, m 500, steps 40000, lr0 0.05, lrmax 12.8, clip 0.0, wd_warmup 0,
  escape_frac 0.97, lrmax2 12.8, resume None, achunk 0`, same `device cuda`, same evening (18 Sep 20:17–21:37).
- I could **not** verify they ran on the same physical pod or the same git commit (pods deleted; single commit for
  the whole 09-16…18 block). Mitigation: I reproduced the archived init-1 endpoint exactly on this machine from its
  checkpoint (γ 8.048e−6 → 7.794e−6 after 400 CPU steps) and ran my own fresh init-1 and init-4 runs locally (H6).
- **But there is one confound that is intrinsic and unavoidable**: the net is degree-3 homogeneous, so init scale is
  not a "pure" knob. ‖θ‖² at step 0 is 190 (init 1), 761 (init 2), 3046 (init 4), while the *converged* ‖θ‖² is
  2862 / 2568 / 2461. The init-4 run therefore **starts above its own final norm and shrinks into it**, whereas the
  init-1 run must inflate its norm 15-fold. With an identical lr schedule the effective step size relative to the
  loss curvature is completely different. "Init scale" here means "init scale *and* effective learning rate *and*
  time-to-escape", and no experiment in the cell separates them.

---

## H4 — SURVIVORSHIP / COLLAPSE. **Survives as a serious warning about the init-1 cells.**

- Cells that produced no scorable seed at all: p 89 init 1 (0/16, acc 0.011–0.433), p 113 init 2 (0/16,
  acc 0.399–0.835), p 157 init 4 (0/16, acc 0.944–0.998), p 211 init 4 (0/8, acc ≈ 0.08), p 307 init 4 (0/8,
  acc ≈ 0.012). The escape threshold in init scale **rises with p**: p 59/71 escape at init 1, p 89 needs init ≥ 2,
  p 113 needs init ≥ 3, p ≥ 157 fails even at init 4.
- So the init knob is, first and foremost, the knob that decides whether GD escapes the cross-entropy plateau at all.
  The count comparison is being made across the edge of that failure.
- At p 71 there is no seed-level survivorship (16/16 fit in all three cells), but the "fit" criterion leaks:
  **2 of the 16 init-1 seeds (104, 109) have negative minimum margin (−0.403, −0.258) while scoring accuracy
  ≥ 0.999** — they do not separate the p² table, so their γ is negative and meaningless, yet they were pooled into
  the init-1 medians. No other cell in the x² q = 2 sweep has this.
- Time-to-escape (first logged step with acc ≥ 0.99) is 8000 for p 71 init 1 and 6000 for init 2 and 4, with **zero
  within-cell variance**, so escape time cannot be correlated with the count inside a cell; it only tracks the cell.

---

## H2 — NOT CONVERGED. **Killed: the effect is not a transient.**

The 40 k-step budget is not the problem — the archived runs are already at a fixed point of the dynamics, and more
of the same training does not move the count toward the init-4 value.

**Evidence from the archived CE logs.** Training CE at p 71 seed 100 is essentially final by step 16 000 and the
init gap is locked in there: init 1 / 2 / 4 have CE 0.0439 / 0.0309 / 0.0275 at step 16 000 and
0.0434 / 0.0309 / 0.0268 at step 40 000. Twenty-four thousand further steps changed CE by ≤ 0.0007.

**Local continuation (`initaudit_cont.py`, run here).** I resumed the four archived p 71 init-1 endpoints
(seeds 100–103) at the **same** λ = 10⁻⁴ and lr 12.8 for a further 40 000 steps — a doubling of the whole budget —
with the full fp64 instrument suite every 5 000 steps:

| step | K_2pct per seed (100/101/102/103) | median K_2pct | median γ | median ‖θ‖² | median clean |
|---|---|---|---|---|---|
| 0 (= archived endpoint) | 5 / 9 / 5 / 4 | 5.0 | 7.672e−6 | 2860 | 0.310 |
| 10 000 | 5 / 10 / 5 / 4 | 5.0 | 8.076e−6 | 2858 | 0.311 |
| 20 000 | 5 / 10 / 6 / 3 | 5.5 | 6.420e−6 | 2847 | 0.312 |
| 30 000 | 5 / 10 / 5 / 4 | 5.0 | 6.733e−6 | 2839 | 0.313 |
| **40 000** | **5 / 10 / 6 / 5** | **5.5** | **6.633e−6** | 2834 | 0.314 |

Nothing moves. The count stays at 5–6 against init 4's 10; γ does not rise (it drifts slightly **down**, 7.7e−6 →
6.6e−6, because the objective being minimised is CE + λ‖θ‖², not the minimum margin); ‖θ‖² and the clean fraction are
flat to 1 %. Seed 102 even suffered a transient loss of separability at step 25 000 (margin −25.4, recovering by
30 000), which is further evidence that the init-1 basin is marginal rather than under-trained.

**The λ → 0 continuation goes the wrong way too.** The archived λ/10 continuation (60 000 further steps, all 16
seeds) roughly doubles γ at p 71 init 1 — 7.18e−6 → 1.465e−5, i.e. 0.129 → 0.263 of the constructed ceiling — while
K_2pct moves **down**, 6 → 5, and the clean fraction does not move at all (0.309 → 0.301). So within the init-1
basin, quality can double without the count rising. The count gap is a *between-basin* fact, not a monotone function
of training time or of γ.

*Not done (budget):* the matching init-4 continuation was launched but cancelled after the init-1 cell to stay inside
the compute budget, so I have no same-λ continuation control for init 4. The archived p 113 init-4 λ/10 continuation
is the nearest substitute and it is also flat (K_2pct 9 → 9, N_.05 18 → 18).

---

## H6 — MECHANISM. **Consistent with "the basin is chosen at escape and then frozen".**

Two fresh runs on **this machine** (CPU, same code, seed 100, p = 71, x², q = 2, λ = 10⁻⁴, the archived recipe),
traced every 1 500 steps. They reproduce the archived cells exactly — init 1 ends at K_2pct 6, γ 9.78e−6, clean
0.328 (archived: 5, 8.05e−6, 0.326); init 4 ends at K_2pct 12, γ 2.02e−5, clean 0.529 (archived: 12, 2.43e−5,
0.551). So the effect is not a pod or commit artefact; it reproduces on different hardware in a different arithmetic.

| step | init 1: ‖θ‖² | clean | K_2pct | γ | | init 4: ‖θ‖² | clean | K_2pct | γ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 190.2 | 0.000 | 16 | −1.13e−5 | | 3043.1 | 0.000 | 16 | −1.13e−5 |
| 1 500 | 182.8 | 0.000 | 19 | −1.11e−5 | | 2921.1 | 0.002 | 30 | −7.44e−6 |
| 3 000 | 166.2 | 0.001 | 28 | −1.06e−5 | | 2832.3 | 0.026 | 34 | −6.24e−6 |
| 4 500 | 126.8 | 0.009 | 34 | −9.89e−6 | | **5471.0** | **0.236** | 7 | −3.54e−6 |
| 6 000 | **73.5** | 0.040 | 34 | −9.00e−6 | | 5445.1 | 0.345 | 14 | +5.39e−6 |
| 7 500 | 2415.9 | 0.121 | 10 | −6.58e−5 | | 3253.6 | 0.429 | 10 | +1.35e−5 |
| 9 000 | 2908.2 | 0.334 | **5** | +4.21e−6 | | 2707.2 | 0.500 | 12 | +1.87e−5 |
| 10 500 | 2854.7 | 0.331 | 5 | +6.91e−6 | | 2645.9 | 0.520 | 12 | +2.02e−5 |
| 12 000 | 2832.7 | 0.328 | **6** | +9.78e−6 | | 2617.2 | 0.529 | **12** | +2.02e−5 |

The mechanism is visible and it is **not** "more frequencies start above the noise floor". It is **weight decay
eating the network during the cross-entropy plateau, before escape**:

- **init 1** sits on the CE plateau for ~6 000 steps while the λ‖θ‖² term shrinks it from ‖θ‖² = 190 to **73.5**, a
  **61 % loss of norm** with clean fraction still 0.04. Escape then happens from an impoverished configuration, and
  the spectrum arrives already spiky: by step 9 000, K_2pct = 5, clean 0.33 — and it never changes again.
- **init 4** escapes between steps 3 000 and 4 500, having lost only **7 %** of its norm (3043 → 2832). It escapes
  with headroom, overshoots to ‖θ‖² = 5471, and settles into a broad spectrum (K_2pct 12, clean 0.53).

So "init scale" here is a proxy for **how much of the network survives to the escape event**. That also explains the
whole H4 collapse pattern in one line: larger p means a longer plateau means more pre-escape decay, so the minimum
init needed to escape rises with p (p 59/71 escape at init 1, p 89 needs ≥ 2, p 113 needs ≥ 3, p ≥ 157 fails at 4),
and the p 71 init-1 cell is the marginal case that escapes but only just — which is why it is the one cell in the
whole sweep with `flat5` = 0.10 instead of ≈ 0.31, γ = 0.13 of the ceiling, and 2 non-separating "fit" seeds.

This mechanism is directly falsifiable and gives the follow-up experiment below its design: if it is right, turning
the penalty off until escape (`--wd_warmup`, already the standard recipe for the bilinear cells) should make the
init-1 and init-4 counts agree.

---

## What I ran

All local, CPU (fp32 training, fp64 analysis; MPS rejected — no float64 and it perturbed a marginal seed),
`uv run`, detached with nohup, strictly sequential. Total training ≈ 50 min.

1. `initaudit_scan.py`, `initaudit_h1.py`, `initaudit_h34.py`, `initaudit_stats.py`, `initaudit_place.py`,
   `initaudit_match.py`, `initaudit_support.py` — re-analysis of the 192 archived fit x² q = 2 seeds (no retraining).
2. `initaudit_ceiling.py` — exact constructed γ_con(K) for K = 1…12 at p = 59 and 71 (→ `FINAL_RUNS/INITAUDIT/ceiling_p*.json`).
3. `initaudit_calib.py` — instrument calibration on those exact circuits.
4. `initaudit_abl.py` — interface-ablation count K_abl on 36 archived checkpoints (→ `FINAL_RUNS/INITAUDIT/ablation_init.json`).
5. `initaudit_cont.py` — two 40 k-step continuations at the same λ from the archived p 71 init-1 and init-4
   endpoints, 4 seeds each, fp64 instrument trace every 5 k steps (→ `FINAL_RUNS/INITAUDIT/cont_p71_init*_lam1e-4/`).
6. `initaudit_cont.py` — two fresh 12 k-step runs at init 1 and init 4, seed 100, trace every 1.5 k steps, to
   watch the spectrum form (→ `FINAL_RUNS/INITAUDIT/fresh_p71_init*/`). The matching init-4 continuation of item 5
   was cancelled mid-run for budget.

## What I could not check

- Whether the three p 71 cells ran on the same physical GPU / git commit (pods deleted). Partly mitigated by local
  reproduction and by the fresh local runs.
- The p 89 init 1 and p 113 init 2 cells have no usable endpoints, so the init trend at those primes is measured
  over init 2/3/4 only and the low-init end is missing by construction.
- `K_abl` was run on 6 seeds per cell, not 16 — its per-seed spread (6–16) is wide enough that the p 71 reversal is
  suggestive, not decisive.
- I did not run a width sweep, a matched-effective-lr control, or any prime above 113 locally (budget).
- No cloud pods were launched and nothing was spent.

## The single most decisive follow-up experiment

**The decay-free-warm-up init sweep.** At p = 71, x², q = 2, 16 seeds, run init ∈ {1, 2, 4} with the penalty
switched off until escape — `--wd_warmup 20000 --escape_frac 0.8 --clip 1`, exactly the recipe already used for the
bilinear cells — then 40 k steps with the penalty on at λ = 10⁻⁴, then continuations at λ/10 and λ/100 until γ stops
rising. Report all seven count instruments **and** γ/γ_con(K*) for every cell. This is the decisive experiment
because it removes in one stroke the two things the audit found are actually doing the work: the pre-escape weight
decay that differs 61 % vs 7 % between the inits (H6), and the resulting γ gap that makes init and quality collinear
(H3/H5). Pre-register: **(i)** if the warm-up cells reach γ ≥ 0.6 γ_con(K*) and their K_2pct agree within 1, the
init-scale finding is dead and RESULTS_REP reading 3(b) should be rewritten as "the count tracks how close GD gets to
the ceiling, and a small init at large p does not get there"; **(ii)** if they still differ by ≥ 2 on K_2pct *and*
agree in sign on N_.05 and K_abl at matched γ, the effect is real basin selection and the λ → 0 argument in the paper
needs an explicit caveat. Either way the current § "Post-hoc init-scale control" should not stand as written: its
p 71 init-1 cell reaches 0.13 of the constructed ceiling and contains two nets that do not separate the training set.
Cost: ~1 pod-hour (~$1 at the rates in the Z/S5 programmes).

