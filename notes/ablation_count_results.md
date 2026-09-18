# Interface-ablation frequency count: definition and validation (2026-09-09)

Why. The count used everywhere so far (nudial/count_definitions.py) is a
threshold on the Fourier amplitude of the LOGITS: DFT the clean a+b-c table,
sort, cut at a 10% relative amplitude or at the largest log-gap. That is a
magnitude gap in an output projection, not a measurement of what the
network uses, and the p = 1009 relu-4 cohort (clean fraction 0.48-0.72,
harmonic sidelobes at 2-9%) is exactly where such a cut could break. This
note defines a count by SURGICAL REMOVAL and validates it on every natural
p = 113 run we hold, plus the theory's constructed circuits where the count
is known exactly.

## Definition (nudial/ablation_count.py)

Architecture-agnostic: every model on a Z_p token task has two token-indexed
interfaces, an input table and an output (logit) table. Nothing else is
assumed about the model.

1. Remove frequency k: project cos(2 pi k x/p), sin(2 pi k x/p) out of the
   token axis of every input table and every output table. Run the model's
   own forward pass on the full p^2 grid. Nothing inside the box is touched.
2. drop_k = 1 - minmargin(cut) / minmargin(full), raw logit min margin, with
   three cuts: input only, output only, both. Mean-margin and 1st-percentile
   drops recorded alongside.
3. Null floor: cut each of 32 random frequencies outside the top 32 by
   interface energy; the floor is the largest drop they produce (per cut
   type). Set by the run itself in margin units; no p-dependence.
4. A frequency COUNTS if (a) its both-sides drop clears the floor, (b) it is
   read AND written: input-only and output-only cuts each clear their own
   floor, and (c) it is not a harmonic corrector: a significant k that is a
   small multiple n f (n <= 5) or n1 f1 + n2 f2 (|n| <= 2) of an already
   counted fundamental AND carries < 25% of that fundamental's interface
   energy is folded into it. Fold runs in interface-energy order so a
   fundamental is always taken before its harmonics.
5. Reported per run: the full sorted drop profile, the null floor, the count
   K_abl, the unfolded count K_abl_raw, the read-and-write count K_rw, the
   folded classes, and the legacy instruments recomputed on the same forward
   pass (K_thr10, K_thr05, K_gap, K_load90, k_eff, clean_frac).

Rule (b) exists because trained transformers carry low-energy input
components at 2k, 3k of each fundamental (0.5-4% of interface energy) whose
removal costs 10-37% of the min margin: they cancel relu sidelobes. They are
read but not written. Output sidelobes (2k written by k-neurons) are written
but not read. Features do both. Rule (c) catches the same objects when they
are read-and-written weakly; on the natural AdamW cohorts (b) and (c) never
remove a fundamental.

## Validation on natural runs (calibration set: AdamW naturals only)

Comparison script nudial/ablation_compare.py; statistic = min margin, null
safety factor 1. "=thr10" is exact agreement of K_abl with K_thr10; "setJ" is
the Jaccard overlap of the counted frequency SET with the legacy top-K set.

| cohort | n | K_abl histogram | K_thr10 histogram | =thr10 | within 1 | =K_gap | =K_load90 | setJ | raw - K_abl | null floor (med) |
|---|---|---|---|---|---|---|---|---|---|---|
| transformer p=113 natural (runs_torch/p-113) | 8 | {3:2, 4:2, 5:3, 6:1} | {3:2, 4:2, 5:3, 6:1} | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3.1 | 0.118 |
| MLP relu-4 AdamW (M, frac 0.5) | 32 | {4:2, 5:15, 6:12, 7:3} | {4:2, 5:16, 6:11, 7:3} | 0.97 | 1.00 | 0.97 | 0.97 | 0.99 | 1.3 | 0.004 |
| MLP relu-6 AdamW (M, frac 0.5) | 32 | {3:5, 4:18, 5:9} | {3:5, 4:19, 5:8} | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 | 2.5 | 0.004 |
| constructed circuits p=113, K=1..7 | 7 | = K | = K | exact 7/7 vs true K, set J 1.00 | | | | | 0 | 1e-6 |
| constructed circuits p=1009, K=1..9 | 9 | = K | = K | exact 9/9 vs true K, set J 1.00 | | | | | 0 | 1e-6 |

The two AdamW disagreements, both resolved in the ablation's favour:
- relu-4 wd1_s218609: a 6th frequency at 1.3% of the top amplitude, read and
  written, removal costs 1.1% of the min margin vs a 0.7% floor. Real but
  tiny; the 10% cut calls it 5.
- relu-6 wd1_s233006: frequency 6 at 9.3% of the top amplitude, removal costs
  50% of the min margin. K_gap and K_load90 also say 5; only the 10% cut
  says 4.

On the AdamW naturals the three margin statistics (min, 1st percentile,
mean) give identical counts under the rule above.

## Reported, NOT used for calibration: GD-trained cells (MG)

| cohort | n | K_abl | K_thr10 | =thr10 | within 1 | clean_frac med | null floor med |
|---|---|---|---|---|---|---|---|
| relu-4 GD (wd 3e-4, lr 0.3) | 32 | {2:3, 3:17, 4:11, 5:1} | {2:9, 3:19, 4:4} | 0.59 | 0.94 | 0.605 | 0.144 |
| relu-6 GD | 26 | {2:13, 3:8, 4:5} | {1:1, 2:15, 3:9, 4:1} | 0.81 | 0.92 | 0.985 | 0.108 |

These nets are half clean (relu-4 GD clean fraction 0.47-0.92, the same
regime as p = 1009) and the legacy instruments disagree with EACH OTHER on
them (K_gap 1 / K_thr10 2 / K_load90 7 on the same run), so there is no
reference count. The ablation profile shows why: the min margin depends more
on the 2nd-5th harmonics of the top frequency than on the top frequency
itself (drops 3-9x vs 2.4x on wd0.0003_lr0.3_s233006), and load-bearing 6th
and 7th harmonics sit beyond the fold range. Nothing was tuned on these
cells (user instruction 09-09). They are the preview of what p = 1009 will
look like under any instrument.

Cells without final checkpoints locally (A quad-3, G frac 0.3/0.7) could not
be scored.

## Files
- nudial/ablation_count.py    instrument (MLP, transformer, constructed adapters)
- nudial/ablation_compare.py  agreement tables; recomputes the count from the
                              stored per-frequency drops (stat / safety / fold
                              parameters can be changed without re-running)
- notes/ablation_count_mlp_p113.json, ablation_count_tf_p113.json,
  ablation_count_construct.json   per-run records incl. full drop profiles
