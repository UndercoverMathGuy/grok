# Review: `semifinal/analysis/claim1_knockouts.py` (2026-08-03)

Audit of whether the script matches its spec (own docstring + `semifinal/README.md`
row for claim 1b + SEMIFINAL.md claim 1) and whether the code is correct.

Verification scripts used are throwaway (scratchpad); all findings below are
reproducible from `legacy/runs/` since `runs/` is empty pending v2.

## Verdict

**Code: correct.** **Two inference-level defects**, both in the write-up layer
(docstring conclusion + SEMIFINAL wording + p-values), not the computation.

## What checks out

- **Legacy capture reproduces exactly.** Recomputing `score()` on 6 legacy runs
  reproduces `results/legacy-2026-08-03/claim1_knockouts.out` to 3 decimals
  (agg 0.712/0.716/0.764/0.671/0.596 vs recorded 0.71/0.72/0.76/0.67/0.60;
  neur likewise). The recorded output is genuine.
- **Indexing is right.** `freq_indices_2d` returns flat `(i*p+j)` indices for the
  8 linear+quadratic 2D terms; `fa2[fidx.reshape(-1)].reshape(nf,8,-1).sum(1)`
  aggregates them correctly. `make_dataset` returns all p² pairs in lexicographic
  order, which is what `fft2d`'s `reshape(p,p,-1)` requires.
- **`W_E_qr` skip is right.** `v_ > d_` skips exactly the p=157 runs (vocab 158 >
  d_model 128), matching the 2 `nan` rows in the natural cohort (n=22 vs 24).
- **Scale matching is right** in both directions: Gaussian `W_E` columns have norm
  ≈1, QR columns exactly 1, so `W_E_qr` swaps the frame while holding energy flat,
  and `rand_like` on an orth-flat run restores tilt at matched per-entry std.
- **State handling is right.** `m.load_weights` is re-called inside the 3-draw loop,
  so draws never compound. Internal proof: `W_in` is scrambled *last* in `VARIANTS`,
  after W_E/W_E_qr/attn, and still returns d=+0.001, p=0.93 — no accumulated damage.
- **`auc`** is the standard Mann–Whitney form with 0-indexed ranks (`- n1(n1-1)/2`).

## Defect 1 — the `neur` → "W_in is not a carrier" conclusion is overscoped

Docstring lines 20-28 and SEMIFINAL.md lines 65-69 ("a second, per-neuron readout
that *can* see W_in settles it fairly"). The readout is `counts[k] = #neurons whose
argmax frequency is k`.

**The scoped conditional is sound.** Docstring line 22 scopes the claim to *neuron
alignments*. Positive control (re-aim 20 W_in rows at the residual-stream direction
carrying a non-committee frequency, rescaled to each row's original norm — pure
re-aim, magnitude held fixed): `counts[tgt]` moved +19/+20/+20/+20 in 4 of 6 runs,
near-perfect recovery. So the readout genuinely has power on the alignment channel.

**The unscoped conclusion is not.** `counts` is *provably, exactly* invariant to the
per-neuron **magnitude** channel:

- `b_in` is exactly 0 in every epoch-0 checkpoint (verified) and `act_type=ReLU`.
- So scaling W_in row *i* by c>0 gives `post_i → c·post_i` (ReLU positive
  homogeneity), hence `per_nk[:,i] → c²·per_nk[:,i]`, hence **argmax over k for
  neuron i is unchanged**, hence `counts` is unchanged.
- Empirically exact: planting a 3× per-neuron gain on the neurons leaning toward a
  target frequency moved `counts[tgt]` by **+0 in 6/6 runs**, while `agg` moved
  massively (rank 35→1, 44→9, 33→0).

This matters because `README.md`'s He et al. note dismisses their lottery as reading
"at noise level" — and theirs is a per-neuron **magnitude**/phase lottery, i.e.
precisely the channel this readout cannot see by construction.

Supporting: Spearman(agg, counts) = 0.67–0.81 across runs, so `neur` is substantially
a re-expression of `agg` rather than an independent probe. Also `argmax` lands on
k=1 for 15–21% of all 512 neurons in every run — a constant artifact (harmless for
paired comparisons, but it is not signal).

**Fix (wording, not code):** scope the conclusion — "no *alignment* signal in W_in;
the per-neuron *magnitude* channel is invisible to this readout by construction and
remains untested." Do not let it stand as "the fair version of 'W_in is not a carrier'".

## Defect 2 — p-values are pseudoreplicated; inconsistent with the sibling script

`README.md`: "Statistics are reported per run AND per independent init cluster
(`common.cluster_key`) — cluster-level is primary." `claim1_readout.py` complies
(imports `cluster_key`, reports both). `claim1_knockouts.py` does **not** import it;
every p-value is a per-run paired t-test.

Independent clusters in the legacy capture:

| cohort | runs | clusters | inflation |
|---|---|---|---|
| natural-normal | 24 | 9 | 2.7× |
| orth-flat | 41 | 8 | **5.1×** |
| double-flat | 16 | 16 | 1.0× (clean) |

So `orth-flat W_E_qr d=-0.127, p=1.4e-07, n=41` is really n=8 evidence — the same
pseudoreplication SEMIFINAL's own weakest-points section already confesses (p≈0.03
one-vote-per-init). Effect *directions* are unaffected; only the p-values are.
double-flat is clean, so the "nothing to kill" null is unaffected.

**Fix:** add `cluster_key` aggregation as in `claim1_readout.py:83-92`.

## Minor (mention, don't headline)

- `auc` uses `argsort(argsort(·))` = ordinal ranks, not mid-ranks, so ties are broken
  arbitrarily. Only bites `neur` (integer counts). Measured effect vs `rankdata`:
  −0.019…+0.027, mean ≈0 — noise, not bias. `agg` is tie-free.
- `tok_cache` keyed on `data_seed` is over-keyed (labels are discarded; `tokens`
  depends only on p). Harmless.
- The `attn` scramble replaces W_K/W_Q as well as W_V/W_O, so it knocks out the QK
  pattern together with the OV circuit the T_k formula names. Reported as "partial",
  which is honest, but it is not an OV-only test.
- SEMIFINAL.md line 64 says the agg readout "can't see W_in by construction". It is
  an approximate statistical invariance (a random W_in acts as a near-isotropic
  projection preserving relative per-frequency energy), not a construction-level one.
  The script's own docstring is more careful ("nearly invariant").
- `runs/` is empty, so the script currently discovers zero runs. Documented in the
  README as pending v2, not a defect.
