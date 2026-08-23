# Committee IDENTITY from t=0: unified predictor + residual (2026-08-22)

Data: 434 grokked add runs with epoch-0 checkpoints (runs_dial 148, runs_degrok2
210, runs_degrok 76), 24,242 (run, frequency) rows. Label = membership in the
final committee (committee_from_coeffs). Script: scratch identity2.py.
Tree = HistGradientBoosting, 5-fold GroupKFold by run. Jaccard = per-run
top-K (K = actual size) vs actual committee. Random K-set baseline 0.05.

## Selection rules (no learning)

| rule | Jaccard |
|---|---|
| top-K by init W_E Fourier energy | 0.132 |
| same + additive-collision veto | 0.133 |
| top-K by mask popularity | 0.038 |
| top-K by logit spectrum @ snapshot 5 (+veto) | 0.55 |

**The veto law is not a selection factor** (+0.001): at K~4 only ~14% of
random sets collide, so the constraint rarely bites at selection; it is an
eviction-time law.

## Tree, cumulative feature sets (AUC / top-K Jaccard)

| features | AUC | Jaccard | dial / degrok2 / degrok |
|---|---|---|---|
| W_E energy (z, rank) | 0.811 | 0.230 | .26/.22/.20 |
| + W_U alignment, W_in gain, attention-OV gain on the k-direction | 0.885 | 0.306 | .33/.30/.29 |
| + mask popularity, e0*pop | 0.905 | 0.407 | .54/.33/.35 |
| + epoch-0 logit spectrum | 0.908 | 0.408 | |
| + k, width, wd, frac, K | **0.912** | **0.417** | .54/.35/.36 |
| + logit spectrum @ epoch 100 | 0.980 | 0.709 | |
| + logit spectrum @ epoch 500 | 0.991 | 0.828 | .75/.87/.87 |

t=0 model: top-1 predicted frequency is in the committee 73% of runs (chance 7%).

Permutation importance (AUC drop), t=0 model: **u0_z 0.076 (W_U alignment),
gov 0.071 (OV path gain)**, L0 0.033, e0*pop 0.031, e0_z 0.029, pop_z 0.027,
gin 0.022, e0_rank 0.021, kfrac 0.017, K 0.017, govin 0.015, frac 0.009,
wd/d_mlp ~0. NEW FACTORS vs prior work: the readout side (W_U) and the
attention OV path gain on each frequency's W_E direction matter as much as
the W_E tilt itself — the lottery ticket is a PATH, not an embedding.

## Residual structure (per-run Jaccard of the t=0 model)

- by width: 128 .27, 256 .49, 512 .38, 1024 .50, 2048 .48, 4096 .71
- by wd: 0.1 .64, 0.3 .55, 1.0 .43, 3.0 .37
- by frac: 0.2 .13, 0.25 .24, 0.3 .45, 0.5 .39, 0.7 .48, 0.9 .40
- by K: flat (~.40-.44)
Predictability from init rises with width and falls with wd and data
scarcity: the unexplained part is the chaotic regime (sub-threshold init
swaps reshuffle winners), not a missing t=0 feature.

## Does the margin's sidelobe term shape identity?

Members' percentile rank among all candidates by "sidelobe completion"
(M(C\{c}+x) maximised over x): 0.28 vs 0.49 for a random candidate — final
committees ARE sidelobe-compatible, equally for well- and poorly-seeded
members (0.282 / 0.289). But adding a sidelobe term to the greedy selector
does nothing (alpha sweep: 0.417 -> 0.422 max). Compatibility is enforced
during dynamics (eviction among near-equal candidates) and the identity of
the survivor is chaotic; it is not a usable t=0 selector.

## Bottom line for the unified formula

K and the set-level constraints come from the margin optimum (nudial);
identity = path-readiness at init (W_E x OV x W_in x W_U on the k-direction,
times mask popularity) with a predictability ceiling ~0.42 Jaccard /
AUC 0.91 at t=0 that rises to 0.71 by epoch 100 and 0.83 by epoch 500.
The remaining factor is not a feature; it is sensitivity to init (chaos),
strongest at low width / high wd / low frac.
