# Claims and Evidence

The frequency set is decided at initialization by each
frequency's *arrival loudness* in the MLP — the frequency energy in `W_emb` carried through the attention `OV` circuit (freq. energy ⋅ freq. boost in OV)

All numbers come from a fresh 98-run dataset (details at the bottom).

## Frequencies are predictable at init

$T_k=\sum_{h}^{} \|W_O^h \cdot W_V^h\cdot W_{emb}|_k\|^2$

- **The frequency set is predictable before training.** Matmuls in
  closed form over the raw starting weights can rank eventual winners above losers with average AUC 0.76. 
- **Both ingredients are visible separately** in untouched networks:
  freq. energy in `W_emb` alone predicts at 0.74; OV gain alone at 0.64.
- **Erase `W_emb` → OV gain keeps working.** 
  With the embedding orthogonalized (`W_emb` freq. energy equalized), the embedding probe drops (0.53 avg. AUC) while T_k — now pure OV gain — still predicts 0.62 avg. AUC
- **Erase both → down to chance** With both ingredients flattened, all
  five probes read 0.50–0.60. The networks
  still grok on schedule and form normal frequency sets with no predictability.
- **Scramble tests agree with the formula's address.** Scrambling the
  embedding at init kills the prediction (0.74 → 0.48); scrambling
  attention damages it (−0.11); scrambling the MLP input (or the MLP weights) does nothing
  (+0.004). 

## The init chooses — training dynamics don't

The same init was trained under seven training recipes (vanilla, 3× LR, 0.3× LR,
4× weight decay, 0.25× weight decay, worst-case loss, tilted loss - CVaR), and
different inits under the same recipe. 

Choice of init: natural inits not chosen as embedding energy is a clear favorite; 
fully flattened inits not chosen as they leave no readable init signal

Same-init frequency sets overlapped (Jaccard) 0.35; different-init
sets under the same recipe overlapped 0.10 -
all-stranger baseline (0.09–0.13). 

## The frequency set can be steered by hand

- **Boosting:** Boosting a doomed frequency's embedding
  energy at init: past a threshold it joins the final frequency set beyond a certain threshold (varies per seed).
- **Suppression:** Halving the strongest destined
  winner's init energy evicted it from the frequency set.
- **Geometry:** Rotating a frequency's embedding
  direction to fit the OV circuit better while keeping its energy
  numerically identical causes adoption at a threshold (varies per seed).
- **Attention directions:** Copying one run's full
  56-frequency energy profile onto another run's embedding while keeping the
  recipient's attention directions fixed left 11 recipient-unique members survived vs 1 donor-unique across 6 transplants. 

## Summary

Which circuit a grokking network learns is decided by a lottery held at
initialization: each frequency's arrival loudness at the MLP. The ticket is readable before training
(AUC 0.76, closed form); the outcome can
be steered through either embedding or OV, and erasing both ingredients leaves nothing measurable — the
networks still grok, but the outcome is no longer readable from the
init.

## Related work

- **Varma et al. (ICLR 2024); Chughtai, Chan & Nanda (ICML 2023):**
  stated the conjecture — an appendix remark ("typically whichever
  frequencies were highest at random initialisation") and a future-work
  item.
- **He, Wang, Chen & Yang (arXiv:2602.16849; 2606.02993):** prove, for
  two-layer MLPs on one-hot inputs, that per-neuron frequency
  competition is won by initial spectral magnitude × phase alignment.
- **Morwani et al. (ICLR 2024):** prove max-margin solutions use all
  frequencies at sufficient width; the framework cannot rank individual
  frequencies.
- **Kwok, Altıntaş, Raffel & Rolnick, "The Butterfly Effect: Neural
  Network Training Trajectories Are Highly Sensitive to Initial
  Conditions" (arXiv:2506.13234):** tiny early-training perturbations
  send identical inits to different solutions.

## Limitations

- **Adoption thresholds are context-dependent** and not yet predictable
  (1.10×–1.50× across bases; one archived target never launched).
- **Two of the 98 runs never consolidated** (final test CE > 5e-3, 10×
  above every other run) and are excluded by a documented rule
  (`BAD_RUNS` in `analysis/common.py`). Both are dynamics arms; no
  conclusion changes if they are kept.

## Addendum: every flattening re-rolls the lottery

The same seed under normal / flat-embedding / flat-both initialization
picks unrelated frequency sets (overlap 0.14).

Flattening does not reveal a deeper preference of the seed; it holds a
new lottery with the ingredients that remain. 

## Reproduction

Dataset: 98 runs (`runs_torch/`), cloud-trained 2026-08-05 with new
seeds and masks, analyzed on 2026-08-06 against predictions registered
before training. Two mask cells × nine recipes plus a steering suite on
two natural bases; two never-consolidated runs excluded (n=96). The
pre-v2 material is archived in `legacy/` and is cited only as
background ("archived"), never as proof.

Each claim is backed by a script in `semifinal/analysis/` that
auto-discovers every compatible run in `runs_torch/`. See
`semifinal/README.md` for the claim→script map and the pre-registered
predictions; captured outputs are in `semifinal/results/`.
