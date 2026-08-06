# Brief: mid-training frequency implant / eviction experiment (seed1)

## Context

Setup: 1-layer transformer (d_model 128, 4 heads, d_mlp 512, ReLU, no LayerNorm)
on (a+b) mod 113, full-batch AdamW lr 1e-3, wd 1.0, frac_train 0.3, f64 CE.
Grokked models solve the task with a sparse set of ~3-4 "key frequencies" k,
each contributing cos(w_k(a+b-c)) to the logits. Per-frequency strength is
measured as the coefficient of the normalized cos(w(a+b-c)) template in the
full-grid logits ("phase-locked coeff"); code in `grok/metrics.py`
(`freq_coeffs_and_energy`). Training-time trajectories of all 56 frequencies
are recorded every 50 epochs to `spectra.npz` per run (spectral logger in
`grok/train.py`).

Larger question under investigation: what determines which frequencies are
selected? Competing hypotheses: (a) init lottery (He et al., arXiv:2602.16849,
prove per-neuron init-determined selection for 2-layer MLPs in a decoupled
regime); (b) selection during early training / rich-get-richer; (c) late,
revisable selection at grokking-time consolidation.

## Prior observation motivating the experiment (loser census)

Across 7 independently-seeded runs (runs/zoo/seed*, spectra.npz each, shared
data split data_seed=0), the audition phase is broad: ~8 frequencies grow well
above background during memorization. Final committees (3-4 freqs) do not
respect audition ranking, and mass evictions occur just AFTER grokking. Extreme
case, run `runs/zoo/seed1` (final committee {7,54,30,4}, grok epoch ~10100):
f5 was rank #2 through all of memorization, peaked at coeff 2484 (~150,000x
background) at epoch 12,200 — after grokking — then was crushed to 0.4; f15
(rank #3) peaked 1647 then went to 0. Meanwhile f54, ranked #6 at epoch 3000,
made the final committee. So "once selected, grow forever" is already false in
natural runs.

## Experiment

Question: is committee membership causally revisable mid-training? Can an
outsider frequency implanted at amplitude parity with a destined winner join
or displace, and do destined evictions still occur?

Base state: `runs/zoo/seed1/checkpoints/epoch_03000.safetensors` (mid-audition;
top coeffs at e3000: 30:814, 5:703, 15:545, 7:514, 4:455, 54:390, 50:335,
42:284; background median ~95).

Implant frequency: f36, verified to be a clean outsider for this run — not in
the audition, not a harmonic (fold(2k), fold(3k)) of any auditioner, not a
pairwise sum/difference of auditioners mod 113.

Surgery: MSE distillation of the checkpoint toward (its own full-grid logits +
lambda * normalized cos(w_36(a+b-c)) template), train rows only, wd=0 during
surgery, lambda=800. 12,000 distill epochs reached coeff 345 (43% of target;
growth quasi-linear, no autocatalytic takeoff). 345 is amplitude parity with
destined-winner f54 (389) and the implant is fully phase-locked by
construction. Script: scratchpad `implant_arm.py`.

Arms (each then resumed with plain CE + wd for 17,000 epochs, fresh Adam,
spectra logged; dirs under `runs/eviction/`):
1. control — resume from raw e3000 checkpoint.
2. implant36 — resume from surgically implanted checkpoint.
3. sham — resume from a checkpoint that underwent the identical 12k-epoch
   distill toward its own UNMODIFIED logits (controls for surgery side effects).

The original seed1 run itself is the no-resume reference.

## Results

| arm      | final committee | grok epoch (orig coords) | f5/f15 fate            | f36 fate  |
|----------|-----------------|--------------------------|------------------------|-----------|
| original | {7,54,30,4}     | 10100                    | purged post-grok       | —         |
| control  | {7,54,30,4}     | 10100                    | peak 6827/3911 -> ~0   | —         |
| sham     | {7,54,30,4}     | 10150                    | peak 6857/3787 -> ~0   | —         |
| implant36| {7,54,30,4}     | 10050                    | peak 6751/3729 -> ~0   | 345 -> 0  |

1. Determinism/robustness: control reproduces the original run essentially
   exactly (same committee, same grok epoch, f5 peaks at the same epoch ~9700
   at the same amplitude). Fresh optimizer and 12k epochs of sham distillation
   change nothing measurable (<~2% trajectory differences).
2. The implant decayed from the first CE epoch (345 -> 201 by +1000 epochs ->
   136 -> 46 -> 0 by ~epoch 11000 orig coords). Decay rate ~x0.58 per 1000
   epochs vs pure-wd x0.37, i.e. it received gradient support but permanently
   below break-even, while f54 at the same amplitude was above break-even and
   grew ~25x. The implant's presence did not perturb any other frequency's
   trajectory, the committee, or grok time.
3. Neuron census at e3000 (threshold 0.85, also 0.5): NO frequency has
   committed neurons yet — destined winners included (all diffuse, IPR > 100
   over 512 neurons, per-freq MLP energy shares 1-4%). The implant added logit
   amplitude with little MLP restructuring (f36 MLP energy share 0.73% ->
   0.85%). Notably destined-loser f5's MLP share (3.5%) exceeds destined-winner
   f54's (2.6%).

## Conclusions (as currently drawn — please attack)

- C1. Network-level frequency selection is NOT an init lottery: outcome is
  fully determined by mid-training state, and is robust to significant
  perturbation of that state.
- C2. Selection is NOT early-amplitude / rich-get-richer: audition rank is
  violated in both directions (rank #2/#3 die after growing for 7000 more
  epochs; rank #6 wins), and a phase-locked outsider at winner-parity amplitude
  is rejected on contact.
- C3. "Once selected, grow forever" is false: large-amplitude frequencies are
  evicted, with evictions time-locked to post-grokking consolidation, and this
  purge replays deterministically from epoch-3000 state.
- C4. The selection variable is latent in the weights by epoch 3000 but is not
  readable from any per-frequency observable tried so far (logit coeff, growth
  rate, phase-lock, MLP energy share, neuron commitment — all fail to separate
  winners from losers). Working hypothesis: the criterion is collective
  (which subset of frequencies forms the most margin-per-weight-norm-efficient
  committee, possibly interacting with the fixed train mask), decided/executed
  at consolidation.
- C5. Circuits cannot be installed at the logit level: 12k epochs of forced
  distillation bought 43% of target amplitude with quasi-linear growth, and CE
  training dissolved it immediately. Organic growth produces something
  (a weight-space realization property) that forcing does not.

## Known caveats / review targets

- Single base run, single implant epoch (3000), single frequency (36), single
  dose (345). Rejection-on-contact could be dose-limited or epoch-limited
  (e3000 may be past commitment time; try e500-e1000) or an artifact of the
  distilled realization being norm-inefficient (the census suggests the
  implant is "logit paint" rather than a real circuit — a grafted or
  organically grown f36 circuit might behave differently).
- The implant reached only 43% of intended dose; parity argument rests on
  comparison with f54 (389) not the audition leader (814).
- Committee-detection uses a largest-log-gap heuristic on sorted |coeff|.
- All farmed runs share data_seed=0 (fixed train/test split); across 14 runs
  freq 7 appears in 7 committees, 14 in 6 — network-level selection may be
  substantially mask-determined; not yet tested with varied splits.
- The deterministic replay means "decided by e3000" in the trivial dynamical
  sense; the nontrivial claims are the robustness to perturbation and the
  failure of all local observables to predict.

## Proposed next probes

1. Dose ladder (lambda -> 2000, 5000): is outsider rejection absolute?
2. Implant earlier (epoch 500/1000) to bracket the commitment time.
3. Predictor hunt (analysis only): compute committee-level scores at e3000
   (pairwise template interference on the train mask, margin-per-norm of each
   4-subset of auditioners) and test retrodiction of final committees across
   all 7 farmed runs.
4. Vary data_seed in the seed farm to test mask-determinism of selection.

## Data locations

- Farmed runs + spectra: `runs/zoo/seed*/spectra.npz` (7 runs with logger)
- Experiment arms: `runs/eviction/seed1_e3000_{control,implant36,sham}/`
- Surgical checkpoints: `runs/eviction/implanted_f36_l800.safetensors`, `runs/eviction/sham.safetensors`
- Analysis scripts (session scratchpad): `analyze_farm.py`, `loser_census.py`,
  `implant_arm.py`, `eviction_results.py`
