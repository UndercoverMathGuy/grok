# Rescope 2026-08-30: one architecture family, the OPTIMIZER as the axis

User decision: stay with ReLU models (relu4/relu6 MLP + 1L ReLU
transformer — the two nu instances we fully understand); diversify
across optimizers instead of architectures. General gamma_hat /
quad-architecture material demoted to an addendum ("for a general
architecture, gamma_hat is defined by representation cost; here is how
to measure it").

## The unified law (paper spine)

Committees land at the max-margin optimum OF THE GEOMETRY THE OPTIMIZER
PRICES DURING SELECTION:

    K_mode(opt) = argmax_K  M_K(p) / K^(s_opt)

    s_opt = (number of chain degrees the optimizer has l2-priced
             by committee-lock time) / 2

One formula; the optimizer sets one exponent. The optimizer spectrum
spans the geometry interval [sign/l_inf ... l2]:

| optimizer            | selection geometry        | s_opt    | predicted K            |
|----------------------|---------------------------|----------|------------------------|
| gradient flow (GF)   | l2 exactly (Lyu-Li THEOREM)| nu/2     | K* (asymptotic)        |
| GD + l2, finite lr   | l2, finite-time            | nu/2     | K*-1..K* (undershoot)  |
| AdamW (eps 1e-8)     | sign until eps-horizon     | nu_sel/2 | K*+1 (nu_sel≈nu−1.3)   |
| AdamW, eps large     | l2-like from epoch 0       | -> nu/2  | -> K*                  |
| AdamW, eps tiny      | sign for longer            | < nu_sel | > K*+1                 |
| Lion (pure sign)     | l_inf forever, no crossover| -> 0     | DENSE (LP: uniform)    |

Anchors already in hand: l2 statics closed 3 ways (round 8.1);
pure-sign LP = uniform-dense by unit-group symmetry (round 8.6,
computed all primes); GD full-grid cohorts sparse/hierarchical,
AdamW democratic K*+1 (round 3); eps-horizon = homeostat 18.4 nats;
sleeping-gradient support freeze; per-tensor eps-crossing order
(Q/K/E first).

## Experimental program (all p=113 primary; relu4 + relu6 MLP;
## transformer where noted; prereg before each cohort)

- E1 GD+l2 GROKKING cohort: opt="gd", frac 0.3/0.5, gd_lrs x gd_wds,
  24-48 seeds. Calls: groks (delayed gen); K hist mode in {K*-1, K*},
  NEVER K*+1; hierarchical profiles. trainer ready (train_nudial.py).
- E2 GF ladder: GD with lr 1e-3 -> 1e-5 (epochs scaled 1/lr): K -> K*
  from below, democracy ratio rises with 1/lr (Lyu-Li is asymptotic).
- E3 AdamW eps-dial (P-N15c): eps 1e-12..1e-4, mode falls ~5 -> 4 -> 3
  monotonically. THE causal one-knob test.
- E4 Lion: pure sign-descent + decoupled wd => committees much denser
  than K*+1 (LP limit; width/wd-box bounded). Strongest extrapolation.
- E5 nu_sel measured directly: checkpointed AdamW cohort; per-tensor
  eps-crossing epochs vs committee-lock epoch; transformer prediction
  nu_sel in (4.66, 4.83), MLP in (3.29, 3.84); laggard tensor id'd.
- Secondary: transformer GD robustness; p1009 (mode 5 call) later.

## Paper structure after rescope

1. l2 margin theory for ReLU mod-add: gamma_hat = M_K/K^(nu/2), K*,
   plateau law (zero fitted params) — proof-grade + 3 instruments.
2. Optimizer geometry: growth laws (GD extinction threshold vs Adam
   coherence gating), two pricing limits (l2 sparse / sign dense-LP),
   partial pricing => s_opt, eps-horizon = homeostat constant.
3. The optimizer table above, each row measured. +1 explained, not
   fitted; Q/seed/beta all retired.
4. Addendum: general gamma_hat via representation cost (quad as the
   non-example), primes sweep, forecasting layer.
