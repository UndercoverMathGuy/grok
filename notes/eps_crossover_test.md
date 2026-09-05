# Eps-dial rescoring: regime at memorisation vs regime at takeover (2026-09-05)

Question (from the 09-05 strategy discussion): is the AdamW +1 an
l_inf-regime effect AT TAKEOVER (Xie & Li 2024: AdamW = KKT of an l_inf-
constrained problem), or a SEED effect set while memorisation runs in the
sign phase? Adam is per-coordinate: sign-like while |g| > eps, GD-like
below. Per run (192 runs of runs_eps + scored crossings), bins by the
E-tensor crossing t_sw vs t_mem (first train acc >= 0.999) and t_grok.
Script: nudial/eps_crossover_test.py -> notes/eps_crossover_test.json.

Result:
- t_mem = 200-300 epochs in ALL 192 runs (memorisation is over before the
  first tensor crosses eps in every cell except eps = 1e-4, where the
  crossing is at epoch 0).
- Bin C (sign regime still on at takeover): n = 0. Takeover happens in
  the GD-like regime in 168/168 runs with eps <= 1e-5, and the mode is
  K*+1 = 4 there (mean K 3.95, {3: 25, 4: 124, 5: 18}).
  => the "l_inf at takeover" reading is DEAD. The +1 is not decided at
  takeover.
- Bin A (eps = 1e-4, GD-like from epoch 0 through memorisation): mean K
  3.58, {2: 1, 3: 8, 4: 15}; vs bin B mean 3.95, one-sided permutation
  p = 0.0023. The mode is STILL 4.
- Mean K rises with the crossing epoch and saturates once the sign phase
  lasts until train CE ~ 1e-5 (crossings 0 / 400 / 600 / 900 / 1100+ ->
  mean K 3.58 / 3.79 / 3.79 / 3.92 / 4.04). So the seed is finalised
  during the descent of train CE to its floor (epochs ~200-1000), not at
  train-acc = 1 (epoch 200) and not at grok (5-13k).

Reading for the paper: the sign phase is amplitude-blind (update ~ lr per
coordinate regardless of |g|), so it hands the GD-like cleanup a
democratic seed; the cleanup projects the seed onto the plateau of the
margin ladder (r(4) = 0.94 at nu = 6) and does not evict from a near-tie.
Direction of the shading (toward denser) is what the l_inf bias predicts;
the magnitude (+1) is measured, not derived; the effect saturates rather
than dialling. Consistent with notes/eps_dial_results.md reading 2.
