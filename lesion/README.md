# Lesion — is the decision stored redundantly, brain-style?

Adopted 2026-08-15, prereg committed before training. The transplant
cohort showed the decision is jointly held (W_E swap transfers nothing;
recipient identity increasingly swap-resistant). But a swap is the wrong
probe for REDUNDANCY: the foreign W_E carries a competing committee. The
brain paradigm is lesion-and-recovery: destroy one copy with NOISE (no
competitor) and ask whether the surviving copy regenerates the memory.

## Design

Donors: the same 8 deterministic donor runs as the transplant cohort
(META_SEED=20260815 draws, p=113, width 512, wd 3.0, frac 0.3, 3000
epochs, ckpts every 50) — retrained on the pod (checkpoints were not
retained; the runs are deterministic).

Lesion arms, for t in {150, 500}, all 8 donors, resume training 3000
epochs from the lesioned state (cold-Adam restart — established
non-destructive: self-restart controls hit J_self = 1.0):

- **rand_we**: embed.W_E replaced by a fresh Gaussian draw rescaled to
  the Frobenius norm of the replaced (trained) matrix — direction
  information destroyed, scale preserved, no competing signal. Everything
  else kept.
- **rand_rest**: every tensor EXCEPT embed.W_E replaced the same way
  (fresh draw, norm-matched to the replaced tensor); tensors whose init
  is zeros (b_in, b_out) reset to zeros. W_E kept.
- **ctrl**: self-restart at t=500 (re-establishes the gate in-cohort).

Lesion draws come from seeds drawn from random.Random(20260816) — one
per (arm, t, donor), fixed before training. 8 donors + 32 lesions + 8
controls = 48 runs, two lockstep batches.

## Metrics

J_recover = Jaccard(final committee after lesion+retrain, the donor's
own pre-lesion final committee). Null floor = shuffled-pair Jaccard
(lesioned run vs OTHER donors' committees, same mask). Anchor from the
transplant cohort: fresh-lottery hybrids at t=0 gave median max-J 0.21.

## Pre-registered predictions

- **Gate:** ctrl median J_self >= 0.8 (expect 1.0). Below: cohort void.
- **P-L1 (downstream copy restores):** rand_we at t=500: median
  J_recover >= 0.5 and above the shuffled null (permutation p < 0.05).
  The handoff curve says the downstream holds the majority share by 500;
  with no competing signal it should re-grow its own committee.
- **P-L2 (W_E copy restores — the genuinely open bet):** rand_rest at
  t=500: median J_recover above the shuffled null (p < 0.05). Branch,
  both outcomes reportable: recovery -> storage is two-copy redundant
  ("break either, the survivor restores it"); no recovery -> storage is
  downstream-only and W_E is a mirror, positively locating the
  controlling copy (outcome (ii) of the transplant branches).
- **P-L3 (redundancy grows over the interregnum):** within each arm,
  J_recover(t=500) >= J_recover(t=150) in median.
- **Strict redundancy verdict** (the headline if it lands): BOTH arms
  above null at t=500.

No re-rolling of seeds, times, arms, or thresholds after results.

## RESULT (2026-08-15, 8 donors + 32 lesions + 8 controls, all re-grok to acc 1.0)

Gate: ctrl median J_self = 1.0 — surgery/restart non-destructive.

| arm | t | med J_recover | null | perm p |
|---|---|---|---|---|
| rand_we (keep downstream) | 150 | 0.27 | 0.00 | 0.019 |
| rand_we (keep downstream) | 500 | 0.45 | 0.00 | <0.001 |
| rand_rest (keep W_E) | 150 | 0.50 | 0.00 | 0.001 |
| rand_rest (keep W_E) | 500 | 0.67 | 0.13 | <0.001 |

- **P-L1 FAIL by pre-set threshold** (median 0.45 < 0.5) but the
  downstream copy DOES restore far above the null (0.45 vs 0.00,
  p<0.001) — reported exactly so, no re-rolling.
- **P-L2 PASS**, and stronger than predicted: the W_E copy alone
  restores 0.67 — a fresh random network wrapped around the trained
  embedding re-selects the same committee. (Consistent with W_E-energy
  AUC 0.995 at epoch 500: the tilt has materialized and a naive
  downstream race reads it back.)
- **P-L3 PASS**: both copies strengthen 150 -> 500.
- **Strict redundancy verdict: NO by prereg letter** (P-L1's 0.5 bar
  missed by 0.05), **YES in substance**: both copies independently
  restore the identity far above chance, recovery grows over the
  interregnum, and every lesioned network re-groks to full accuracy.
  Brain-style statement: destroy either copy and the survivor
  regenerates most of the memory; the embedding is the stronger engram,
  the downstream the weaker one.
- Reconciliation with the transplant: a FOREIGN W_E (competing signal)
  suppresses the downstream copy (swap J_B 0.33) more than a NOISE W_E
  does (lesion 0.45) — interference, not absence, explained the swap's
  low numbers.

## Files

- lesion/train_lesion.py — donors (idempotent), lesioned-state
  construction, lesion+control batch. --smoke: CPU mechanics asserts
  (lesioned tensor differs + norm matched; retained tensors identical).
- lesion/analyze_lesion.py — J_recover per arm/time, shuffled null,
  verdicts; writes notes/lesion_summary.json.
