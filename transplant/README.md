# Transplant — where does the decision live between selection and use?

Adopted 2026-08-15, prereg committed before training. Selection finishes
by epoch ~25; consolidation uses the decision thousands of epochs later.
The decision must be physically stored somewhere across that interregnum.
The clock cohort said the storage is not one place: the downstream keeps
selecting over a frozen W_E, and W_E alone re-runs the race over a frozen
downstream; by epoch ~500 the winner is legible in W_E (AUC 0.995) and
evidently also carried downstream. This experiment maps the ledger by
force: swap W_E between paired runs at increasing times and see whose
committee the hybrid consolidates — a HANDOFF CURVE, not a single bet.

Known prior: a tilt-only statistic transplant at init FAILED (2-vs-8,
direction geometry dominates) — this is a FULL W_E swap, carrying the
geometry, so that failure does not predict this outcome either way.
The init ticket is W_E-resident by knockout, so the t=0 swap should
follow the donor.

## Design

Mainline p=113, width 512, wd 3.0, frac 0.3 (grok ~1.5-2.5k), seeds =
the degrok draws (META_SEED=20260815: 2 masks x 4 inits).

- **Donors:** the 8 (mask, init) runs trained fresh for 3000 epochs with
  checkpoints every 50 (the degrok cohort's ckpt cadence is too coarse).
- **Hybrids:** within each mask, ordered init pairs (i1,i2),(i2,i1),
  (i3,i4),(i4,i3). For swap time t in {0, 50, 150, 500}: take recipient
  B's checkpoint at t, replace embed.W_E with donor A's checkpoint-at-t
  W_E, train 3000 epochs from that state. 8 ordered pairs x 4 times =
  32 hybrids.
- **Controls (self-restart):** same procedure with A=B (own W_E) at
  t in {150, 500} for all 8 runs = 16 controls. Isolates the restart
  shock (Adam moments are NOT transplanted — a warm-start resets them to
  zero for hybrids and controls alike, so the control carries the same
  shock).

One donor batch (8) + one hybrid/control batch (48). Driver builds the
hybrid init files from the donor checkpoints on the pod.

## Metrics

Per hybrid: final committee; J_A = Jaccard vs donor's final committee,
J_B = vs recipient's final committee; "adopts donor" = J_A > J_B.
Per control: J_self vs own final committee.

## Pre-registered predictions

- **Gate (controls):** median J_self >= 0.8 at both times. If the restart
  procedure alone scrambles the committee, hybrids are uninterpretable
  and the experiment is declared void (redesign, do not interpret).
- **P-X1 (t=0):** hybrids adopt the DONOR committee (median J_A > median
  J_B, and J_A > J_B in >= 6/8 pairs) — the init ticket rides W_E.
- **P-X2 (decision survives transplant):** at t=500, median
  max(J_A, J_B) >= 0.6 — the swap picks one parent's committee rather
  than destroying the decision. **KILL for the ledger framing** if median
  max(J_A, J_B) < 0.3 (swap re-runs the race fresh; decision is not
  portable state).
- **P-X3 (handoff curve — branch, all outcomes reportable):**
  (i) J_A rises with t toward donor dominance -> the controlling copy is
  in W_E; (ii) J_B rises with t -> the controlling copy is downstream and
  W_E is a replaceable mirror; (iii) donor-dominant at t=0 but mixed
  later -> the decision is jointly held; the crossover epoch is the
  handoff time. The curve itself is the result; no re-rolling.

No re-rolling of seeds, times, pairings, or thresholds after results.

## RESULT (2026-08-15, 8 donors + 32 hybrids + 16 controls)

- **GATE OK, emphatically:** self-restart controls reproduce their own
  committee EXACTLY (median J_self 1.0 at t=150 and t=500). The surgery/
  restart procedure is non-destructive; hybrids are interpretable, and
  committees are perfectly stable under cold-Adam restart.
- **P-X1 FAIL:** at t=0, hybrids adopt NEITHER parent (med J_A 0.13,
  J_B 0.07, med max 0.21; A-wins 4/8). Swapping W_E at init does NOT
  transfer the donor's committee — a mixed init is a fresh lottery draw.
  Combined with the knockout result, W_E is NECESSARY for the identity
  but not SUFFICIENT: the ticket is not a token in W_E, it is the
  W_E-downstream ALIGNMENT. (Consistent with the earlier tilt-transplant
  failure and with the clock cohort's bilinear-race picture.)
- **P-X2 GRAY (0.5, between the 0.3 kill and 0.6 pass):** at t=500 the
  swap half-preserves a parent's decision.
- **P-X3 (the curve, the actual result):** med J_B (recipient) grows
  0.07 -> 0.20 -> 0.33 -> 0.33 across t = 0/50/150/500 while J_A (donor,
  the W_E side) stays flat ~0.14; B wins 5-6/8 at t >= 150. The
  controlling share of the decision migrates into the recipient's
  NON-W_E weights as training proceeds: by t=500 the recipient partially
  keeps its committee despite a wholesale foreign W_E. Outcome (ii/iii)
  of the pre-declared branches: the decision is jointly held, with the
  downstream share growing over time; it is never a portable W_E token.

Reading for the paper: the transplant closes the loop with the clock
cohort — selection's product is an alignment between the two factors,
physically redundant and increasingly downstream-weighted, which is why
it survives substrate freezes (either factor can re-express it) but not
factor swaps (alignment is destroyed by mixing parents).

## Files

- transplant/train_transplant.py — donors, hybrid construction, hybrids
  (idempotent; --smoke does a CPU mechanics pass incl. surgery asserts).
- transplant/analyze_transplant.py — verdicts, handoff curve, gate.
