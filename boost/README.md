# Boost/Jump — the content-level double dissociation

Adopted 2026-08-15, prereg committed before training. Timing evidence
cannot prove selection and learning are distinct (any one-process story
fits some timing pattern). The proof standard here is interventional and
content-level: one scalpel that changes WHICH solution wins while the
learning trajectory stays inside a pre-registered equivalence band, and
one scalpel that moves the learning trajectory while the SAME solution
wins. Weight decay is BANNED as a lever (user veto: touches every
parameter every step — saw, not scalpel); it is fixed background (3.0).

## Levers

- **I_S (boost, epoch 20, mid-race):** multiply the target frequency's
  components of W_E by dose alpha: W_E += (alpha-1) * (W_E B_k) where
  B_k projects the vocab axis onto span(cos_k, sin_k). Touches one
  frequency's subspace of one matrix, nothing else.
  Target rule (pre-registered): the highest-energy frequency at epoch
  20 that is NOT in the donor's natural final committee (donors are
  deterministic, so the natural outcome is known).
  Sham control: identical boost applied to the donor's natural TOP
  committee frequency (same magnitude of intervention, no content
  change expected).
- **I_L (data jump, epoch 200, post-lock):** resume with frac_train
  0.3 -> 0.7, same data seed. The seeded-shuffle-prefix split makes the
  new training set an exact SUPERSET of the old — no example changes
  side; data is only revealed. No parameter, optimizer, or
  hyperparameter is touched. (Data quantity is the consolidation dial
  certified by the preregistered de-grok result: 36x on t_gen.)
  Gate control: self-restart at 200 with unchanged data.

## Design

Donors: the 8 deterministic mod-add donors (META_SEED=20260815 draws),
p=113, width 512, wd 3.0, frac 0.3, 3000 epochs, ckpt every 10.
Arms (stage-2 runs resume from edited/plain epoch-t checkpoints, cold
Adam — gate J_self=1.0 established twice):

- boost alpha in {2, 4, 8} x 8 donors = 24
- sham alpha in {4, 8} x 8 donors = 16
- jump (frac 0.7 at t=200) x 8 = 8
- jump-gate (self-restart t=200, frac 0.3) x 8 = 8

Stage-2 epochs 3000; absolute times = stage-1 offset + resumed epoch.
t_gen for jump arms is measured on the post-jump test complement
(smaller test set; threshold acc >= 0.99 unchanged — noted, not hidden).

## Pre-registered predictions

- **P-B1 (S is steerable mid-race):** target enters the final committee
  at a rate rising with alpha; at alpha=8, >= 6/8 flips.
  KILL: <= 1/8 flips at alpha=8 — the racing variable is not
  manipulable mid-race and the dissociation program fails on the S side.
- **P-B2 (learning blind to the S-intervention):** paired
  |t_gen(boost) - t_gen(donor)| / t_gen(donor) median <= 0.15, and not
  worse than sham's same statistic by more than 0.05. Equivalence band
  declared here, before data.
- **P-B3 (L moves under the data jump):** paired t_gen(jump, absolute)
  <= 0.5 x t_gen(donor). (De-grok predicts ~0.2.)
- **P-B4 (S blind to the L-intervention):** median J(final committee
  after jump, donor natural committee) >= 0.8; jump-gate must hold
  J_self >= 0.8 for the arm to be interpretable. If instead the
  committee changes, lock-in is data-conditional — reported as a
  first-class finding, not a failure to hide.

Verdict rule: P-B1 + P-B2 + P-B3 + P-B4 all passing = the content-level
double dissociation stands and becomes the paper's central figure. Any
single failure is reported with its branch meaning above. No re-rolling
of seeds, doses, times, target rules, or bands after results.

## RESULT round 1 (2026-08-15) — both scalpels caught by their controls

- P-B1 nominally PASS (8/8 flips at all doses) but the sham exposed it:
  at alpha >= 4 boosting even the run's OWN winner destroys grokking
  (no grok in 3000 ep, committee scrambled to J ~0.25) — multiplicative
  boost = norm injection = a saw. Flips at alpha >= 4 are wreckage, not
  steering. alpha=2 genuinely steers (8/8, but t_gen shifts ~36% ->
  **P-B2 FAIL**).
- **P-B3 PASS**: data jump moves learning 5.7x (t_gen ratio 0.176),
  gate perfectly clean (J_self = 1.0, dev 0.14).
- **P-B4 FAIL -> pre-declared branch**: J(jump) = 0.71 vs gate 1.0 —
  revealing new data at t=200 PARTIALLY RE-OPENS selection. The
  training set is a selection input (consistent with mask-popularity);
  data is a shared input like init, not a pure L-lever at t=200.

## Round 2 (declared 2026-08-15 before its runs; same donors)

Fixes implied by the round-1 control failures:

- **I_S v2 — subspace SWAP (isometry):** exchange the coefficient
  contents of the leader and target frequency subspaces of W_E at
  epoch 20 (right-multiplication by an orthogonal matrix; Frobenius
  norm exactly preserved; nothing outside the two subspaces touched).
  Sham: swap two frequencies that are both outside the committee.
- **I_L v2 — LATE data jump:** frac 0.3 -> 0.7 at t in {500, 1000}
  (plus gate at 1000), measuring lock hardening as J vs jump time.

Predictions:
- **P-B5:** swap steers — target in final committee >= 6/8.
- **P-B6:** learning blind to the swap — paired |dt_gen|/t_gen median
  <= 0.15 and <= sham + 0.05. (The equivalence claim, now with the norm
  confound removed by construction.)
- **P-B7 (lock hardening):** J(jump@t) rises with t; scored:
  median J(jump@1000) >= 0.8 with gate@1000 >= 0.8.
- **P-B8:** late jump still moves learning — t_gen(jump@1000, absolute)
  <= 0.7 x donor.

P-B5 + P-B6 = the S-side dissociation; P-B7 + P-B8 = the L-side at a
hardened lock. All four = the content-level double dissociation stands,
with the round-1 failures documented as the path to the working levers.

## RESULT round 2 (2026-08-15)

- **P-B5 FAIL, P-B6 FAIL — and informative:** the isometric swap
  demotes the leader in 8/8 runs but the target inherits the win in
  0/8; committees reorganize (J 0.34) and t_gen shifts +21%. The
  sham (swapping two IRRELEVANT frequencies, norm-preserving) still
  scrambles committees to J 0.50. Conclusion: mid-race, ANY W_E
  subspace surgery — even provably norm-neutral, even content-
  irrelevant — partially re-runs selection. Amplitude in W_E is not
  the selection state; the state is the W_E-downstream alignment
  (third independent confirmation after transplant and lesion), and it
  is chaotic to surgery mid-race. Mid-flight S-levers do not exist in
  W_E space.
- **P-B7 PASS (lock hardening):** J(jump@t) = 0.71 / 0.71 / 0.875 at
  t = 200/500/1000, gates clean (J_self = 1.0). The lock hardens over
  the interregnum — matching the engram write-then-replicate timeline.
- **P-B8 PASS:** the late (t=1000) data reveal still accelerates
  learning (t_gen ratio 0.68) with the hardened committee preserved.

## The dissociation that actually holds (analysis of runs_compiler, 96 runs)

**RETRACTED 2026-08-16 and rebuilt — see CLAIMS.md C3.** The original
statistic here ("identity-within-dose permutation p = 0.93 over 12
committees") was vacuous: Phase B redrew its target set inside the
innermost loop, so all 96 sets are distinct and the `(K, set_i)` identity
label maps to 8 different frequency sets across the dose x base cells.
The permuted factor carried no content. The number is struck.

Rebuilt on the same runs by `compiler/analyze_sside.py`: controlling dose
AND base (R2 = 0.595; dose alone 0.456), nine genuinely-varying content
contrasts — six content coordinates, a Mantel content-similarity test, and
two install-realisation contrasts — are all equivalent to zero at a
pre-registered 15% band (largest swing 46 ep vs a 60 ep band; MDE 12%).
Compiled runs (Phase B) grok at 200-1100 epochs (median 400) vs
5100-13400 natural for the same bases, so the compiler is NOT
learning-neutral and the result is scope-limited to that accelerated
regime on a QR-flat substrate with 2 bases off one data mask. Phase C
(200 arms, natural substrate, content fully crossed with dose and base)
is the replication on good runs.

Final form of the claim, replacing "two independent clocks":
**the CONTENT of selection and the TRAJECTORY of learning are mutually
causally inert; they couple only through a scalar STRENGTH channel.**
- Content -> learning: nil (compiler, 12 committees x fixed dose,
  p = 0.93).
- Learning -> content: nil at hardened lock (late data reveal, 1.5x
  speedup, J 0.875, gate 1.0).
- Strength (tilt magnitude / dose / init luck) is a shared scalar
  input to both — the source of every coupling observed (per-run
  correlation, compiler acceleration, boost destruction, S5 wd
  coupling candidate).

## Files

- boost/train_boost.py — donors, checkpoint surgery, stage-2 batches
  (--smoke asserts the boost edits exactly one frequency subspace and
  the jump mask is a strict superset).
- boost/analyze_boost.py — verdicts; writes notes/boost_summary.json.
