# Cold email draft — Neel Nanda

Goal: get engagement (feedback, a short call, or a signal that this is
ICLR-worthy) from one screenful. Neel reads fast and hostile-skims; lead
with the causal result in his own model organism, keep every number
load-bearing, and volunteer the retraction — self-adversarial epistemics is
the strongest credibility signal for this audience. No attachments; one
link to a repo/dossier.

Fill before sending: [repo link], [dossier/PDF link], [your one-line bio],
optionally a single figure link (the dose–response / surgery trajectory
plot once the farm lands).

---

**Subject options** (pick one, all < 60 chars):
1. Which frequencies a grokking transformer picks — and why
2. Causal account of committee selection in your mod-add setup
3. Init surgery flips Fourier committees in grokked mod-add

---

Hi Neel,

Your Progress Measures paper mapped *what* algorithm a grokked mod-add
transformer learns; I've been chasing the question it left open: *which*
of the ~56 equivalent frequencies it picks. Working in your exact setup
(1L transformer, p=113, frac_train 0.3, AdamW wd=1.0; MLX port of your
codebase), I have what I believe is the first intervention-grade answer:

- **Committee identity is set by the embedding's random draw at init.**
  Component knockout: the epoch-0 committee readout (AUC 0.70 across 24
  runs) collapses to chance when W_E is randomized and is untouched by
  randomizing any other component. Surgery: scaling one frequency's W_E
  energy ×1.2 at init — inside the natural chi-square noise range — flips
  it into the final committee; ×2.25 makes it dominant; ×0.5 on the
  strongest incumbent annihilates it (peak coeff 24,622 → 0). Logits at
  init are blind (AUC 0.56) — the lottery is in weight space, which is
  maybe why it went unnoticed. (Decomposition, from surgical-edit and
  transplant experiments: the energy tilt is a causal *lever* — concentrated
  boosts control membership — but the majority carrier of natural identity
  is the draw's geometry as transmitted through the attention pathway;
  transplanting a donor run's full energy spectrum transfers essentially
  none of its committee.)
- **The carrier is embedding energy, not per-neuron gradient alignment** —
  He et al.'s 2-layer-MLP lottery mechanism does not transfer to the
  transformer (their variable carries no committee signal here).
- **A repair operator cleans up degenerate draws.** Final committees strip
  additive relations (sum/difference pairs: 4 observed vs 18.4 expected,
  p ≈ 3e-4 cluster-corrected) that are still present at base rate in the
  amplitude leaders they replace — and we can trigger the repair on demand
  by implanting a degenerate trio at init. What *triggers* natural repair
  is open: we retracted our own margin-trigger result after adversarial
  re-analysis (survivor-floor circularity).

Everything is pre-registered where causal, adversarially re-analyzed, and
reproducible from one repo: [repo link] (claim-by-claim audit: [dossier]).

I'm drafting this for ICLR. Would you be up for a 20-minute call, or
pointers on where this most needs hardening? Happy to send the 2-page
summary.

[name / one-line bio]

---

## Reviewer-anticipation notes (for us, not the email)

- Expect "isn't this just He et al.?" → the knockout directly tests and
  rejects their variable in this architecture; also S2 (their effect is
  architecture-specific — a quadratic MLP shows zero mask favoritism).
- Expect "committee changed ⇒ anything changed it" (chaos objection) → the
  claim is NOT "perturbations change committees" but (a) readout knockout
  (no retraining in the loop) and (b) dose-controlled targeted adoption;
  bystander chaos is reported separately and honestly.
- Expect "n=24" → every load-bearing p-value is cluster-corrected across
  masks; causal claims rest on surgery, not the ensemble.
- Do NOT cite the 18.4-nat homeostat or margin-as-objective anywhere — both
  deflated in-house (loss-constancy tautology; Adam-ε floor).
