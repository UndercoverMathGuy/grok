# SEMIFINAL — the claims, from scratch, in plain language

Status: 2026-08-03. This is the plain-language master summary of what we
claim, how strong each claim is, and how it was proven. Technical
receipts: the captured analysis outputs in `semifinal/results/`.
(Everything that predates the v2 dataset — old runs and the exploratory
lab notebook — is archived under `legacy/` and is not load-bearing.)

**Reproduction:** every claim below is backed by an analysis script in
`semifinal/analysis/` that auto-discovers all compatible runs under
`runs/`, and every underlying run has a cleaned training script in
`semifinal/training/` — see `semifinal/README.md` for the claim→script
map, expected numbers, and the 2026-08-03 captured outputs in
`semifinal/results/`.

## The setup

A small transformer learns to add two numbers mod 113. Nanda et al. showed
*how* the trained network does it: it builds an internal algorithm out of
sine waves, using a handful of "frequencies" — typically 3–5 out of 56
possible ones. Any handful would work equally well. So there's a question
their paper left open: **out of 56 interchangeable options, what decides
which 3–5 this particular network ends up using?** We call the final set
the *committee*. That question — why these frequencies and not others — is
what all of this is about.

## The hypothesis

**The choice is made by tiny random accidents in the starting weights, and
we can say exactly which accidents.** When the network is initialized,
every frequency's signal has to travel from the input tokens, through the
attention block, to the part of the network that computes. Along the way,
each frequency arrives with a slightly different loudness — purely by luck
of the random starting numbers. Our claim: that arrival loudness is the
lottery ticket. It has two independent ingredients that multiply:
(1) how big the frequency happens to be in the network's **embedding
table**, and (2) how well its particular direction happens to **fit
through the attention weights**. Frequencies that arrive louder get
amplified by training, win the early race, and become the committee —
unless the resulting set is mathematically awkward, in which case training
repairs it (that's the second half of the story).

## Claim 1: we located the lottery — and can compute the ticket with pen and paper

**Strength: the strongest thing we have. Multiple independent proofs.**

- *You can read the future at step zero.* Before a single training step, a
  simple formula on the starting weights — multiply three matrices,
  measure each frequency's arrival loudness — picks eventual winners over
  losers about 72% of the time on average (coin flip would be 50%), across
  24 natural runs spanning 3 primes and 9 train/test splits. Per-run it
  ranges from 0.49 to 0.99 — the average is strongly above chance
  (p ≈ 6e-4 counted honestly, one vote per independent mask), but on any
  single run the read can fail.
- *Each ingredient is readable on its own, in untouched networks.*
  Ingredient 1 alone (embedding size) predicts at ~0.70. Dividing it back
  out and reading only ingredient 2 (attention fit) still predicts at
  ~0.65 in the same natural runs — and the product beats either alone
  (~0.72–0.74). So the two-ingredient structure isn't inferred only from
  surgery; both factors are visible separately in ordinary runs.
- *Scramble tests are consistent with the formula's address.* Scramble the
  embedding table → the prediction dies. Scramble the MLP input → the
  aggregate readout is unaffected (as it must be — that readout can't see
  W_in by construction, so this row is a sanity check, not evidence). A
  second, per-neuron readout that *can* see W_in settles it fairly: it
  reads the committee before training, but scrambling W_in doesn't dent it
  while scrambling the embedding kills it — whatever per-neuron structure
  exists is inherited from the embedding-through-attention stream; the MLP
  contributes none of its own. Scramble attention → partial damage.
  Consistent everywhere, but the real "nothing else carries it" proof is
  the erasure experiment below, not the scrambles.
- *The two-ingredient structure was confirmed by deleting each
  ingredient.* We built special initializations where ingredient 1
  (embedding size) is *exactly equal* for all 56 frequencies. Those
  networks train totally normally — and their committees are still
  predictable (66%) from ingredient 2 alone. Honest caveat on that number:
  the 41 flattened runs recycle 8 independent starting draws (the same
  inits retrained under many recipes), so counted one-vote-per-init it is
  suggestive (p ≈ 0.03) rather than overwhelming — the v2 dataset's fresh
  orthWE cells (8 never-used inits on 2 never-used masks,
  `train_semifinal_v2.py`) are the pre-registered fix, and the in-place
  ingredient-2 readout above already shows the same thing independently. Then we built initializations where **both** ingredients
  are exactly equal. Result, over 16 runs: **no probe we own predicts the
  committee any better than chance** — yet the networks still grok on
  schedule and form normal committees. We erased precisely two things, and
  precisely the predictability vanished. Nothing else was touched, and
  nothing else picked up the slack (at the precision 16 runs buys; see
  weakest points).

One honest note: along the way, one run in the "both flat" cohort looked
like it revealed a third hidden carrier in the MLP. Sixteen runs later
that was exposed as a statistical fluke (odds it was real: worse than
200-to-1 against). We retracted it. The lottery has two readable carriers,
and below them: nothing readable.

## Claim 2: the "orthonormalize the embedding" experiment — demoted and replaced

**Strength: the original version was weak and we no longer lean on it; its
replacement is solid.**

The old argument was "flatten the embedding energies → committees change →
so the energies mattered." The problem: committees change if you sneeze at
the initialization (we proved assembly is chaotic — swapping one *dead,
irrelevant* frequency's starting value changes which committee forms). So
"it changed" proves nothing by itself. What replaced it: the
flattened-embedding runs are still *predictable* from ingredient 2 (66%),
the same flat init trained under different recipes keeps choosing similar
committees, and deleting ingredient 2 as well removes all predictability.
On the recipe diversity, the honest accounting: we ran ten recipe
*variants*, but six of them are small perturbations of one recipe (tilted
loss), so the fair count is **four genuinely different training regimes**
(plain, tilted, noisy-gradient, worst-case-focused). Counting only pairs
across those four groups, same-init overlap is 0.29 vs 0.11 for strangers
(permutation p < 1e-4) — smaller than the headline 0.35 we used to quote,
still ~3× baseline. That's the necessity argument done properly: not "it
changed," but "we can account for exactly what information decides, and
show the ledger goes to zero when we remove both entries."

## Claim 3: we can steer the committee by hand

**Strength: solid causally; the fine print got more honest overnight.**

- Take a frequency the network was going to ignore and boost its starting
  embedding energy: past a threshold it joins the final committee, and at
  2.25× it dominates — on every base we tried. On the original run the
  threshold was as small as natural luck (a 10–20% bump sufficed); on
  other runs it took the full 2.25×, and one stubborn target never
  launched (see fine print). Halve the starting energy of the *strongest*
  destined winner: it gets annihilated. The *response* (the target's peak
  amplitude) is dose-monotone everywhere we've tested.
- The cleanest experiment of the whole project: we steered the committee
  **through the other ingredient too** — rotating a frequency's
  *direction* to fit better through attention while keeping its energy
  mathematically identical (to 15 decimal places). At matched arrival
  loudness, the outcomes matched the energy experiments almost exactly —
  at the top dose, the same rival got evicted and the *identical* final
  committee formed. Same loudness, different knob, same result: it really
  is arrival loudness that matters, not either knob per se.
- The honest fine print from the overnight cross-checks: (a) the
  *threshold* dose needed for adoption varies a lot with context — 1.1× on
  the original run, 2.25× on two others, and one stubborn target never
  launched at all; "you can steer" is proven, "a fixed dose always works"
  is false. (b) You control the *target* frequency reliably; what happens
  to the *bystanders* is chaotic — real but unpredictable. Causal work now
  spans four different train/test splits, so this is no longer a one-run
  result.

## Claim 4: the safety net — training refuses broken committees

**Strength: the refusal itself is solid; the mechanism details are partly
open.**

Some committees are mathematically bad — for example, sets where one
frequency is the sum of two others, which makes the network's answers
easier to confuse. Two findings:

- *A floor.* Ranking every final committee against random alternatives by
  a quality score: across all 122 selection runs — including every weird
  initialization we built — **121 clear the bottom quarter**, and the one
  exception proves the rule: it's a run where we *forced* a frequency into
  the committee with a huge 2.25× implant, pinning a set (just barely,
  22.9th percentile) that free training never chooses. Left to itself,
  training never keeps a bad hand; only surgery can push below the floor.
  (Provenance note: an earlier version of this count relied on two per-run
  hand-corrections of the committee detector. Those are gone — one
  detector rule, largest-gap plus a 2%-of-maximum floor derived from the
  pooled amplitude statistics of all 122 runs, is now applied uniformly
  everywhere, and it reproduces the same 121/122.)
- *Repair — with an honest split by cohort.* In natural runs, the
  frequencies leading mid-training contain bad sum-relations at exactly
  the rate chance predicts, but the final committees almost never do (4
  observed where chance predicts 18): cleanup happens during the
  consolidation phase. In the flattened-init cohorts the picture is
  different and arguably more interesting: the mid-training leaders are
  *already clean* — with no init bias forcing the early race, training
  freely picks a clean set from the start and there is nothing to repair.
  Forced starts (natural luck or our surgery) sometimes propose broken
  sets and get repaired; free starts almost never propose them. And repair
  can be triggered on demand: engineer a bad trio into the starting
  weights, and training breaks it up — demonstrated 6 out of 6 times
  across three different data splits. Downgrades from the overnight runs:
  the repair isn't always the minimal one-member swap we first thought
  (usually but not always), and we still **don't know what signal triggers
  a repair** — our best candidate hypothesis (a margin score) was killed
  by our own re-analysis, and we say so.
- *The shortlist rule.* Final committees came from the run's own
  mid-training top-8 shortlist in 119 of 122 runs (checked both at a fixed
  epoch and relative to each run's own grokking time). The three
  exceptions: the 2.25× forced implant, one engineered-collision arm, and
  one ordinary p=127 run — so "winners come from the early shortlist" is a
  strong regularity, not an absolute law.

## The one-paragraph version

Which circuit a grokking network learns is decided by a lottery held at
initialization, and we hold the winning formula: each frequency's loudness
on arrival at the computation site, the product of two random ingredients
— its size in the embedding, and its geometric fit through attention. We
can read the ticket before training starts (each ingredient separately,
and the product best of all), steer the outcome with either ingredient,
erase both and leave nothing our probes can read — and around it sits a
lawful boundary: winners came from the early-training shortlist in 119 of
122 runs, all 121 freely-trained runs clear a quality floor, and
mathematically degenerate sets get repaired when a forced start proposes
them (free starts almost never do). What's lawful is the boundary; inside
it, it's dice — and we know exactly which dice.

## Weakest points (so no review ever surprises us)

- The repair *trigger* is unknown; the margin-score candidate was killed
  by our own adversarial re-analysis and is publicly retracted.
- Adoption thresholds are context-dependent in a way we can't yet predict.
- A faint third carrier below detection (predicting at ~55% instead of
  50%) can't be excluded at n=16 flat-flat runs; ~60–100 runs would
  settle it. Worse: all 16 of those runs share a single train/test split,
  so the "nothing readable" result has n=1 on the mask axis. (The v2
  dataset adds double-flats on two fresh masks.)
- The flattened-embedding (orth-flat) evidence recycles 8 independent
  starting draws across recipe variants; counted one-vote-per-init its
  significance is marginal (p ≈ 0.03). The v2 dataset's fresh orthWE
  cells (8 never-used init seeds on 2 never-used masks) are the
  pre-registered fix: the claim predicts cluster-level readout ≈ 0.66,
  chance predicts 0.5.
- The mask-popularity effect (the same data split favors similar
  frequencies across seeds) is replicated but unexplained.
- All of it is one architecture and one task family (mod-add, p ∈
  {113, 127, 157}).
