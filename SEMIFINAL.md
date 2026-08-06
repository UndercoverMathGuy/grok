# SEMIFINAL — the claims, from scratch, in plain language

Status: 2026-08-06. This is the plain-language master summary of what we
claim, how strong each claim is, and how it was proven. Technical
receipts: the captured analysis outputs in `semifinal/results/`, run on
the fresh 98-run v2 dataset (`runs_torch/`, cloud-trained 2026-08-05,
all-new seeds and masks) against predictions pre-registered before it
trained. Every headline number below is from that fresh dataset unless
explicitly marked otherwise. (Everything that predates v2 — old runs and
the exploratory lab notebook — is archived under `legacy/` and is not
load-bearing; where the archived dataset found the same thing, we say
"replicates the archive" as background, never as proof.)

**Reproduction:** every claim below is backed by an analysis script in
`semifinal/analysis/` that auto-discovers all compatible runs under
`runs_torch/` — see `semifinal/README.md` for the claim→script map, the
pre-registered predictions with realized outcomes, and the 2026-08-06
captured outputs in `semifinal/results/`.

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

## The scoreboard — what's strong and what isn't

Ranked by how hard each result is to argue with on the fresh data:

1. **The init chooses the committee, not training noise** (claim 2).
   Same starting weights under six genuinely different training recipes
   keep choosing overlapping committees; different starting weights under
   the same recipe don't. Permutation p < 1e-4, independently in both
   data-split cells. The single cleanest fact we own.
2. **Training vetoes and repairs bad committees** (claim 4). Implant a
   specific broken trio into the starting weights and training breaks it
   up — 4/4 across two independent datasets, and both fresh repairs were
   the predicted kind (evict one endpoint). Corroborated by a census:
   untouched runs almost never end with such a set (3/70 vs ~24 expected
   by chance, p < 1e-5) even though their mid-training leaders carry them
   at roughly chance rate.
3. **The erasure chain** (claim 1): flatten ingredient 1 and only
   ingredient 2 still predicts; flatten both and nothing we own predicts.
   The logic is airtight and every prediction landed, but the decisive
   cluster-level number is modest (0.62, p = 0.044 over 8 fresh inits) —
   real, not overwhelming.
4. **Steering** (claim 3): causally demonstrated in both directions and
   through both knobs, but the fine print grew — adoption thresholds vary
   by base, and the two knobs are interchangeable at high dose, not
   exactly at threshold.
5. **The quality floor** (claim 4): 0/70 freely-trained runs below the
   25th percentile. Solid, but partly a corollary of the veto above.
6. **Open / weak**: what signal triggers a repair (unknown; our best
   candidate was killed by our own re-analysis), why the data split
   biases frequency popularity (replicated, unexplained), and everything
   in "weakest points" below.

## Claim 1: we located the lottery — and can compute the ticket with pen and paper

**Strength: the logic is the strongest in the project; the fresh numbers
land exactly where pre-registered, with one leg modest. Framing note:
the conjecture that init decides was already in the air (see
Positioning) — what's ours is the first quantitative execution and
causal test of it.**

- *You can read the future at step zero.* Before a single training step, a
  simple formula on the starting weights — multiply three matrices,
  measure each frequency's arrival loudness (T_k) — picks eventual winners
  over losers 76% of the time on average across the 8 fresh natural runs
  (coin flip would be 50%; run-level p = 7e-5; replicates the archived
  dataset's 72% over 24 runs and 3 primes). Per-run it still ranges
  widely — on any single run the read can fail.
- *Each ingredient is readable on its own, in untouched networks.*
  Ingredient 1 alone (embedding size) predicts at 0.74. Dividing it back
  out and reading only ingredient 2 (attention fit) still predicts at
  0.64 in the same runs. So the two-ingredient structure isn't inferred
  only from surgery; both factors are visible separately in ordinary
  runs.
- *Delete ingredient 1 → ingredient 2 keeps predicting, ingredient 1
  stops.* This was THE pre-registered decisive test, on 8 never-used
  init draws (54 runs counting recipe variants). In networks whose
  embedding is exactly flat across all 56 frequencies, the
  embedding-size probe collapses to chance (0.53) — the erasure worked —
  while arrival loudness, now carried purely by attention fit, still
  predicts the committee: 0.62 counted one honest vote per independent
  init (p = 0.044; run-level p = 7e-6). This replaces the old
  pseudoreplicated version of this number (8 recycled inits dressed as
  41 runs) that our own review flagged.
- *Delete both → nothing left to read.* In networks where both
  ingredients are exactly equalized, all five probes sit at 0.50–0.60
  and none is significant (8 fresh draws). The networks still grok on
  schedule and form normal committees. We erased precisely two things,
  and precisely the predictability vanished — at the precision 8 runs
  buys (see weakest points; the archived 16-run cohort agrees).
- *Scramble tests are consistent with the formula's address.* Scramble
  the embedding table at init → the prediction dies (0.74 → 0.48).
  Scramble attention → partial damage (−0.11). Scramble the MLP input →
  nothing (+0.004); and a second, per-neuron readout that *can* see the
  MLP settles the "hidden MLP carrier" worry fairly: it reads the
  committee before training, but scrambling the MLP doesn't dent it
  while scrambling the embedding kills it — the per-neuron structure is
  inherited from the embedding-through-attention stream. In the
  flattened cohorts, scrambling *either* the embedding frame or the
  attention kills the read, exactly as a two-sided relational carrier
  demands. Consistency checks, not causal proofs — the causal proof is
  the erasure chain above.

One honest note: along the way, one run in the "both flat" cohort looked
like it revealed a third hidden carrier in the MLP. Sixteen runs later
that was exposed as a statistical fluke and retracted. The lottery has
two readable carriers, and below them: nothing readable so far.

## Claim 2: the init chooses — training dynamics don't

**Strength: the strongest single result, and it got stronger on fresh
data.**

The question: is the committee decided by the starting weights, or by the
noise of training? The test: take the SAME starting weights and train
them under genuinely different regimes — plain, 3× learning rate, 0.3×
learning rate, 4× weight decay, 0.25× weight decay, worst-case-focused
loss, tilted loss — and separately, take DIFFERENT starting weights under
the same regime. If training noise decided, same-init runs would overlap
no more than strangers.

Fresh result, on flattened-embedding inits (so the harder version, where
ingredient 1 is already erased): same-init committee overlap 0.35 vs
stranger baseline 0.09–0.13, permutation p < 1e-4 — independently in
BOTH data-split cells, and it holds whether you count all seven recipe
variants or collapse them into the five genuinely distinct regimes (the
honest grouping our review demanded; the archived dataset's grouped
number was 0.29 vs 0.11). The init is the decision; the dynamics mostly
execute it.

Two companion facts sharpen this:

- *Every flattening re-rolls the lottery.* The same init seed taken
  through normal / flat-embedding / flat-both initialization chooses
  essentially unrelated committees each time (overlap 0.14 ≈ stranger
  baseline, both cells). Flattening doesn't "reveal" a deeper preference
  of the seed — it holds a new lottery with the ingredients that remain.
  This is also why the OLD version of this claim ("flatten → committees
  change → energies mattered") was circular and is retired: committees
  change if you sneeze at the init. We proved assembly is chaotic —
  swapping one dead frequency's starting value can change the outcome —
  so "it changed" proves nothing. What replaced it is the ledger above:
  same init → same choice; erase the ingredients → predictability goes
  to zero.

## Claim 3: we can steer the committee by hand

**Strength: the causal core is solid and now replicated on two fresh
bases; the quantitative fine print is genuinely context-dependent.**

- *Boost a loser → it wins.* Take a frequency the network was going to
  ignore and boost its starting embedding energy: past a threshold it
  joins the final committee, and by 2.25× it always has (both fresh
  bases, plus the archived base). The threshold itself varies: one fresh
  base adopted from 1.20×, the other needed 1.50×, the archived base
  went as low as 1.10× — and the target's peak amplitude is
  dose-monotone everywhere. "You can steer" is proven; "a fixed dose
  always works" is false.
- *Suppress a winner → it dies.* Halve the starting energy of the
  strongest destined winner: it is annihilated — 2/2 fresh bases, target
  completely evicted from the committee.
- *The other knob steers too.* The best experiment in the project:
  rotate a frequency's *direction* to fit better through attention while
  keeping its energy mathematically identical (energy error < 1e-7). At
  2.25× arrival-loudness gain the target is adopted on both fresh bases
  — steering with zero energy change. At 1.20× gain, honest downgrade:
  neither rotation arm adopted, while the 1.20× *energy* arm on one base
  did. So the two knobs are interchangeable in kind and at high dose,
  but not exactly equivalent at threshold, and (unlike the archived
  base) the bystander committee members shuffle differently under the
  two knobs. Arrival loudness is the right currency; the exchange rate
  between the knobs isn't exactly 1.
- *Energy alone does not carry identity.* Copy one run's entire
  56-frequency energy profile onto another run's embedding (keeping the
  recipient's directions): the outcome follows the recipient, not the
  donor — across 6 fresh transplants, 11 recipient-unique members
  survived vs 1 donor-unique. Whatever the donor "wanted" is not
  transported by its energy spectrum. This is the cleanest evidence that
  ingredient 2 (direction geometry) is not a nuisance term.
- *Bystanders are chaotic.* Change anything sub-threshold — even boost a
  frequency that never launches — and the rest of the committee can
  reorganize anyway (fresh data: one of two chaos pairs diverged; both
  arms of the other pair differ from their own base). You control the
  target; you do not control the neighborhood.

## Claim 4: training refuses broken committees — and repairs them on demand

**Strength: the repair-on-demand result is the strong one; the census
backs it; the trigger mechanism is still open.**

Some committees are mathematically bad: sets where one frequency is the
sum or difference of two others. Mechanically, the network multiplies
sine waves, and a product of frequencies i and j spills energy onto
i+j and i−j — if that lands on a third member, the members interfere.

- *Repair on demand — the headline.* Engineer exactly such a bad trio
  into the starting weights (boost a dead frequency t chosen so that two
  incumbent members sum to it) and watch: training breaks the trio, both
  fresh bases, and in both cases by the predicted minimal move — evict
  one endpoint, keep the rest (base {5,35,52} + implanted 40: final
  {35,40,52}, the 5 evicted since 5+35=40; base {10,11,14,43} + 53:
  final {5,11,43,53}, the 10 evicted since 10+43=53). With the archived
  dataset that's 4/4 on four different bases. This is an intervention
  with a named, falsifiable prediction — the strongest kind of evidence
  we have for the veto. (Archive caveat kept honest: in the older pair
  the repair was not always the minimal one-member swap; in the fresh
  pair it was.)
- *The census agrees.* In 70 freely-trained runs (no surgery), only 3
  final committees contain any sum/difference collision, where random
  same-size sets would give ~24 (p < 1e-5, plain run-level count — we
  no longer quote a mask-clustered statistic here; this dataset has only
  two data-split cells, too few to cluster on). Meanwhile the
  mid-training *leaders* carry collisions at near chance rate (34 across
  all 96 runs vs 5 in the finals). So awkward candidates get in the door
  and get cleaned up during consolidation.
- *Provenance differs by cohort, and the claim must say so.* In natural
  runs the mid-training leaders are dirty and the finals are clean:
  active repair during consolidation. In flattened-init runs the leaders
  are *already* clean — with no init bias forcing the early race,
  training freely picks a clean set from the start and there is nothing
  to repair. Forced starts (natural luck or our surgery) sometimes
  propose broken sets and get repaired; free starts almost never propose
  them.
- *A floor.* Ranking every freely-trained final committee against random
  alternatives by an optimal-margin score: 0 of 70 fall in the bottom
  quartile (minimum: 43rd percentile). Only surgical arms ever get near
  the line (a transplant at 27) — you can push a committee toward the
  floor by force, but free training never goes there. One detector rule
  (largest amplitude gap + 2%-of-max floor) is applied uniformly
  everywhere; the per-run hand-corrections an earlier version relied on
  are gone.
- *The shortlist rule.* Final committees came from the run's own
  mid-training top-8 shortlist in 95 of 96 runs (measured relative to
  each run's own grokking time; 93/96 at a fixed early epoch). The one
  exception is a transplant arm. "Winners come from the early shortlist"
  is a strong regularity, not an absolute law.
- *What's still open:* we do not know what signal triggers a repair. Our
  best candidate (a margin score) was killed by our own re-analysis and
  is retracted. The veto is a fact; its sensor is not identified.

## The one-paragraph version

Which circuit a grokking network learns is decided by a lottery held at
initialization, and we hold the winning formula: each frequency's
loudness on arrival at the computation site, the product of two random
ingredients — its size in the embedding, and its geometric fit through
attention. The same starting weights keep choosing the same committee
under six different training regimes (p < 1e-4, two independent cells);
we can read the ticket before training starts, steer the outcome through
either ingredient, and erase both to leave nothing our probes can read.
Around the lottery sits a lawful boundary: winners come from the
early-training shortlist (95/96), every freely-trained run clears a
quality floor (70/70), and mathematically self-interfering sets are
vetoed — implant one on purpose and training repairs it by evicting an
endpoint, 4/4 across two datasets, while untouched runs end with such a
set 3 times in 70 where chance predicts 24. What's lawful is the
boundary; inside it, it's dice — and we know exactly which dice.

## Positioning against the literature (2026-08 uniqueness audit)

How each claim sits relative to prior work, after reading the close
calls ourselves:

- **The conjecture was stated before us; the execution wasn't.** Varma
  et al. (ICLR 2024) say in an appendix, in passing: the learned
  frequencies are "typically whichever frequencies were highest at the
  time of random initialisation" — no metric, no definition of
  "highest," no pathway. Chughtai, Chan & Nanda (ICML 2023) list
  init-time lottery tickets for feature identity as future work. So our
  headline is not "nobody thought of this"; it is **the first
  quantitative execution and causal test of a stated-but-untested
  conjecture** — a definition of the ticket (T_k, arrival loudness
  through the attention pathway), its measured predictive power, and
  intervention in both directions.
- **The closest analytic result is real but a different object.** He,
  Wang, Chen & Yang (arXiv:2602.16849, Feb 2026; companion
  arXiv:2606.02993 for general groups) prove, for TWO-LAYER MLPs on
  one-hot inputs, that per-neuron frequency competition is won by
  "initial spectral magnitude and phase alignment" — a magnitude ×
  alignment factorization in spirit like ours. Differences that matter:
  no embedding matrix and no attention (their "alignment" is a
  neuron-internal phase relation, not geometric fit through an OV
  circuit); per-neuron winners, not a network-level committee; and
  their theory predicts *dense* coverage — we checked their
  "diversification condition" directly: it requires every frequency to
  be represented across the neuron population with phase symmetry
  inside each frequency group, and places NO constraint on arithmetic
  relations between frequencies. So it anticipates the flavor of claim
  1, and does not touch the committee, the veto, the repair, or the
  floor.
- **Sparse committees sit in TENSION with margin theory — we must say
  so, not hide it.** Morwani et al. (ICLR 2024) prove that max-margin
  solutions use ALL frequencies at sufficient width (their appendix is
  literally titled "Proof that all frequencies are used"), and their
  framework is invariant under frequency relabeling, so it structurally
  cannot rank individual frequencies. He et al.'s diversification is
  dense too. Our transformers reliably pick 3–5 of 56. The honest
  reading: the sparse-committee regime is a finite-width /
  finite-training phenomenon outside those asymptotic regimes — which
  is exactly where a lottery CAN operate (a symmetric infinite-width
  theory has no tiebreaker; reality does). This is an argument we make,
  not a contradiction we fear; but it needs to be made explicitly.
- **Weight-level chaos strengthens, not threatens, claim 2.** Recent
  work (arXiv:2506.13234) shows single-weight early perturbations send
  identically-initialized networks to functionally different solutions
  — and our own experiments agree (assembly is chaotic to
  sub-threshold init swaps). Yet the discrete committee identity
  survives across six different training regimes (overlap 0.35 vs 0.10
  strangers). The right statement: the weight *trajectory* is chaotic;
  the committee *identity* is an attractor readable from the init. A
  discrete, robust observable riding on a chaotic continuous system is
  the interesting fact.
- **Init-time pruning failures are not our failure.** Grokking-tickets
  work found that standard init-pruning criteria (random, SNIP, GraSP,
  SynFlow) cannot find the winning subnetwork. Those are mask-based,
  weight-space criteria; ours is a closed-form spectral quantity — and
  our own legacy result agrees masks are the wrong language (mask-based
  proxies were null while the spectral readout works). Nobody had
  tested a spectral init predictor; that is the gap we fill.
- Caution for the writeup: don't cite Nanda's {14,35,41,42,52} as a
  reproducible committee (one mainline seed), and Gromov (2301.02679)
  is safe to cite briefly and dismiss (frequency set task-determined
  and dense, only phases init-random, MLP under MSE).
- Audit coverage caveat: ~8 primary sources plus targeted searches, not
  a systematic sweep; "novel" verdicts on claims 3–6 (steering,
  init-over-dynamics, veto/repair, floor) are medium confidence.

## Weakest points (so no review ever surprises us)

- **The v2 dataset is one prime (113) and two data splits.** The
  three-prime breadth lives only in the archived dataset. Anything
  "across masks" now has n=2 — which is why we removed mask-level
  statistics rather than quote degenerate ones. More mask cells is the
  single cheapest upgrade.
- **The natural cohort's cluster-level significance is underpowered by
  design** (8 runs, 2 mask clusters): run-level p-values carry that
  cohort. The orth-flat cohort is the one with honest cluster-level
  support (8 independent inits, p = 0.044) — real but modest.
- **"Nothing readable" in the double-flat cohort is n=8** on fresh data
  (n=16 archived, all on one mask). A faint third carrier reading at
  ~0.55 can't be excluded; one probe (forward alignment) sits at 0.60
  with p = 0.065 — consistent with noise, not proof of it.
- **Knob equivalence is approximate.** Energy and rotation steering
  agree at 2.25×; at 1.20× they disagreed on one base, and bystander
  outcomes differ between knobs. "Arrival loudness is the currency" holds;
  "the knobs are exactly interchangeable" doesn't.
- **The repair trigger is unknown**; the margin-score candidate was
  killed by our own adversarial re-analysis and is publicly retracted.
- **Adoption thresholds are context-dependent** in a way we can't yet
  predict (1.10×–1.50× across bases; one archived target never launched).
- **The mask-popularity effect** (the same data split favors similar
  frequencies across seeds) is replicated but unexplained.
- **Two of the 98 fresh runs never consolidated** (final test CE > 5e-3,
  10× above every other run) and are excluded by a documented rule
  (`BAD_RUNS` in `analysis/common.py`); both are dynamics-variant arms,
  and no conclusion changes if they are kept.
- All of it is one architecture and one task family (modular addition).
