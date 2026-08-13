export const meta = {
  name: 'grok-hypothesis-fanout',
  description: 'Four-angle hypothesis generation + opus-run analysis tests on existing grokking runs, then synthesis',
  whenToUse: 'Ultracode research fan-out over the grok repo natural runs',
  phases: [
    { title: 'Hypothesize', detail: '4 fable agents, one per angle: generate, self-criticize, novelty-check' },
    { title: 'Test', detail: 'one opus-implementer per surviving angle: build + run pre-registered analysis tests', model: 'claude-opus-5' },
    { title: 'Synthesize', detail: 'one fable agent compiles preregistrations vs results' },
  ],
}

const BRIEF = `
## Research context (read carefully)

You are part of a multi-agent research push targeting an **ICLR 2027 main-track submission** — you know the bar that implies: findings must be novel, surprising, rigorously tested, and not incremental restatements of known mechanistic-interpretability results.

**Domain**: grokked modular addition. Small transformers trained on (a+b) mod p (mostly p=113; some legacy runs at p=127, p=157), 1 layer, d_model=128, 4 heads, d_mlp=512, ReLU, full-batch AdamW, frac_train=0.3. These models grok: memorize first, then generalize abruptly. The known background (Nanda et al.): after grokking the network implements a Fourier / trig algorithm — logits ≈ Σ_{k∈S} A_k cos(w_k(a+b−c)), w_k=2πk/p, over a sparse set S of 3–6 "committee" frequencies out of the 56 folded frequencies.

**Repo**: /Users/ruhaanrajadhyaksha/projects/grok
**Run data — the entire evidence base (NO NEW TRAINING ALLOWED, verify your ideas need none):**
- \`runs_torch/\` — the v2 dataset: ~96 kept runs at p=113 across several cohorts (natural, surgical-init, orthogonalized-embedding, etc.).
- \`legacy/runs/\` — older archive: natural families seed0/seed1/seed2/og_seed0 and cross-prime cells p-113/p-127/p-157, plus experiment folders.
- Per run: \`spectra.npz\` (arrays: epochs (~501 samples, 100-epoch cadence), coeffs (501×56 Fourier coefficient magnitudes), energy, train_acc, test_acc), \`config.json\`, and **checkpoints every 1000 epochs as .safetensors** — so full weight trajectories and forward/backward passes on frozen checkpoints ARE available and allowed (cheap on-device analysis; use device='cpu' in torch to avoid Metal concurrency crashes; NEVER launch training).
- Shared utilities: \`semifinal/analysis/common.py\` — discover() (iterates v2 runs w/ configs+spectra+final committee, excludes known-bad runs), committee_from_coeffs(), fold(x,p)=min(x%p,p−x%p), grok_epoch(), jaccard(). Reuse these rather than re-deriving run discovery.

**IMPORTANT — stance toward the repo's own docs**: the repo contains many .md files with prior claims, experiments, and conclusions (init-lottery, collision veto, margin/flatness, etc.). Treat these ONLY as seed material / known territory — do NOT adopt their framings, do NOT build your contribution as an extension of their headline claims, and do NOT re-derive them and present them as new. You are explicitly asked to approach the problem of finding NEW phenomena differently. Anything already claimed in the repo's markdown = already known internally = zero novelty credit.

**Novelty check (mandatory)**: before finalizing any hypothesis, use the WebSearch tool (load it via ToolSearch "select:WebSearch" if deferred) to check whether the insight already exists in the literature — Nanda et al. progress measures, Gromov analytic solutions, Morwani et al. max-margin Fourier features, Varma et al. efficiency, Zhong et al. pizza/clock, He et al. 2602.16849, Kwok et al., Liu et al. omnigrok/effective theory, Thilak slingshot, etc. If a hypothesis is a known result, discard or sharpen it into the unknown part.

**Hardware**: Apple M1, no GPU training. Analysis scripts in python (numpy/scipy/torch available). Write all scripts and temp files under /private/tmp/claude-502/-Users-ruhaanrajadhyaksha-projects-grok/c77a5e1a-76f9-4612-8e9f-2425148a400e/scratchpad/ (create subdirs per angle). Keep any single computation < ~10 min.
`

const HYP_SCHEMA = {
  type: 'object', required: ['angle_summary', 'hypotheses', 'discarded'],
  properties: {
    angle_summary: { type: 'string', description: 'One paragraph: how you approached this angle' },
    hypotheses: {
      type: 'array', minItems: 1, maxItems: 3,
      items: {
        type: 'object', required: ['id', 'statement', 'preregistered_prediction', 'test_plan', 'novelty_notes', 'why_surprising'],
        properties: {
          id: { type: 'string', description: 'short slug, e.g. math-1' },
          statement: { type: 'string', description: 'The hypothesis, precisely stated' },
          preregistered_prediction: { type: 'string', description: 'Concrete quantitative prediction(s) committed BEFORE any test runs: what number/pattern will be observed if true, and what outcome falsifies it' },
          test_plan: { type: 'string', description: 'Exact analysis procedure over existing runs (files, quantities, statistics, null model). Must need NO new training.' },
          novelty_notes: { type: 'string', description: 'What the web search found; why this is not already in the literature or the repo docs' },
          why_surprising: { type: 'string', description: 'Why a reviewer would find this non-obvious' },
        },
      },
    },
    discarded: { type: 'array', items: { type: 'string' }, description: 'Candidates you killed in criticism and the one-line reason (known result / needs training / untestable / boring)' },
  },
}

const TEST_SCHEMA = {
  type: 'object', required: ['results', 'scripts', 'surprises'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object', required: ['id', 'verdict', 'evidence', 'caveats'],
        properties: {
          id: { type: 'string', description: 'hypothesis id being tested' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'PARTIAL', 'REFUTED', 'INCONCLUSIVE'] },
          evidence: { type: 'string', description: 'Key numbers: effect sizes, n, p-values, null-model comparisons — enough for a reader to judge' },
          caveats: { type: 'string', description: 'Confounds, power limits, multiple-testing exposure, anything a hostile reviewer would attack' },
        },
      },
    },
    scripts: { type: 'array', items: { type: 'string' }, description: 'Absolute paths of analysis scripts written' },
    surprises: { type: 'string', description: 'Any unplanned anomaly noticed while testing (or "none")' },
  },
}

const ANGLES = [
  {
    key: 'math',
    title: 'Mathematical modelling',
    prompt: `Your angle: **mathematical modelling**. Attack the problem with mathematics first, data second — both model-agnostic math (properties of cosine sums over Z_p, character sums, extremal/combinatorial structure of frequency sets, invariances such as multiplicative dilation, number-theoretic structure of p) and model-internal math (we KNOW how the trained model computes: embeddings concentrate on committee frequencies, attention is roughly uniform/bilinear, the ReLU MLP squares sums to produce cos(w_k(a+b)) terms, unembedding reads out cos(w_k(a+b−c))). Derive, from first principles, statements about what such a computation MUST or CANNOT do — quantities that are forced, conserved, bounded, or forbidden — and turn the most surprising derivations into hypotheses testable purely on the existing run artifacts (spectra time-series, checkpoints, forward passes). Prioritize derivations whose predictions are sharp (exact zeros, exact scalings, parameter-free constants) over soft directional claims.`,
  },
  {
    key: 'observation',
    title: 'Observation / internals',
    prompt: `Your angle: **observational internals**. Go looking inside the trained networks for unexplained structure, with no theory commitments up front. Allowed and encouraged: forward-pass activation peeking on frozen checkpoints (attention patterns, MLP neuron activations over the full 113×113 input grid, logit decompositions), weight-distribution analysis (per-neuron norms, spectra of weight matrices, embedding geometry across the checkpoint trajectory), and selective gradient analysis (per-example or per-frequency gradients computed on frozen checkpoints — backward passes are fine, optimizer steps are not). Hunt for anomalies: things that are bimodal when they should be unimodal, structured when they should be isotropic, synchronized when they should be independent, or asymmetric when the task is symmetric. When you find one, form a hypothesis about its source, with a pre-registered discriminating prediction. First do a quick reconnaissance pass on 2-3 checkpoints yourself before committing hypotheses, so your hypotheses are anchored in real observations rather than guesses — but keep recon shallow; the heavy testing belongs to the tester agent.`,
  },
  {
    key: 'phenomena',
    title: 'Phenomena / dynamics',
    prompt: `Your angle: **phenomena-driven dynamics**. Mine the existing training trajectories — 500-point Fourier-coefficient time series per run across ~120 runs (v2 + legacy), plus 1000-epoch checkpoint snapshots — for WEIRD, unexplained temporal behaviour: non-monotonicities, oscillations, synchronized transitions across frequencies, sudden reorderings, near-deaths and revivals of frequencies, differences between cohorts nobody engineered, epochs where many runs do the same strange thing. Actually load and look at a sample of trajectories yourself (quick plots/statistics) before committing hypotheses. Then, for the strangest reproducible phenomenon you find, use logic to determine candidate causes and design a discriminating test with a pre-registered prediction that separates the causes. The deliverable is not "here is a phenomenon" but "here is a phenomenon + a causal hypothesis + a test that could kill it".`,
  },
  {
    key: 'elegance',
    title: 'Elegance / normative',
    prompt: `Your angle: **elegance / normative design**. Reason as if you were the model: given the task (a+b mod p), the architecture (1-layer, 4 heads, 512 ReLU neurons), and the training pressures (full-batch AdamW, weight decay, cross-entropy), what SHOULD an optimal or maximally elegant solution do? Enumerate ideal properties a beautiful solution would have — e.g. optimal allocation of neurons across frequencies, phase arrangements that minimize interference, error-correcting redundancy, load balancing across heads, some invariance or symmetry the solution ought to respect, an information-theoretically optimal readout — and then check which of them the real networks actually satisfy and which they conspicuously violate. BOTH outcomes are findings: "the network achieves near-optimal X without being trained for it" and "the network is far from optimal at Y in a systematic, structured way" are each papers-worth if sharp. Commit predictions with quantitative optimality baselines you compute from math before looking at the networks.`,
  },
]

function hypPrompt(a) {
  return `${BRIEF}

# Your task: hypothesis generation — angle "${a.title}"

${a.prompt}

## Process (do all of it)
1. Skim the repo enough to know the data formats and what the repo already claims (so you can avoid it). Read semifinal/analysis/common.py.
2. Generate 5–8 candidate hypotheses from your angle.
3. Run an INTERNAL CRITICISM pass — be your own hostile reviewer: kill anything that (a) is already in the literature or the repo's docs, (b) would need new training runs, (c) has no falsifiable quantitative prediction, (d) would not surprise an ICLR reviewer. Use WebSearch on your survivors (load via ToolSearch "select:WebSearch" if needed).
4. Keep the best 1–3 survivors. For each, write a PRE-REGISTERED prediction: the specific quantitative outcome expected if true AND the outcome that falsifies it, committed now, before any test is run.
5. Verify each test plan is executable purely on existing artifacts (spectra.npz / checkpoints / config.json) in <10 min of M1 compute.

Your structured output is handed verbatim to an implementation agent who will run the tests exactly as you specify — write test_plan concretely enough for that (which runs/cohorts, which files, which statistic, which null model, which threshold).`
}

function testPrompt(a, hyp) {
  return `${BRIEF}

# Your task: implement and run pre-registered tests — angle "${a.title}"

A hypothesis-generation agent produced the following pre-registered hypotheses. Your job: implement each test_plan faithfully, run it over the existing runs, and report verdicts against the preregistered_prediction. You may NOT modify a hypothesis to fit the data; if a test plan is underspecified, make the most natural choice and record it as a caveat. If a plan turns out to be infeasible without new training, report INCONCLUSIVE with the reason.

## Pre-registered hypotheses (verbatim)
${JSON.stringify(hyp, null, 2)}

## Rules
- Write scripts under the scratchpad dir (subdir "${a.key}/"); reuse semifinal/analysis/common.py for run discovery where possible.
- torch on device='cpu' only; never train; keep each computation < ~10 min.
- Statistics: always include a null model / baseline; report effect size AND n AND a p-value where meaningful; prefer paired tests within-run; watch for pseudoreplication (runs sharing an init seed are not independent).
- Judge verdicts against the preregistered prediction ONLY. A result that is interesting but different from the prediction goes in "surprises", not in a bent verdict.
- Be adversarial toward your own results: before reporting CONFIRMED, attempt at least one falsification (a competitor explanation, a permutation null, or a robustness split).`
}

function synthPrompt(bundle) {
  return `${BRIEF}

# Your task: final synthesis

Four research angles each produced pre-registered hypotheses; an implementation agent then ran the tests. The full record (preregistrations + verdicts + evidence, verbatim) is below. Compile the honest scoreboard:

1. **What worked**: hypotheses CONFIRMED or PARTIAL with strong evidence — for each, restate the finding in plain language, its effect size, and why it is (or is not) genuinely novel vs the literature and the repo's prior claims. Rank by ICLR-worthiness.
2. **What died**: refuted or inconclusive hypotheses, and what each death teaches.
3. **Cross-angle connections**: places where two angles' results illuminate or contradict each other.
4. **Surprises registry**: every unplanned anomaly the testers logged, triaged by follow-up value.
5. **Verdict + next step**: is there a headline-candidate here? What is the single highest-value follow-up analysis (still no new training) — and what would need new training runs?

Be maximally honest — deflate weak confirmations (multiple testing across ~10 hypotheses, pseudoreplication, post-hoc flexibility). The user will use this to decide their ICLR direction; a false positive costs them weeks.

## Full record
${JSON.stringify(bundle, null, 2)}`
}

log('Fanning out 4 hypothesis agents (math, observation, phenomena, elegance); each surviving slate flows to an opus tester')

const bundles = await pipeline(
  ANGLES,
  (a) => agent(hypPrompt(a), { label: `hypothesize:${a.key}`, phase: 'Hypothesize', schema: HYP_SCHEMA }),
  (hyp, a) => {
    if (!hyp || !hyp.hypotheses || hyp.hypotheses.length === 0) { log(`angle ${a.key}: no surviving hypotheses`); return null }
    log(`angle ${a.key}: ${hyp.hypotheses.length} pre-registered hypothesis(es) -> opus tester`)
    return agent(testPrompt(a, hyp), { label: `test:${a.key}`, phase: 'Test', agentType: 'opus-implementer', schema: TEST_SCHEMA })
      .then((test) => ({ angle: a.key, preregistration: hyp, test }))
  }
)

const record = bundles.filter(Boolean)
log(`Testing complete for ${record.length}/4 angles; synthesizing`)

const synthesis = await agent(synthPrompt(record), { label: 'synthesis', phase: 'Synthesize' })

return { synthesis, record }