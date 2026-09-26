export const meta = {
  name: 'grok-research-opus55',
  description: 'Rigorous, expansive research fan-out on Opus 5.5: known-territory + literature ledgers, 9 preregistered angles, hostile referee, per-hypothesis proof/analysis, multi-lens adversarial verification, critic-driven extra rounds, audited synthesis',
  whenToUse: 'Deep research pushes on the grok/heracles programme (theory, exact computation, or analysis of existing runs). args: {question, tag, rounds, angles, extraAngles, maxHypPerAngle, maxNewDirections, allowTraining, context, effort, model, outDir}',
  phases: [
    { title: 'Scope', detail: 'known-territory ledger + 3-modality literature sweep', model: 'claude-opus-5-5' },
    { title: 'Hypothesize', detail: 'one agent per angle: 6-10 candidates, self-criticism, novelty check, preregistration', model: 'claude-opus-5-5' },
    { title: 'Referee', detail: 'hostile preregistration referee per angle (kill / tighten / proceed), then cross-angle dedup', model: 'claude-opus-5-5' },
    { title: 'Test', detail: 'one agent per hypothesis: full proof or analysis, judged against the preregistration only', model: 'claude-opus-5-5' },
    { title: 'Verify', detail: '3-4 lens skeptics per positive result; kill-audit per refutation', model: 'claude-opus-5-5' },
    { title: 'Critique', detail: 'completeness critic after each round: what is missing -> next round angles', model: 'claude-opus-5-5' },
    { title: 'Synthesize', detail: 'scoreboard + paper integration, then an independent overclaim audit', model: 'claude-opus-5-5' },
  ],
}

// ---------------------------------------------------------------- configuration

const REPO = '/Users/ruhaanrajadhyaksha/projects/grok'
const MEMDIR = '/Users/ruhaanrajadhyaksha/.claude/projects/-Users-ruhaanrajadhyaksha-projects-grok/memory'
const A = args && typeof args === 'object' && !Array.isArray(args) ? args
  : (typeof args === 'string' && args.trim() ? { question: args.trim() } : {})
const MODEL = A.model || 'claude-opus-5-5'
const TAG = String(A.tag || 'run').replace(/[^A-Za-z0-9_-]/g, '_')
const OUT = A.outDir || (REPO + '/heracles/wf_research/' + TAG)
const ROUNDS = Math.max(1, Math.min(4, Number(A.rounds ?? 2)))
const MAX_HYP = Math.max(1, Math.min(5, Number(A.maxHypPerAngle ?? 3)))
const MAX_NEW = Math.max(0, Math.min(8, Number(A.maxNewDirections ?? 4)))
const ALLOW_TRAIN = A.allowTraining === true
const EFFORT = Object.assign(
  { scope: 'medium', hyp: 'medium', referee: 'xhigh', dedup: 'medium', test: 'medium', verify: 'xhigh', critic: 'xhigh', synth: 'xhigh' },
  A.effort || {})

const DEFAULT_QUESTION = 'What else can be rigorously derived, explained, or predicted from the two-sided margin bound for the product model on modular addition (unrestricted upper bound B_p Gamma_s(p); exact junk-free-class optimum nu^(1-s) B_p Gamma_s(p); Gamma_s(p) = Theta((log p)^(1-s)) and K* = Theta(log p)) -- new theorems, sharp zero-parameter predictions for trained networks checkable on existing artifacts, and exact extensions to other tasks, groups, penalties and optimisers -- that make the ICLR paper materially stronger?'
const QUESTION = A.question || DEFAULT_QUESTION

const slug = (s) => String(s || 'x').toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'x'
const clip = (s, n) => {
  const t = typeof s === 'string' ? s : JSON.stringify(s ?? '')
  return t.length > n ? t.slice(0, n) + ' ...[truncated; full text in the files]' : t
}
const opts = (label, phase, effort, schema) => {
  const o = { label, phase, model: MODEL, effort }
  if (schema) o.schema = schema
  return o
}

// ---------------------------------------------------------------- shared brief

const BRIEF = `
## Mission
You are one agent in a multi-agent research push for an ICLR 2027 main-track paper, "Explaining Feature Sparsity Through the Margin-Regularization Trade-off" (abstract submitted; full paper due 2026-09-26 17:30 IST). The bar: novel, surprising, rigorously established, not a restatement of known results. A false positive costs the author weeks; an honest null beats an inflated positive.

## Research question for this run
${QUESTION}
${A.context ? '\n## Extra context from the author\n' + A.context + '\n' : ''}
## The established core (check details in the files; never present it as new)
- Product model: logits z(a,b,c) = sum_i e1_i(a) e2_i(b) w_i(c) for the task c = (a+b) mod p, p an odd prime; n = 2 operands, degree nu = n+1 = 3 (general nu via the n-operand-sum family). Weight penalty P_b (b = 2 is plain weight decay: P_2 = sum of squared weights). Normalised margin gamma = (smallest logit gap between the right answer and any wrong answer, over all inputs) / P_b^(nu/b). Exponent s = nu/b.
- Share programme: Gamma_s(p) = max over shares y_k >= 0 with sum_k y_k^(1/s) <= 1 of min over offsets D != 0 of sum_k y_k (1 - cos(2 pi k D / p)). "Coverage" of the wrong answer at offset D by frequency k is 1 - cos(2 pi k D / p). K* = number of active frequencies at the optimum. Exact code: heracles/chain/common.py (programme, block_net, gamma_sym, spectrum).
- Two-sided bound: nu^(1-s) B_p Gamma_s(p) <= gamma*(p) <= B_p Gamma_s(p), with B_p = 2 (2 p nu)^(-nu/2). The upper bound holds for every network; the lower bound is the exact optimum of the junk-free class, attained by common-phase (n+1)-neuron blocks. At nu = 3, b = 2 the two sides differ by sqrt(3).
- Conjecture (T) (the global optimum is junk-free) is OPEN and SHELVED. Do not work on it unless the research question asks; its dead routes are recorded in heracles/CHAIN.md and heracles/chain/VERIFY.md.
- New 2026-09-25 (hand-checked; literature check pending): Gamma_s(p) = Theta((log p)^(1-s)), K* = Theta(log p) for every programme optimiser, and gamma*(p) = Theta(p^(-nu/2) (log p)^(1-nu/b)) for ALL networks -- heracles/astra/LOGARITHMIC_SCALING.md. Order statements only; the explicit constants are ~185x loose at s = 3/2.
- Trained networks are NOT junk-free. Empirical statements about them are trends and ceilings, not instances of the theorem.

## Known territory (zero novelty credit)
- The author's research-memory index: ${MEMDIR}/MEMORY.md -- one line per finding; the linked files hold details. It records what is proven, what died and why, and the expected reviewer attacks.
- Repo docs: heracles/*.md (THEORY*.md, RESULTS_*.md, CHAIN.md, LADDER.md, ...), heracles/chain/*.md, heracles/astra/, heracles/s5theory/, heracles/paper/theory_modadd.md, and the paper itself: heracles/paper/iclr2027/paper.tex + sections/.
- Anything claimed there is known internally. Anything recorded as dead must not be re-proposed unless you name the new idea that answers why it died.
- Prior art already mapped by the author: Morwani et al. (max-margin Fourier features; dense L_{2,nu} endpoint), Nanda et al. (progress measures), Gromov (analytic solution), Chughtai et al. (toy model of universality), Stander et al., Li et al. (x^n family; general-nu dense endpoint), McCracken (log-n feature-count observation), Jacobs & Montufar (exponent decides sparse vs dense), He et al. arXiv 2602.16849, Ding (width law), Wei et al. 2019 (weak regularisation -> max margin), Lyu & Li (implicit bias of homogeneous nets), Xie & Li 2024 (AdamW <-> l_inf-constrained optimisation), Mohamadi et al. 2024, Alon & Roichman (random Cayley expanders).

## Evidence base
- Exact tools: heracles/chain/common.py; heracles/*.py (ladders, kstar tools, constructions); heracles/astra/*.py.
- Trained-network artifacts: heracles/FINAL_RUNS/*, runs_torch/ (v2 p=113 transformer runs; semifinal/analysis/common.py has discover()), legacy/runs/, nudial/ (ReLU/MLP nu-dial runs). Read the relevant RESULTS_*.md first to learn the formats.

## Hard rules
- Python: uv run --with numpy --with scipy [--with torch] python <script>. torch on device='cpu' only. Prefix heavy numerics with OMP_NUM_THREADS=2 -- several agents share one 8-core M1.
- Keep every single computation under ~10 minutes. No cloud, no pods, no git commits.
- ${ALLOW_TRAIN ? 'Small local CPU training runs are allowed: at most 3 per agent, each under 10 minutes, run sequentially.' : 'NO new training. Allowed: exact computation, constructed networks, analysis of existing artifacts, forward/backward passes on frozen checkpoints.'}
- Write ONLY inside your own subdirectory of ${OUT}/ (create it). Never edit existing repo files.
- Plain language: define every symbol the first time you use it and give a one-line plain-English reading of every formula.
- Web: load WebSearch / WebFetch via ToolSearch ("select:WebSearch,WebFetch") when you need them.

## Rigor standards
- THEOREM: complete proof with every inequality justified, all hypotheses stated, scope limits explicit, imported results cited precisely (paper + theorem number); plus a numerical check on exact small cases -- a failed check kills the claim.
- EMPIRICAL: preregistered prediction and falsifier, null model or baseline, effect size + n + uncertainty; runs sharing an init seed are not independent; account for multiple testing (this run tests roughly 20-40 hypotheses).
- COMPUTATION: reproducible script, optimality / convergence evidence, stated tolerances.
- Preregistrations are immutable once refereed. A result that is interesting but differs from the prediction is a surprise, never a bent verdict.
`

// ---------------------------------------------------------------- scope phase

const LEDGER_SCHEMA = {
  type: 'object', required: ['summary', 'path'],
  properties: {
    summary: { type: 'string', description: 'At most ~700 words: proven (file + verification status), open, dead (with why), empirical facts, expected reviewer attacks -- the parts relevant to the research question' },
    path: { type: 'string', description: 'Absolute path of the full ledger file written' },
  },
}

const LIT_SCHEMA = {
  type: 'object', required: ['summary', 'nearest', 'path'],
  properties: {
    summary: { type: 'string', description: 'At most ~500 words; lead with anything that already proves or nearly proves one of our results or the research target' },
    nearest: {
      type: 'array', maxItems: 15,
      items: {
        type: 'object', required: ['citation', 'claim', 'overlap'],
        properties: {
          citation: { type: 'string', description: 'authors, title, venue or arXiv id, year, URL' },
          claim: { type: 'string', description: 'what it actually proves or shows, quoted or closely paraphrased from the paper body' },
          overlap: { type: 'string', description: 'overlap with our results / the research question, and the risk it poses' },
        },
      },
    },
    path: { type: 'string', description: 'Absolute path of the full literature file written' },
  },
}

const LIT_MODES = [
  { key: 'ml', text: 'ML literature 2019-2026 (include 2025-2026 arXiv preprints): grokking and modular arithmetic (mechanistic and theoretical), max-margin and implicit bias of homogeneous networks and of specific optimisers (GD, Adam/AdamW, sign methods, Muon), feature counts, sparsity and superposition, the representation cost of weight decay.' },
  { key: 'math', text: 'Mathematics literature relevant to the share programme: the Chowla cosine problem, weighted covering of Z_p (or finite abelian groups) by cosines, minima of trigonometric polynomials with non-negative or constrained coefficients, random Cayley graphs and expanders, Sidon and dissociated sets, Gauss and character sums, Kronecker / Dirichlet approximation, concave-cost covering and sparse approximation. Top priority: any known result equivalent to Gamma_s(p) = Theta((log p)^(1-s)) or K* = Theta(log p), or to the deletion and positive-Fourier arguments in heracles/astra/LOGARITHMIC_SCALING.md.' },
  { key: 'cite', text: 'Citation chasing: forward and backward citations of Morwani et al., Li et al. (x^n), McCracken, Jacobs & Montufar, He et al. 2602.16849, Ding (width law), Mohamadi et al. 2024, Gromov. List every 2024-2026 paper that overlaps with an exact-optimum, feature-count, or margin-scaling claim for modular arithmetic or group tasks.' },
]

function ledgerPrompt() {
  return BRIEF + `
# Your task: known-territory ledger (Scope)
Build the internal ledger every later agent will use to avoid rediscovering known material and dead ends.
1. Read ${MEMDIR}/MEMORY.md and open every linked memory file relevant to the research question.
2. Read the core docs: heracles/paper/theory_modadd.md, the paper sections in heracles/paper/iclr2027/sections/, heracles/CHAIN.md, heracles/chain/VERIFY.md, heracles/astra/LOGARITHMIC_SCALING.md, plus the THEORY*/RESULTS_* files relevant to the question.
3. Write ${OUT}/scope/LEDGER.md with sections: PROVEN (statement, file, verification status), OPEN, DEAD (what died and exactly why), EMPIRICAL FACTS (numbers with provenance), EXPECTED REVIEWER ATTACKS, TOOLS (scripts that compute exact quantities, and how to call them).
Cite file paths. Where docs disagree, say so rather than silently picking one.`
}

function litPrompt(m) {
  return BRIEF + `
# Your task: literature sweep -- ${m.key}
${m.text}
Use WebSearch and WebFetch. Read the actual statements of the closest papers (the abstract is not enough when overlap is plausible). Write ${OUT}/scope/LIT_${m.key}.md with full citations and URLs.`
}

function scopeBlock(scope) {
  return `
## Scope ledgers (open the files for detail)
Internal ledger (${scope.ledgerPath}):
${clip(scope.ledger, 6000)}

Literature (${scope.litPaths.join(', ') || 'unavailable'}):
${scope.lits.map((l) => clip(l, 3500)).join('\n\n') || '(literature sweep unavailable -- do your own searches)'}
`
}

// ---------------------------------------------------------------- angles

const ANGLES = [
  {
    key: 'derive', title: 'Theorem derivation',
    prompt: 'Mine the two-sided bound for new theorems. Within the junk-free class the bound fixes the optimal network almost completely (frequency shares, count, neuron phases, width, the full logit table), and it fixes the global order in p for all networks. Derive consequences that are forced, sharp and provable: corollaries, finite-p refinements, structural characterisations of optimisers (uniqueness up to automorphism, share floors, the set of tight offsets), width lower bounds, robustness / stability margins, extensions to general nu and b. Prefer statements with complete short proofs over ambitious conjectures; every theorem needs a numerical check on exact small cases.',
  },
  {
    key: 'numtheory', title: 'Number theory / additive combinatorics transfer',
    prompt: 'The share programme is a weighted cosine-covering problem over Z_p under a concave budget -- kin to the Chowla cosine problem, Sidon and dissociated sets, Gauss and character sums, difference sets, Kronecker / Dirichlet approximation, random Cayley expanders and sum-product bounds. Import known results to (i) sharpen the Theta(log p) constants toward something that predicts finite p (p = 59 ... 4001), (ii) characterise the arithmetic structure of optimal frequency sets (which additive relations are forbidden or forced), (iii) handle composite moduli and other abelian groups. Cite exact theorems; check every prediction against exact programme values (heracles/chain/common.py programme(), the existing ladders).',
  },
  {
    key: 'crossdomain', title: 'Cross-domain transfer',
    prompt: 'Treat the result as an instance of a general principle -- features are the cheapest cover of the confusable outputs under a concave per-feature cost -- and look for exact dictionaries with other fields: sidelobe / code design (radar, CDMA), compressed sensing and sparse approximation, spherical and t-designs (neuron phase blocks), statistical physics of covering, and implicit-bias theory for other optimisers (each maximises margin in its own geometry: GD <-> l_2, AdamW <-> l_inf box (Xie & Li 2024), sign-GD, Muon <-> spectral norm). A transfer counts only if it yields a new, checkable statement here -- e.g. the l_inf programme predicts a different feature count, testable against the AdamW K*+1 observation recorded in the memory.',
  },
  {
    key: 'bridge', title: 'Theory -> trained networks',
    prompt: 'Turn the exact optimum into sharp, zero-parameter predictions about trained networks that can be checked on EXISTING artifacts (heracles/FINAL_RUNS, runs_torch, legacy/runs, nudial): which wrong answers get the highest logits (the programme\'s tight offsets), per-frequency share ratios, neuron phase designs, per-frequency ablation effects, final weight norm from margin, the efficiency gamma / ceiling decomposed into share error / count error / junk / cross-frequency terms, K*/log p across primes. Do a shallow reconnaissance on 2-3 real networks before committing, so predictions are anchored in what the artifacts actually contain.',
  },
  {
    key: 'redteam', title: 'Red team / counterexamples',
    prompt: 'Try to break the paper. Hunt for where the theorems\' hypotheses silently fail, where a Theta-law misleads at the primes actually used, where an empirical claim rests on an instrument artefact, and where a hostile ICLR reviewer will attack (product model vs ReLU / transformers, the junk-free restriction, loose constants, trained nets not at the optimum, the Ding width law, AdamW and l_inf). Each hypothesis is a precise attack with a preregistered test that either lands (then say exactly how the paper must change) or fails (then the paper can cite the failed attack as robustness).',
  },
  {
    key: 'scope', title: 'Generalisation and scope',
    prompt: 'Map exactly how far the machinery extends: other finite groups and tasks (modular multiplication via the discrete log to Z_{p-1}, subtraction, n-operand sums, x^n and polynomial tasks, non-abelian groups beyond S5), other degrees nu and penalties b (including the b -> nu endpoint), and other architectures (which ones satisfy the budget (B) and sign-rule (S) assumptions). For each extension state what transfers exactly, what transfers up to constants, and what fails -- with a proof or an exact computation for each claim. New falsifiable count predictions for tasks people actually train are the prize.',
  },
  {
    key: 'normative', title: 'Elegance / normative',
    prompt: 'Reason as the optimiser: what SHOULD the optimal solution look like -- share allocation, phase arrangement, redundancy, width, symmetry, readout -- given the exact optimum? Compute the ideal properties from the math first, then check which ones real trained networks satisfy and which they conspicuously violate. Both outcomes are findings if sharp and quantitative.',
  },
  {
    key: 'dynamics', title: 'Dynamics against the ceiling',
    prompt: 'Use the exact ceiling as an absolute yardstick for training trajectories in existing runs: normalised margin gamma(t) / ceiling as a progress measure, when and how networks approach or stall below the class optimum, whether frequency count and shares converge to the programme values, how weight decay, data fraction and optimiser change the approach. Look at real trajectories (checkpoints, spectra) before hypothesising; deliver phenomenon + causal hypothesis + discriminating test.',
  },
  {
    key: 'positioning', title: 'Positioning / reviewer gaps',
    prompt: 'Read the nearest prior work (literature ledgers) and identify precisely what each paper left open that our machinery closes, and what each will make a reviewer demand from us. Turn the strongest gaps into concrete provable or checkable items (e.g. "we prove McCracken\'s log-n observation", "Morwani\'s dense result is exactly the b = nu endpoint"), each with a preregistered check that the claimed relationship is exactly right -- quote the prior paper\'s actual statement, not its abstract.',
  },
]

// ---------------------------------------------------------------- hypothesis + referee

const HYP_SCHEMA = {
  type: 'object', required: ['angle_summary', 'hypotheses', 'discarded'],
  properties: {
    angle_summary: { type: 'string', description: 'One paragraph: how you approached this angle and what reconnaissance showed' },
    hypotheses: {
      type: 'array', minItems: 1, maxItems: MAX_HYP,
      items: {
        type: 'object',
        required: ['id', 'kind', 'statement', 'preregistered_prediction', 'falsifier', 'test_plan', 'nearest_prior_work', 'novelty_notes', 'why_surprising', 'paper_value'],
        properties: {
          id: { type: 'string', description: 'short slug, e.g. derive-1' },
          kind: { type: 'string', enum: ['theorem', 'empirical', 'computation'] },
          statement: { type: 'string', description: 'The claim, precisely stated with all hypotheses and scope' },
          preregistered_prediction: { type: 'string', description: 'The exact outcome expected if true (numbers, patterns, or the proved statement), committed before any test' },
          falsifier: { type: 'string', description: 'The exact outcome that kills it' },
          test_plan: { type: 'string', description: 'Executable plan: files, quantities, statistic, null, threshold; for theorems the proof strategy and the small-case numerical check' },
          nearest_prior_work: { type: 'string', description: 'Closest paper(s) with exact citation and what they prove' },
          novelty_notes: { type: 'string', description: 'What the searches found; why this is not already known (literature or known territory)' },
          why_surprising: { type: 'string', description: 'Why an ICLR reviewer would find this non-obvious' },
          paper_value: { type: 'string', description: 'Where it would go in the paper and what it would change' },
        },
      },
    },
    discarded: { type: 'array', items: { type: 'string' }, description: 'Candidates killed in self-criticism, one line each with the reason' },
  },
}

function hypPrompt(a, round, scope, explored) {
  return BRIEF + scopeBlock(scope) + `
# Your task: hypothesis generation -- angle "${a.title}" (round ${round})
${a.prompt}
${explored.length ? '\n## Already explored in earlier rounds (do NOT repeat; build past them)\n' + explored.map((s) => '- ' + s).join('\n') + '\n' : ''}
## Process (do all of it)
1. Read the scope ledgers and the repo files your angle needs. Anchor in real artifacts or exact computations where relevant (shallow reconnaissance only; heavy testing belongs to the tester).
2. Generate 6-10 candidates.
3. Criticise them as a hostile reviewer -- kill anything that (a) is already in the literature or the known territory, (b) cannot be decided under this run's rules (training, >10 min compute, a proof that would take more than a day), (c) has no sharp falsifier, (d) would not interest an ICLR reviewer, (e) repeats an explored item.
4. Novelty-check every survivor with WebSearch; when a paper is close, read its actual statement.
5. Keep the best 1-${MAX_HYP}. Preregister each: exact prediction, exact falsifier, and a test plan concrete enough for another agent to execute without asking.
6. Write full notes (all candidates, kills, reasoning) to ${OUT}/r${round}/${a.key}/HYPOTHESES.md.
Your output goes verbatim to a hostile referee and then to a tester.`
}

const REFEREE_SCHEMA = {
  type: 'object', required: ['decisions'],
  properties: {
    decisions: {
      type: 'array',
      items: {
        type: 'object', required: ['id', 'decision', 'reasons'],
        properties: {
          id: { type: 'string' },
          decision: { type: 'string', enum: ['proceed', 'tighten', 'kill'] },
          reasons: { type: 'string' },
          tightened_prediction: { type: 'string', description: 'Only for tighten: the stricter prediction (never looser)' },
          tightened_test_plan: { type: 'string', description: 'Only for tighten: the corrected, better-controlled plan' },
          required_controls: { type: 'string', description: 'Extra nulls / controls / checks the tester must run' },
        },
      },
    },
    notes: { type: 'string' },
  },
}

function refereePrompt(a, round, hyp) {
  return BRIEF + `
# Your task: hostile referee of preregistrations -- angle "${a.title}" (round ${round})
Decide each hypothesis below BEFORE any test runs: proceed, tighten, or kill.
- Kill if: already known (cite it -- search yourself with WebSearch, do not trust the author's novelty notes), unfalsifiable, undecidable under the rules, circular (the prediction is fitted on the data that tests it), trivial, or a re-proposal of a dead route without a new idea.
- Tighten if the idea is sound but the prediction is vague, the null is weak, the statistic is wrong, a control is missing, or a theorem's scope is overstated. You may only make predictions STRICTER and add controls -- never loosen.
- Proceed otherwise.
Check each test plan is executable with the stated resources. Write your reasoning to ${OUT}/r${round}/${a.key}/REFEREE.md.

## Preregistrations (verbatim)
${JSON.stringify(hyp, null, 2)}`
}

function applyReferee(a, round, hyp, ref) {
  const dec = new Map(((ref && ref.decisions) || []).map((d) => [d.id, d]))
  return hyp.hypotheses.map((h, i) => {
    const d = dec.get(h.id) || { decision: 'proceed', reasons: 'referee gave no decision for this id (fail-open)' }
    const f = Object.assign({}, h, { uid: `r${round}-${a.key}-${slug(h.id)}-${i}`, angle: a.key, round, referee: d })
    if (d.decision === 'tighten') {
      if (d.tightened_prediction) f.preregistered_prediction = d.tightened_prediction
      if (d.tightened_test_plan) f.test_plan = d.tightened_test_plan
    }
    if (d.required_controls) f.required_controls = d.required_controls
    return f
  })
}

const DEDUP_SCHEMA = {
  type: 'object', required: ['keep', 'dropped'],
  properties: {
    keep: { type: 'array', items: { type: 'object', required: ['uid'], properties: { uid: { type: 'string' }, absorbs: { type: 'array', items: { type: 'string' } }, note: { type: 'string' } } } },
    dropped: { type: 'array', items: { type: 'object', required: ['uid', 'duplicate_of', 'reason'], properties: { uid: { type: 'string' }, duplicate_of: { type: 'string' }, reason: { type: 'string' } } } },
  },
}

function dedupPrompt(round, live, tested) {
  const rows = live.map((h) => ({ uid: h.uid, kind: h.kind, statement: h.statement, prediction: h.preregistered_prediction, controls: h.required_controls || '' }))
  return `# Task: cross-angle dedup before testing (round ${round})
Below are refereed hypotheses from several independent angles. Group TRUE duplicates -- same claim, or decided by the same test. In each group keep the one with the sharpest prediction and strongest controls (list the others under its "absorbs") and drop the rest with duplicate_of set. Also drop anything that duplicates an item already tested in an earlier round. Do NOT drop merely related hypotheses, and do not edit any hypothesis. Every uid must appear exactly once, in keep or in dropped.

## Already tested in earlier rounds
${tested.length ? tested.map((s) => '- ' + clip(s, 300)).join('\n') : '(none)'}

## Candidates
${JSON.stringify(rows, null, 2)}`
}

// ---------------------------------------------------------------- test + verify

const TEST_SCHEMA = {
  type: 'object', required: ['uid', 'verdict', 'evidence', 'caveats', 'report_path'],
  properties: {
    uid: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'PARTIAL', 'REFUTED', 'INCONCLUSIVE'] },
    evidence: { type: 'string', description: 'Key numbers (effect sizes, n, nulls, p-values) or the proof skeleton (lemma chain, where each is proved). <= 400 words' },
    caveats: { type: 'string', description: 'Everything a hostile reviewer would attack; scope limits' },
    deviations: { type: 'string', description: 'Any choice not fixed by the preregistration, and why' },
    report_path: { type: 'string', description: 'Absolute path of REPORT.md (PROOF.md sits next to it for theorems)' },
    scripts: { type: 'array', items: { type: 'string' } },
    surprises: { type: 'string', description: 'Unplanned anomalies, or "none"' },
  },
}

function itemDir(h) { return `${OUT}/r${h.round}/${h.angle}/${slug(h.id)}` }

function testPrompt(h) {
  return BRIEF + `
# Your task: decide ONE preregistered hypothesis (${h.uid})
Implement the test plan faithfully and judge ONLY against the preregistered prediction and falsifier. You may not modify the hypothesis. If the plan is underspecified, make the most natural choice and record it under deviations. Run every control the referee required.
Work directory: ${itemDir(h)}/ -- write REPORT.md there (and PROOF.md for theorems).
- THEOREM: write the complete proof in PROOF.md (every step justified; imported results cited precisely), then an independent numerical check on exact small cases. CONFIRMED only if the proof is complete AND the check passes; PARTIAL if proved only under extra hypotheses (state them) or only in part; REFUTED if you find a counterexample (give it explicitly); INCONCLUSIVE otherwise.
- EMPIRICAL / COMPUTATION: include the null / baseline, effect size, n, uncertainty; prefer paired within-run comparisons; beware pseudoreplication. Before reporting CONFIRMED, try at least once to falsify it yourself (competing explanation, permutation null, robustness split).
Independent skeptics will re-derive and re-run your work, so make REPORT.md self-contained and every number traceable to a script.

## Final preregistration (after referee; verbatim)
${JSON.stringify(h, null, 2)}`
}

const NOVELTY_LENS = { key: 'novelty', text: 'Prior art: search hard (WebSearch / WebFetch, arXiv, standard references and textbooks) for this exact result or a trivial specialisation of a known one; also check the internal known territory. If it is known, upheld=false with severity fatal and an exact citation.' }
const LENSES = {
  theorem: [
    { key: 'proof', text: 'Proof audit: read PROOF.md line by line and re-derive every inequality yourself; check that each hypothesis is actually satisfied where it is used and that the stated scope matches what is proved. Find the gap if one exists.' },
    { key: 'counterexample', text: 'Counterexample hunt: write your OWN numerical search (do not reuse the tester scripts) over small and edge cases (smallest p, composite moduli if claimed, s near 1, K = 1, degenerate shares) trying to violate the statement.' },
    NOVELTY_LENS,
  ],
  empirical: [
    { key: 'repro', text: 'Independent reproduction: BEFORE opening the tester scripts, re-implement the analysis from the preregistration alone in your own directory; then compare numbers with the tester and explain every discrepancy.' },
    { key: 'stats', text: 'Statistics: nulls, effect sizes, n, pseudoreplication, multiple testing across this whole run (roughly 20-40 hypotheses), and post-hoc flexibility relative to the preregistration.' },
    { key: 'confound', text: 'Alternative explanations: construct the strongest competing mechanism or instrument artefact that would produce the same numbers, and test it if feasible.' },
    NOVELTY_LENS,
  ],
  computation: [
    { key: 'repro', text: 'Independent recomputation: BEFORE opening the tester scripts, recompute the key quantities with your own code (a different algorithm where possible) and compare.' },
    { key: 'certificate', text: 'Certificate audit: could the search or optimiser have missed the true optimum? Check certificates, dual bounds, tolerances, initialisations and grid coverage; try to find a better point.' },
    NOVELTY_LENS,
  ],
}

const VERDICT_SCHEMA = {
  type: 'object', required: ['upheld', 'severity', 'issues'],
  properties: {
    upheld: { type: 'boolean', description: 'true only if, through your lens, the reported verdict stands' },
    severity: { type: 'string', enum: ['none', 'minor', 'major', 'fatal'] },
    issues: { type: 'string', description: 'Concrete problems found (or "none"), with evidence' },
    independent_numbers: { type: 'string', description: 'Numbers you computed yourself and how they compare' },
  },
}

function verifyPrompt(h, t, lens) {
  return BRIEF + `
# Your task: adversarial verification -- lens "${lens.key}" -- ${h.uid}
${lens.text}
A tester reported verdict ${t.verdict}. Find out whether that verdict survives. Do the work yourself (re-derive, re-run, search); do not trust the tester's summary. If you cannot confirm it, upheld=false. Write your notes to ${itemDir(h)}/verify_${lens.key}/NOTES.md.

## Preregistration
${JSON.stringify(h, null, 2)}

## Tester's report (summary; full report at ${t.report_path})
${JSON.stringify(t, null, 2)}`
}

const KILL_SCHEMA = {
  type: 'object', required: ['sound', 'issues'],
  properties: {
    sound: { type: 'boolean', description: 'true if the refutation is correct' },
    issues: { type: 'string' },
    corrected_result: { type: 'string', description: 'If unsound and cheaply re-runnable: what the correct test gives' },
  },
}

function killPrompt(h, t) {
  return BRIEF + `
# Your task: audit a refutation -- ${h.uid}
A tester REFUTED this hypothesis. A buggy refutation kills a true result, so audit the kill: script bugs, wrong statistic, wrong run set, misread prediction, numerical tolerance, a "counterexample" that is not one. Re-run cheaply where you can. Write notes to ${itemDir(h)}/killaudit/NOTES.md.

## Preregistration
${JSON.stringify(h, null, 2)}

## Tester's report (summary; full report at ${t.report_path})
${JSON.stringify(t, null, 2)}`
}

async function testAndVerify(h) {
  const t = await agent(testPrompt(h), opts(`test:${h.uid}`, 'Test', EFFORT.test, TEST_SCHEMA))
  if (!t) { log(`${h.uid}: tester returned nothing`); return { h, t: null, status: 'NO_RESULT' } }
  if (t.verdict === 'CONFIRMED' || t.verdict === 'PARTIAL') {
    const lenses = LENSES[h.kind] || LENSES.empirical
    const vs = await parallel(lenses.map((l) => () =>
      agent(verifyPrompt(h, t, l), opts(`verify:${h.uid}:${l.key}`, 'Verify', EFFORT.verify, VERDICT_SCHEMA))
        .then((v) => (v ? Object.assign({ lens: l.key }, v) : null))))
    const got = vs.filter(Boolean)
    if (got.length < lenses.length) log(`${h.uid}: ${lenses.length - got.length} verifier(s) returned nothing (counted as not upheld)`)
    const fatal = got.some((v) => v.severity === 'fatal')
    const major = got.some((v) => v.severity === 'major')
    const up = got.filter((v) => v.upheld).length
    const status = fatal || up * 2 <= lenses.length ? 'KILLED' : (major ? 'WOUNDED' : 'SURVIVED')
    log(`${h.uid}: ${t.verdict} -> ${status} (${up}/${lenses.length} upheld${fatal ? ', fatal' : ''}${major ? ', major' : ''})`)
    return { h, t, status, verdicts: got }
  }
  if (t.verdict === 'REFUTED') {
    const k = await agent(killPrompt(h, t), opts(`killaudit:${h.uid}`, 'Verify', EFFORT.verify, KILL_SCHEMA))
    const status = k && k.sound === false ? 'REFUTATION_UNSOUND' : 'REFUTED'
    log(`${h.uid}: REFUTED -> ${status}`)
    return { h, t, status, kill: k }
  }
  log(`${h.uid}: INCONCLUSIVE`)
  return { h, t, status: 'INCONCLUSIVE' }
}

// ---------------------------------------------------------------- rounds

async function runRound(round, angles, scope, explored, tested) {
  const refereed = await pipeline(
    angles,
    (a) => agent(hypPrompt(a, round, scope, explored), opts(`hyp:r${round}:${a.key}`, 'Hypothesize', EFFORT.hyp, HYP_SCHEMA)),
    (hyp, a) => {
      if (!hyp || !hyp.hypotheses || !hyp.hypotheses.length) {
        log(`r${round} ${a.key}: no surviving hypotheses`)
        return { a, hyp, finals: [] }
      }
      return agent(refereePrompt(a, round, hyp), opts(`referee:r${round}:${a.key}`, 'Referee', EFFORT.referee, REFEREE_SCHEMA))
        .then((ref) => ({ a, hyp, ref, finals: applyReferee(a, round, hyp, ref) }))
    },
  )
  const rows = refereed.filter(Boolean)
  const all = rows.flatMap((r) => r.finals)
  const killedByReferee = all.filter((h) => h.referee.decision === 'kill')
  const live = all.filter((h) => h.referee.decision !== 'kill')
  log(`r${round}: ${all.length} preregistered, ${killedByReferee.length} killed by referees, ${live.length} live`)

  // Barrier: dedup needs every angle's live hypotheses at once, before the expensive test + verify stage.
  let toTest = live
  let dropped = []
  if (live.length > 1) {
    const dd = await agent(dedupPrompt(round, live, tested), opts(`dedup:r${round}`, 'Referee', EFFORT.dedup, DEDUP_SCHEMA))
    if (dd) {
      const known = new Set(live.map((h) => h.uid))
      const keep = new Set((dd.keep || []).map((k) => k.uid))
      const drop = new Set((dd.dropped || []).map((d) => d.uid).filter((u) => known.has(u) && !keep.has(u)))
      toTest = live.filter((h) => !drop.has(h.uid))
      dropped = (dd.dropped || []).filter((d) => drop.has(d.uid))
      if (dropped.length) log(`r${round}: dedup dropped ${dropped.length}: ${dropped.map((d) => `${d.uid} (dup of ${d.duplicate_of})`).join('; ')}`)
    } else {
      log(`r${round}: dedup agent returned nothing; testing all ${live.length}`)
    }
  }
  log(`r${round}: testing ${toTest.length} hypothesis(es)`)
  const results = (await parallel(toTest.map((h) => () => testAndVerify(h)))).filter(Boolean)
  return {
    round,
    angles: rows.map((r) => ({ key: r.a.key, title: r.a.title, summary: r.hyp && r.hyp.angle_summary, discarded: r.hyp && r.hyp.discarded })),
    allStatements: all.map((h) => clip(h.statement, 250)),
    killedByReferee, dropped, results,
  }
}

function compact(rounds) {
  return rounds.map((r) => ({
    round: r.round,
    angles: r.angles.map((a) => ({ key: a.key, title: a.title, summary: clip(a.summary, 600) })),
    referee_kills: r.killedByReferee.map((h) => ({ uid: h.uid, statement: clip(h.statement, 300), reason: clip(h.referee.reasons, 400) })),
    dedup_drops: r.dropped,
    results: r.results.map((x) => ({
      uid: x.h.uid, kind: x.h.kind, angle: x.h.angle,
      statement: clip(x.h.statement, 700), prediction: clip(x.h.preregistered_prediction, 600),
      verdict: x.t ? x.t.verdict : null, status: x.status,
      evidence: x.t ? clip(x.t.evidence, 1500) : null,
      caveats: x.t ? clip(x.t.caveats, 700) : null,
      surprises: x.t ? clip(x.t.surprises, 500) : null,
      report: x.t ? x.t.report_path : null,
      checks: (x.verdicts || []).map((v) => ({ lens: v.lens, upheld: v.upheld, severity: v.severity, issues: clip(v.issues, 600) })),
      kill_audit: x.kill ? { sound: x.kill.sound, issues: clip(x.kill.issues, 600), corrected: clip(x.kill.corrected_result, 400) } : undefined,
    })),
  }))
}

const CRITIC_SCHEMA = {
  type: 'object', required: ['dry', 'missing', 'directions'],
  properties: {
    dry: { type: 'boolean', description: 'true if nothing important remains that a further round could decide' },
    missing: { type: 'array', items: { type: 'string' }, description: 'Modalities not run, claims unverified, sources unread, obvious follow-ups of survivors, unsound refutations to redo' },
    directions: {
      type: 'array', maxItems: Math.max(1, MAX_NEW),
      items: {
        type: 'object', required: ['key', 'title', 'prompt', 'why'],
        properties: {
          key: { type: 'string', description: 'short slug' },
          title: { type: 'string' },
          prompt: { type: 'string', description: 'A full angle prompt in the style of the original angles: what to attack, how, what counts as a finding' },
          why: { type: 'string' },
        },
      },
    },
  },
}

function criticPrompt(rec, round, isLast) {
  return BRIEF + `
# Your task: completeness critic after round ${round}
The compact record so far is below (full reports under ${OUT}/). Ask what is missing: modalities not run, claims left unverified, sources unread, the sharpest consequences of SURVIVED results, WOUNDED / PARTIAL results one more push could settle, REFUTATION_UNSOUND items to redo, and high-value gaps no angle covered.
Propose up to ${Math.max(1, MAX_NEW)} new directions, each a full angle prompt. Priority: (1) upgrading WOUNDED / PARTIAL to SURVIVED, (2) redoing unsound refutations, (3) uncovered high-value gaps, (4) consequences of survivors. Set dry=true only if nothing important remains.
${isLast ? 'No further round will run: your directions and missing list are advisory input for the final synthesis.' : ''}

## Record
${JSON.stringify(rec, null, 2)}`
}

// ---------------------------------------------------------------- synthesis

function synthPrompt(rec, critic) {
  return BRIEF + `
# Your task: final synthesis
The compact record is below; every item has REPORT.md / PROOF.md / verifier NOTES.md under ${OUT}/ -- open them whenever the summary is not enough. Write ${OUT}/SYNTHESIS.md:
1. Headline: is there anything paper-worthy? One honest paragraph.
2. SURVIVED results ranked by ICLR value: plain-language statement, precise statement with scope, evidence (numbers / proof status), novelty vs literature and known territory, and where it goes in the paper (section + a ready-to-paste claim sentence with its limits).
3. WOUNDED / PARTIAL: exactly what is missing and the cheapest fix.
4. Deaths: refuted, killed by verifiers, killed by referees -- and what each teaches. List REFUTATION_UNSOUND items separately.
5. Cross-angle connections and contradictions.
6. Surprises registry, triaged by follow-up value.
7. The critic's missing list and ranked next steps given the paper deadline (2026-09-26 17:30 IST): what fits before it, what is post-submission.
Deflate: many hypotheses were tested (multiple testing), verifiers can share blind spots, and a tester's CONFIRMED is not a result until verification survives. Explain every formula in plain words. Return the synthesis text.

## Critic after the last round
${JSON.stringify(critic || {}, null, 2)}

## Record
${JSON.stringify(rec, null, 2)}`
}

function auditPrompt() {
  return BRIEF + `
# Your task: overclaim audit of the synthesis
Read ${OUT}/SYNTHESIS.md. Check EVERY factual claim in it against the underlying files (REPORT.md, PROOF.md, verifier NOTES.md, scope ledgers) -- not against the synthesis's own summaries. Flag: numbers that do not match, results described beyond their proved scope, novelty claims the novelty verifier did not back, WOUNDED items presented as SURVIVED, missing caveats, paper sentences that overclaim. Then write ${OUT}/SYNTHESIS_AUDITED.md: the corrected full synthesis followed by an "Audit log" of every change and why. Return the corrected synthesis text.`
}

// ---------------------------------------------------------------- main

const extra = Array.isArray(A.extraAngles)
  ? A.extraAngles.filter((x) => x && x.key && x.prompt).map((x) => ({ key: slug(x.key), title: x.title || x.key, prompt: x.prompt }))
  : []
const base = Array.isArray(A.angles) && A.angles.length ? ANGLES.filter((a) => A.angles.includes(a.key)) : ANGLES
const FIRST = base.concat(extra)
if (Array.isArray(A.angles) && A.angles.length && base.length < A.angles.length) {
  log(`Unknown angle keys ignored: ${A.angles.filter((k) => !ANGLES.some((a) => a.key === k)).join(', ')}`)
}

log(`Research run "${TAG}" on ${MODEL}: ${FIRST.length} angle(s) [${FIRST.map((a) => a.key).join(', ')}], up to ${ROUNDS} round(s), <=${MAX_HYP} hypotheses/angle, training ${ALLOW_TRAIN ? 'ALLOWED' : 'forbidden'}; output ${OUT}`)

phase('Scope')
const scopeRes = await parallel([
  () => agent(ledgerPrompt(), opts('scope:ledger', 'Scope', EFFORT.scope, LEDGER_SCHEMA)),
  ...LIT_MODES.map((m) => () => agent(litPrompt(m), opts(`scope:lit-${m.key}`, 'Scope', EFFORT.scope, LIT_SCHEMA))),
])
const ledger = scopeRes[0]
const lits = scopeRes.slice(1).filter(Boolean)
if (!ledger) log('WARNING: ledger agent returned nothing; hypothesis agents fall back to MEMORY.md and the repo docs')
if (lits.length < LIT_MODES.length) log(`WARNING: ${LIT_MODES.length - lits.length} literature sweep(s) returned nothing`)
const scope = {
  ledger: ledger ? ledger.summary : '(ledger unavailable -- read MEMORY.md and the heracles docs directly)',
  ledgerPath: ledger ? ledger.path : `${OUT}/scope/LEDGER.md`,
  lits: lits.map((l) => l.summary + '\nNearest: ' + (l.nearest || []).map((n) => `${n.citation} -- ${n.claim} [overlap: ${n.overlap}]`).join('; ')),
  litPaths: lits.map((l) => l.path),
}

const rounds = []
let angles = FIRST
let critic = null
for (let round = 1; round <= ROUNDS; round++) {
  if (!angles.length) break
  if (budget.total && budget.remaining() < 400000) {
    log(`Budget low (${Math.round(budget.remaining() / 1000)}k left): skipping round ${round} and everything after it`)
    break
  }
  const explored = rounds.flatMap((r) => r.allStatements)
  const tested = rounds.flatMap((r) => r.results.map((x) => x.h.statement))
  log(`Round ${round}: ${angles.length} angle(s): ${angles.map((a) => a.key).join(', ')}`)
  const r = await runRound(round, angles, scope, explored, tested)
  rounds.push(r)
  const tally = {}
  r.results.forEach((x) => { tally[x.status] = (tally[x.status] || 0) + 1 })
  log(`Round ${round} done: ${Object.entries(tally).map(([k, v]) => `${k} ${v}`).join(', ') || 'no results'}; referee kills ${r.killedByReferee.length}, dedup drops ${r.dropped.length}`)

  critic = await agent(criticPrompt(compact(rounds), round, round === ROUNDS), opts(`critic:r${round}`, 'Critique', EFFORT.critic, CRITIC_SCHEMA))
  if (round === ROUNDS) break
  if (!critic || critic.dry || !critic.directions || !critic.directions.length) { log('Critic: dry -- no further round'); break }
  if (MAX_NEW === 0) { log('maxNewDirections = 0: critic directions recorded, no further round'); break }
  angles = critic.directions.slice(0, MAX_NEW).map((d) => ({ key: `x${round + 1}-${slug(d.key)}`, title: d.title, prompt: d.prompt }))
  if (critic.directions.length > angles.length) log(`Critic proposed ${critic.directions.length} directions; running ${angles.length} (maxNewDirections cap)`)
}

phase('Synthesize')
const rec = compact(rounds)
const synthesis = await agent(synthPrompt(rec, critic), opts('synthesis', 'Synthesize', EFFORT.synth))
let final = null
if (synthesis) {
  final = await agent(auditPrompt(), opts('audit', 'Synthesize', EFFORT.synth))
  if (!final) log('WARNING: audit agent returned nothing; the unaudited synthesis is in SYNTHESIS.md')
} else {
  log('WARNING: synthesis agent returned nothing; the record is returned raw')
}

return { final, synthesis, outDir: OUT, critic, scope, record: rec }
