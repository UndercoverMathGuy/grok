export const meta = {
  name: 'paper-litcheck-opus55',
  description: 'Literature check of every claim in the ICLR paper on Opus 5.5: extract claims, 3-modality sweeps, per-claim prior-art/citation checks, adversarial verification, verified bib + citation plan',
  whenToUse: 'Novelty and citation audit of heracles/paper/iclr2027 before submission',
  phases: [
    { title: 'Extract', detail: 'claim inventory + 3 literature sweeps (ML, math, citation chasing)', model: 'claude-opus-5-5' },
    { title: 'Check', detail: 'one agent per claim: prior art, citation support, recommended citations', model: 'claude-opus-5-5' },
    { title: 'Verify', detail: 'adversarial re-search for novelty claims; fetch-and-confirm every citation', model: 'claude-opus-5-5' },
    { title: 'Synthesize', detail: 'LITCHECK.md, verified bib additions, citation insertion plan, then an audit', model: 'claude-opus-5-5' },
  ],
}

const MODEL = 'claude-opus-5-5'
const PAPER = '/Users/ruhaanrajadhyaksha/projects/grok/heracles/paper/iclr2027'
const OUT = '/Users/ruhaanrajadhyaksha/projects/grok/heracles/paper/litcheck'
const E = { extract: 'medium', sweep: 'medium', check: 'medium', verify: 'xhigh', synth: 'xhigh' }
const o = (label, phase, effort, schema) => { const x = { label, phase, model: MODEL, effort }; if (schema) x.schema = schema; return x }
const slug = (s) => String(s || 'x').toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'x'
const clip = (s, n) => { const t = typeof s === 'string' ? s : JSON.stringify(s ?? ''); return t.length > n ? t.slice(0, n) + ' ...[truncated]' : t }

const BRIEF = `
## Context
You are part of a literature audit for an ICLR 2027 submission, "Explaining Feature Sparsity Through the Margin-Regularization Trade-off" (due 2026-09-26 17:30 IST). The sources are in ${PAPER}: paper.tex and sections/{intro,prelim,approach,cyclic,groups,experiments,discussion,appendix}.tex; the bibliography is refs.bib.

The paper studies maximum L_{2,b}-margin one-hidden-layer networks (product/multilinear and power/polynomial neurons) on modular addition and group composition. It reduces the margin to a budget-splitting programme Gamma_s(p) over Fourier frequencies (s = nu/b), proves an exact optimum in an interference-free class (the nu-block), and proves structural results:
- finitely many optimal shapes in s;
- bottleneck offsets;
- a feature-ablation bound;
- logarithmic scaling: Gamma_s(p) = Theta((log p)^{1-s}), K* = Theta(log p), and gamma* = Theta(p^{-nu/2}(log p)^{1-nu/b}) for all networks.

## Rules
- Use WebSearch and WebFetch (load them via ToolSearch "select:WebSearch,WebFetch"). Prefer primary sources: arXiv abstract/HTML pages, DOIs, OpenReview, proceedings pages. The Semantic Scholar API (api.semanticscholar.org/graph/v1/paper/search?query=...) is useful for citation chasing.
- NEVER invent a citation. Every reference you recommend must come from a page you actually fetched this session. Record its URL, and confirm the title, authors, year and venue from that page. If you could not confirm it, say "unconfirmed" and do not emit a bib entry.
- When you claim a paper proves something, quote or closely paraphrase the actual statement and give its theorem/section number. An abstract is not enough when the overlap is close.
- Be adversarial about novelty. Our results count as "new" only if a real search failed to find them. Search adjacent fields too: harmonic analysis, additive combinatorics, coding/sidelobe design, sparse approximation, implicit-bias theory.
- Write your notes under ${OUT}/ in your own subdirectory. Do not edit the paper files.
`

const ITEMS_SCHEMA = {
  type: 'object', required: ['items', 'cite_usages'],
  properties: {
    items: {
      type: 'array', minItems: 1, maxItems: 70,
      items: {
        type: 'object', required: ['id', 'kind', 'location', 'statement', 'what_to_check'],
        properties: {
          id: { type: 'string', description: 'short unique slug' },
          kind: { type: 'string', enum: ['our_result', 'our_technique', 'framing_claim', 'cited_claim', 'uncited_background'] },
          location: { type: 'string', description: 'file + LaTeX label or a quoted phrase that locates it' },
          statement: { type: 'string', description: 'the claim, precisely' },
          current_citations: { type: 'array', items: { type: 'string' }, description: 'bib keys cited for it, if any' },
          what_to_check: { type: 'string' },
        },
      },
    },
    cite_usages: { type: 'integer', description: 'total number of \\cite commands in the paper' },
  },
}

const LIT_SCHEMA = {
  type: 'object', required: ['summary', 'papers'],
  properties: {
    summary: { type: 'string', description: '<= 500 words; lead with anything that already proves or nearly proves one of the paper\'s results' },
    papers: {
      type: 'array', maxItems: 25,
      items: {
        type: 'object', required: ['citation', 'url', 'claim', 'relevance'],
        properties: { citation: { type: 'string' }, url: { type: 'string' }, claim: { type: 'string' }, relevance: { type: 'string' } },
      },
    },
  },
}

const LIT_MODES = [
  { key: 'ml', text: 'ML literature 2019-2026 (including 2025-2026 arXiv): grokking and modular arithmetic (mechanistic and theoretical), max-margin and implicit bias of homogeneous networks (and of specific optimisers), feature counts, sparsity, superposition, representation cost of weight decay, group-composition networks and irreps, polynomial/multilinear/tensor networks.' },
  { key: 'math', text: 'Mathematics relevant to the programme and proofs: the Chowla cosine problem and minima of cosine sums, weighted coverings of Z_p by cosines, random Cayley graphs and expanders, positivity of Fourier coefficients of exp(z cos) and Bessel-function bounds, concave minimisation over polyhedra (vertex optimality), exponential sums / generalised Descartes rule of signs, sparse approximation with l_r (r<1) budgets, and AM-GM / elementary symmetric bounds of the kind used in the cost-of-a-feature lemma.' },
  { key: 'cite', text: 'Citation chasing: forward and backward citations of Morwani et al. (feature emergence via margin maximization), Nanda et al. (progress measures), Chughtai et al. (toy model of universality), McCracken et al. 2025, Gromov (grokking modular arithmetic), Wei et al. 2019 (regularization matters), Lyu & Li 2020, and any 2024-2026 paper on feature counts/sparsity in modular arithmetic or group tasks.' },
]

function extractPrompt() {
  return BRIEF + `
# Task: claim inventory
Read the ENTIRE paper, every section and the whole appendix, plus refs.bib. List every item that needs a literature check:
- our_result: every theorem, lemma, proposition and corollary presented as ours;
- our_technique: proof ideas or constructions that may be known (e.g. vertex optimality of concave minimisation, the positive-Fourier trick, random frequency sets, the Hadamard/Gram determinant argument, nu-th roots of unity phase blocks);
- framing_claim: statements about the state of the field ("trained networks use a handful of frequencies", "the count moves from dense to sparse as the degree changes", "not a restatement of...");
- cited_claim: every sentence that carries a \\cite. Group the sentences citing the same source for the same point into one item.
- uncited_background: statements that should carry a citation but do not.

Be exhaustive, but merge duplicates. Write the full inventory to ${OUT}/extract/INVENTORY.md.`
}

function sweepPrompt(m) {
  return BRIEF + `
# Task: literature sweep (${m.key})
${m.text}
Read the paper first so you know what to look for. Write ${OUT}/sweep/LIT_${m.key}.md with full references and URLs.`
}

const CHECK_SCHEMA = {
  type: 'object', required: ['id', 'verdict', 'evidence', 'recommended_citations', 'recommended_change'],
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['KNOWN', 'PARTIALLY_KNOWN', 'RELATED_ONLY', 'NOVEL_AS_FAR_AS_FOUND', 'SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_SUPPORTED', 'WRONG_CITATION', 'CITATION_FOUND', 'NO_CITATION_FOUND'] },
    evidence: { type: 'string', description: 'what you found: papers, exact statements / theorem numbers, URLs; the search queries you ran' },
    recommended_citations: {
      type: 'array',
      items: {
        type: 'object', required: ['key', 'bibtex', 'url', 'confirmed', 'why'],
        properties: {
          key: { type: 'string' }, bibtex: { type: 'string' }, url: { type: 'string' },
          confirmed: { type: 'boolean', description: 'true only if you fetched the page and confirmed the title/authors/year/venue' },
          why: { type: 'string', description: 'which sentence it supports, and what exactly it proves' },
        },
      },
    },
    recommended_change: { type: 'string', description: 'concrete edit to the paper text (or "none")' },
  },
}

function checkPrompt(it, sweeps) {
  const guide = {
    our_result: 'Search hard for prior art on this exact result or a trivial specialisation of it. Verdict KNOWN / PARTIALLY_KNOWN / RELATED_ONLY / NOVEL_AS_FAR_AS_FOUND. Recommend citations for anything related that reviewers will expect.',
    our_technique: 'Find where this technique is standard or originated, and recommend the canonical citation. Verdict KNOWN / PARTIALLY_KNOWN / RELATED_ONLY / NOVEL_AS_FAR_AS_FOUND.',
    framing_claim: 'Check whether the literature supports the framing claim, and find the citations that back it (or contradict it). Use verdict SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED, or CITATION_FOUND / NO_CITATION_FOUND.',
    cited_claim: 'Fetch each cited source and check that it actually says what the sentence attributes to it, including theorem numbers, conditions and the direction of the claim. Verdict SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / WRONG_CITATION. Give the fix.',
    uncited_background: 'Find the right citation(s). Verdict CITATION_FOUND / NO_CITATION_FOUND.',
  }[it.kind]
  return BRIEF + `
## Literature sweep summaries (starting points only; do your own search)
${sweeps}

# Task: check ONE item (${it.id}, kind ${it.kind})
${guide}
Write notes to ${OUT}/items/${slug(it.id)}/CHECK.md.

## Item
${JSON.stringify(it, null, 2)}`
}

const VERIFY_SCHEMA = {
  type: 'object', required: ['upheld', 'issues'],
  properties: {
    upheld: { type: 'boolean', description: 'true only if the checker\'s verdict AND every confirmed citation survive your independent check' },
    corrected_verdict: { type: 'string' },
    issues: { type: 'string', description: 'prior art the checker missed, citations that do not exist or do not say what is claimed, wrong bib fields' },
    extra_citations: {
      type: 'array',
      items: { type: 'object', required: ['key', 'bibtex', 'url', 'why'], properties: { key: { type: 'string' }, bibtex: { type: 'string' }, url: { type: 'string' }, why: { type: 'string' } } },
    },
  },
}

const LENS = {
  novelty_ml: 'Adversarial novelty search, ML lens. Assume the result IS already in the ML / learning-theory literature and try to find it: different terminology, workshop papers, 2025-2026 arXiv, theses. Also verify every citation the checker confirmed by fetching it yourself.',
  novelty_math: 'Adversarial novelty search, mathematics lens. Assume the result or technique is a known fact in harmonic analysis, additive combinatorics, optimisation or probability, and try to find it (books, surveys, MathOverflow, classical papers). Also verify every citation the checker confirmed by fetching it yourself.',
  citation: 'Citation verification. Fetch every recommended and existing citation yourself. Confirm it exists, that the bib fields are right, and that it says what the sentence claims (theorem numbers, conditions, direction). Look for any better or more canonical citation.',
}

function verifyPrompt(it, chk, lens) {
  return BRIEF + `
# Task: independent verification -- lens "${lens}" -- item ${it.id}
${LENS[lens]}
Do the searching and fetching yourself; do not trust the checker's summary. Write notes to ${OUT}/items/${slug(it.id)}/VERIFY_${lens}.md.

## Item
${JSON.stringify(it, null, 2)}

## Checker's result
${JSON.stringify(chk, null, 2)}`
}

// ---------- run
log('Extracting the claim inventory and running 3 literature sweeps in parallel')
phase('Extract')
const first = await parallel([
  () => agent(extractPrompt(), o('extract', 'Extract', E.extract, ITEMS_SCHEMA)),
  ...LIT_MODES.map((m) => () => agent(sweepPrompt(m), o(`sweep:${m.key}`, 'Extract', E.sweep, LIT_SCHEMA))),
])
const inv = first[0]
const sweeps = first.slice(1).filter(Boolean)
if (!inv || !inv.items || !inv.items.length) { log('Extraction failed -- nothing to check'); return { error: 'no inventory' } }
if (sweeps.length < LIT_MODES.length) log(`WARNING: ${LIT_MODES.length - sweeps.length} sweep(s) returned nothing`)
const sweepText = sweeps.map((s, i) => `[${LIT_MODES[i] ? LIT_MODES[i].key : i}] ` + clip(s.summary, 2500) + '\nPapers: ' + (s.papers || []).map((p) => `${p.citation} <${p.url}> -- ${p.claim}`).join('; ')).join('\n\n')
const kinds = {}
inv.items.forEach((it) => { kinds[it.kind] = (kinds[it.kind] || 0) + 1 })
log(`Inventory: ${inv.items.length} items (${Object.entries(kinds).map(([k, v]) => `${k} ${v}`).join(', ')}); ${inv.cite_usages} \\cite usages`)

const results = await pipeline(
  inv.items,
  (it) => agent(checkPrompt(it, clip(sweepText, 9000)), o(`check:${slug(it.id)}`, 'Check', E.check, CHECK_SCHEMA)),
  (chk, it) => {
    if (!chk) { log(`${it.id}: checker returned nothing`); return { it, chk: null, ver: [] } }
    const lenses = (it.kind === 'our_result' || it.kind === 'our_technique') ? ['novelty_ml', 'novelty_math'] : ['citation']
    return parallel(lenses.map((l) => () => agent(verifyPrompt(it, chk, l), o(`verify:${slug(it.id)}:${l}`, 'Verify', E.verify, VERIFY_SCHEMA)).then((v) => v && Object.assign({ lens: l }, v))))
      .then((vs) => {
        const got = vs.filter(Boolean)
        const up = got.filter((v) => v.upheld).length
        log(`${it.id}: ${chk.verdict} -- ${up}/${lenses.length} verifier(s) upheld`)
        return { it, chk, ver: got }
      })
  },
)

const rec = results.filter(Boolean).map((r) => ({
  id: r.it.id, kind: r.it.kind, location: r.it.location, statement: clip(r.it.statement, 500),
  verdict: r.chk ? r.chk.verdict : null, evidence: r.chk ? clip(r.chk.evidence, 1500) : null,
  citations: r.chk ? (r.chk.recommended_citations || []).map((c) => ({ key: c.key, url: c.url, confirmed: c.confirmed, why: clip(c.why, 300) })) : [],
  change: r.chk ? clip(r.chk.recommended_change, 600) : null,
  verification: r.ver.map((v) => ({ lens: v.lens, upheld: v.upheld, corrected: v.corrected_verdict || null, issues: clip(v.issues, 800), extra: (v.extra_citations || []).map((c) => ({ key: c.key, url: c.url, why: clip(c.why, 200) })) })),
  notes: `${OUT}/items/${slug(r.it.id)}/`,
}))

phase('Synthesize')
const synth = await agent(BRIEF + `
# Task: synthesis
The full record of the per-item checks and verifications is below; notes are under ${OUT}/items/, and the sweeps are in ${OUT}/sweep/. Produce:
1. ${OUT}/LITCHECK.md:
   (a) MUST-FIX list: our results that are KNOWN or PARTIALLY_KNOWN (with exact prior statements), wrong or unsupported citations, and missing citations reviewers will flag.
   (b) A per-item table: item, location, final verdict (the checker's verdict as corrected by the verifiers; where they disagree, say so), citations.
   (c) A novelty statement for each of our results that survives, worded exactly as the paper can defensibly claim it.
   (d) A citation insertion plan: sentence (quoted, with file) -> bib keys.
   (e) A draft related-work paragraph (the intro has a TODO for one) that uses only verified citations.
2. ${OUT}/refs_additions.bib: ONLY entries that a checker or verifier confirmed by fetching the page, deduplicated against ${PAPER}/refs.bib (read it). Every entry carries a url or doi field. List every unconfirmed candidate separately in LITCHECK.md; do not put them in the bib.
Be conservative: when the evidence is mixed, say so.

## Record
${JSON.stringify(rec, null, 2)}`, o('synthesis', 'Synthesize', E.synth))

const audit = await agent(BRIEF + `
# Task: audit the synthesis
Read ${OUT}/LITCHECK.md and ${OUT}/refs_additions.bib. For EVERY bib entry: fetch its url/doi and confirm the title, authors, year and venue; delete any entry that fails, and log the deletion. For every "KNOWN / NOVEL" statement in LITCHECK.md, check it against the item notes under ${OUT}/items/. Fix errors in place, then append an "Audit log" section listing every change. Return a <= 300 word summary: the must-fix count, the novelty verdicts for our main results, the number of verified bib additions, and any deletions.`, o('audit', 'Synthesize', E.synth))

return { audit, synthesis: synth, items: rec.length, outDir: OUT }