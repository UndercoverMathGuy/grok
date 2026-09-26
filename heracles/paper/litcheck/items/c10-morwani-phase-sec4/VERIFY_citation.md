# VERIFY (citation lens): C10-morwani-phase-sec4, task 84

Claim (approach.tex:36): the margin is largest when each beta_j is real and positive, so each frequency contributes a cosine peaked at the label, "the analogue of the phase condition of \citet[Section~4]{morwani2024feature}".

## What I fetched myself
- arXiv abs page, 2311.07568. Title: "Feature emergence via margin maximization: case studies in algebraic tasks". Authors: Morwani, Edelman, Oncescu, Zhao, Kakade. Comments: "Accepted as Spotlight at ICLR 2024". Versions: v1 (13 Nov 2023) and v2 (19 Feb 2024).
- The arXiv HTML of v2 (curl, then text extraction). Sections: 1 Introduction, 2 Preliminaries, 3 Theoretical Approach (3.1, 3.2, 3.3 Blueprint), 4 Cyclic groups (modular addition), 5 Sparse parity, 6 Finite Groups with Real Representations, 7 Discussion.
- Theorem 7 is in Section 4 and is quoted from the extracted text: "Any network achieving this margin satisfies ... u(a) = λ cos(θ_u* + 2πζa/p), v(b) = λ cos(θ_v* + 2πζb/p), w(c) = λ cos(θ_w* + 2πζc/p) for some phase offsets θ_u*, θ_v*, θ_w* ∈ ℝ satisfying θ_u* + θ_v* = θ_w*." The theorem assumes m ≥ 4(p−1) and gives γ* = sqrt(2/27)/(p^{1/2}(p−1)).
- η_{u,v,w}(δ) := E_{a,b}[(u(a)+v(b))^2 w(a+b−δ)] is defined in Appendix F.1, which proves Theorem 7. approach.tex:11 cites it without a section number. That is fine.

## Direction check
The cross term u(a)v(b) contains cos(θ_u+θ_v+2πζ(a+b)/p). Pairing it with w(c) = cos(θ_w+2πζc/p) under θ_u+θ_v = θ_w leaves a relative phase of 2πζ(a+b−c)/p with no offset. So the frequency-ζ contribution is a cosine in (a+b−c), peaked at c = a+b, the label. This is the same direction as the paper's "β_j real and positive". Morwani's condition is per neuron, on weight phases. The paper's is per network frequency, on the complex coefficient. "Analogue" is the right word.

## Bib
refs.bib matches: title, authors, ICLR 2024, arXiv note. No changes needed. I did not fetch OpenReview because the web-search budget was exhausted (see below), so the bib has no OpenReview URL. That is optional.

## Verdict
SUPPORTED, and the checker's verdict and its confirmed citation are upheld. Section 4 is correct. Citing Theorem 7 would be more precise, as the checker recommended. Optionally write the condition out as θ_u*+θ_v*=θ_w*.

## Operational note (workflow speed)
My WebSearch call returned: "this session has used its web search budget (200 of 200 WebSearch calls)". The litcheck workflow's verifiers appear to share one session-wide cap (CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION = 200). Every verifier that starts after the cap is reached cannot search. It has to fall back to WebFetch or curl, or it stalls. This probably contributes to the slow and degraded verify stage.
