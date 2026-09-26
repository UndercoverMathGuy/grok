# R3-phase-lemma (approach.tex lem:phase)

Statement: h(theta) <= sum_j Re(beta_j) kappa_j(delta) <= sum_j A_j kappa_j(delta), kappa_j(delta)=1-cos(2 pi j delta/p), for every theta and every delta != 0.
Proof in the paper: average Lemma avg at delta and -delta, then use the cosine spectrum of eta_theta (app:spectrum).

## Verdict: PARTIALLY_KNOWN
Each ingredient is known. The combination, stated as a rigorous upper bound for an ARBITRARY network with the class weight put on one offset, was not found.

## Evidence (all fetched 2026-09-25)
1. Morwani et al. ICLR 2024 (arXiv 2311.07568v2 HTML).
   - Sec 3.2 / Lemma 5 is the class-weighted margin framework.
   - Sec 4 proof outline uses UNIFORM class weighting tau(a,b)[c'] = 1/(p-1). The resulting quantity is E_{a,b}[psi'] = 2/((p-1)p^2) sum_{j!=0} u^(j) v^(j) w^(-j).
   - The phase condition theta_u* + theta_v* = theta_w* is stated in Theorem 7, which is in Section 4.
   - This settles the open point in LIT_cite.md: citing "Section 4" for the phase condition is correct.
   - Ours differs in one way: all the weight goes on the single offset delta, which gives the coverage kappa_j(delta) rather than a uniform sum. If the per-offset bound is averaged uniformly over delta != 0 it becomes (p/(p-1)) sum_j Re beta_j, which is Morwani's quantity (this is the step used in cor:dense). So our lemma strictly refines theirs.
2. Nanda et al. ICLR 2023. The logits of the "Fourier multiplication algorithm" are proportional to sum_k alpha_k cos(w_k(a+b-c)), so the separation at offset delta is sum alpha_k (1-cos(w_k delta)). This is an empirical/idealised form; there is no bound for arbitrary networks. (Known from the paper; not re-fetched here.)
3. McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266.
   - Thm 4.7 uses the idealised logit h(k) = sum_l cos(2 pi f_l (k-i-j)/n) and bounds h(i+j) - h(k) > delta m.
   - Here h(i+j) - h(k) is implicitly the equal-amplitude coverage sum_l (1-cos).
   - This is an idealised model, not an upper bound for arbitrary networks.
4. Kam, Bessafi, Cadet, arXiv 2609.17926 (15 Sep 2026, under review NeurReps 2026), App. A.3.2 Eq. (15): Delta_S(beta) = min_{s!=0} sum_k beta_k (1-cos(omega_k s)).
   - It is introduced as a DEFINITION or heuristic: "the logit separation between the correct class and the competitor at offset s is sum beta_k(1-cos(omega_k s))". The loss bound is L <= log(1+(p-1)e^{-Delta_S}).
   - There are no phases (amplitude only), no Re(beta) <= |beta| step, and no statement for arbitrary networks.
   - This is the closest statement of the "amplitude-coverage" quantity. It is concurrent work.
5. Not found: a lemma saying that the min-margin of ANY network (including one with interference and misaligned phases) is bounded by the amplitude-weighted coverage at every offset.

## What is new vs known
- Known: the averaging/class-weighting trick (Morwani), the phase-alignment condition (Morwani Thm 7), and the 1-cos separation for idealised aligned cosine logits (Nanda; McCracken Thm 4.7 model; Kam Eq 15).
- Ours: a rigorous per-offset upper bound valid for all theta, with phase misalignment paid through Re(beta) <= |beta| and interference averaged away. It is elementary. Present it as an adaptation, not a headline.

## Recommended change
The sentence after lem:phase currently reads: "(the analogue of the phase condition of \citet[Section~4]{morwani2024feature})".
Replace it with: "(the analogue of the phase condition of \citet[Theorem~7, Section~4]{morwani2024feature}; averaging the bound uniformly over $\delta\neq0$ recovers their uniformly class-weighted quantity). For idealized phase-aligned cosine logits the right-hand side is the familiar logit separation $\sum_jA_j(1-\cos(2\pi j\delta/p))$ \citep{nanda2023progress,mccracken2025}; concurrently, \citet{kam2026symmetry} use its worst case as a heuristic loss proxy."
Also fix the mccracken2025 bib entry (wrong title and authors; see LIT_ml.md).

Searches: arXiv HTML of Morwani v2 (two targeted reads); Kam abs + HTML; McCracken HTML; WebSearch "modular addition margin upper bound Fourier amplitude 1-cos offset logit separation neural network lemma".
