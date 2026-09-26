# R9 interference-free definition (cyclic.tex def:if)

Verdict: PARTIALLY_KNOWN. The cancellation mechanism is folklore. Naming it as a class, and pricing it, is new as far as we found.

## What we define
beta_{j,T}=0 for every j and every non-empty T subset [n]. The sign-flipped same-frequency modes must cancel in the network's logit table, summed across neurons.

## Prior art (texts fetched and read this session)
1. Morwani et al., ICLR 2024 (arXiv 2311.07568)
   - App. F.2 builds 8 neurons per frequency: two neurons per trig monomial of cos(a+b-c), with phases (0,0,0),(0,pi,pi) etc. Their sum is exactly sum_zeta cos_zeta(a+b-c). "for each input (a,b), f takes the same value on all incorrect labels c', satisfying C.1."
   - So their max-margin witness is interference-free, but they never name or isolate that property. Condition C.1 plus Eq. 1 is stronger: the whole logit is constant on the wrong labels.
   - Their Thm 8 (sparse parity), App. G: 2^{k-1} sign-pattern neurons, "every term other than Pi x_{S_i} cancels out in the sum". This is the Boolean analogue of cancelling sign-flipped copies.
2. Li, Liang, Shi, Song, Zhou, AISTATS 2025 (arXiv 2402.09469)
   - App. E, general k: 2^k * 2^{k-1} neurons per frequency, built by sign-pattern polarisation. The network computes exactly sum cos(sum a_i - c) and satisfies "Condition 3.8".
   - This is an explicit interference-free construction for every k. It spends 2^{2k-1} neurons per frequency, compared with our nu = k+1. It is not a class, and it has no cost lemma.
3. Gromov, arXiv 2301.02679 (2023)
   - Eq. 11 expands the product into four sign-flipped waves: cos(n+m+q), cos(n-m-q), cos(n-m+q), ...
   - Eq. 12 is the phase constraint phi1+phi2=phi3, which gives "constructive interference". Every other term must "interfere destructively". Eq. 15 is the "destructive interference condition", met only approximately via random phases, with error O(sqrt N).
   - This is the closest conceptual precedent and even uses the word "interference". Here the cancellation is approximate and statistical, not exact.
4. Doshi, He, Das, Gromov, "Grokking Modular Polynomials" (ICLR 2024 BGPT workshop, arXiv 2406.03495)
   - Multi-operand generalisation of Gromov: "all other terms will vanish due to random phase approximation" (Eq. 15 area).
5. He, Wang, Chen, Yang, arXiv 2602.16849 (2026)
   - Def. 4.1(iii): sum_{m in N_k} exp(i*iota*phi_m)=0 for iota in {2,4}. This is an exact phase-moment cancellation across the neurons of one frequency group, and it kills the "residual noise" terms in Eq. 4.1.
   - Prop. 4.2: with it, the logits are -1 + (p/2)1(x+y=j) + (p/4)sum 1(2z=j). Some noise survives (spurious peaks at 2x, 2y), so this is not interference-free in our sense.
   - This is the closest exact-cancellation condition. It is formally a cousin of our Eq. (JF): sum_l beta_l prod xi = 0.
6. Zhong, Liu, Tegmark, Andreas, NeurIPS 2023 (arXiv 2306.17844)
   - Clock logits are cos(w(a+b-c)). Pizza logits are |cos(w(a-b)/2)| cos(w(a+b-c)).
   - The extra a-b factor in Pizza is exactly label-independent same-frequency interference. Clock vs Pizza therefore lines up informally with interference-free vs interfering.

## Novelty assessment
- Known: exact cancellation of sign-flipped / a-b copies across several neurons, used as a construction device (Morwani F.2, Li App. E) and as an approximate random-phase mechanism (Gromov, Doshi, He).
- Not found: the property as a *class* over which the margin is maximised; the sign rule (Lemma sign); the finite system (JF); and the lemma that a clean feature costs exactly nu neurons with the factor nu^{1-r}. The last was checked separately.
- Searches run: arXiv full texts of the six papers above (grep for cancel/interference/phase/a-b).

## Recommended change
After def:if, add one sentence crediting the device. Suggested text: "Exact cancellation of these copies across neurons is how the max-margin constructions of \citet{morwani2024feature} (App. F.2) and \citet{li2025nsum} (App. E) reach logits depending only on a_1+...+a_n-c, and is the exact form of the 'destructive interference' that \citet{gromov2023grokking} obtains approximately from random phases (see also \citealp{he2026mechanism}, Def. 4.1; the Clock/Pizza contrast of \citealp{zhong2023clock}); we isolate it as a class and price it (Lemma~\ref{lem:cost})." Also note that Li's construction uses 2^{2k-1} neurons per frequency against the nu-block's nu.
