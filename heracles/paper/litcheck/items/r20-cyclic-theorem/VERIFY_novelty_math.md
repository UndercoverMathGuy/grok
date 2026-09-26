# VERIFY (novelty_math) -- R20-cyclic-theorem (thm:cyclic, app:thm)

Verdict: checker's PARTIALLY_KNOWN upheld. All five confirmed citations verified by direct PDF read.
Math-lens additions: the rank shadow of Lemma cost has a known symmetric analogue (real Waring rank of
binary forms with all real roots), and the equality case is a complex Hadamard matrix (classical object).
The value nu^{-(nu-b)/b} B_p Gamma_{nu/b}(p), the l_r (r<1) cost lemma, and the structure/count statements
were not found in the math literature.

## Confirmed citations (all re-read this session, PDFs via arxiv.org/pdf + pdftotext)
- marchetti2026sequential (arXiv 2602.03655v2; abs page comment "Accepted at ICML 2026"). Lemma B.8 Case 1,
  verbatim: replicate k+1 times; "multiply S_h^i by the unitary scalar e^{pi i j/(k+1)}, and similarly multiply
  s_w^i by e^{-pi i j k/(k+1)}/(k+1). (When rho* is real, we multiply by the real parts...)"; eq (71)
  (1/(k+1)) sum_j e^{-2 pi i j (k-|A|)/(k+1)}; "Since A is a proper subset ... k-|A| != 0 (mod k+1). This
  implies that (71) vanishes". Sufficient construction only (N >= (k+1) n_rho^{k+1}); no norm-optimality,
  minimality or necessity claim. Lemma B.6 / Remark B.7: Waring decomposition of the square-free monomial,
  2^{k-1} terms necessary (citing Carlini et al. 2011). CONFIRMED.
- tian2025composing (arXiv 2410.01779v4, "NeurIPS 2025 camera ready"). Corollary 2 "Order-6 global solutions
  ... each frequency are affiliated with 6 hidden nodes (order-6)"; Theorem 5 "The Occam's Razer: Preference
  of low-order solutions". CONFIRMED.
- gromov2023grokking (2301.02679): eq (12) phi1+phi2=phi3, random uniform phases, eq (15) sum ~ sqrt(N) << N:
  approximate cancellation only. CONFIRMED.
- morwani2024feature (2311.07568): Thm 7, m >= 4(p-1), gamma* = sqrt(2/27) / (p^{1/2}(p-1)), every neuron one
  frequency. CONFIRMED (b = nu = 3 endpoint).
- li2025nsum (2402.09469): Thm 4.1, m >= 2^{2k-1}(p-1)/2, max L_{2,k+1} margin 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}).
  CONFIRMED (b = nu endpoint, power neurons).

## Math-lens prior art the checker missed
1. Rank shadow of Lemma cost (r -> 0). Prop esym gives "at least nu non-zero beta_l", i.e. the real tensor rank of
   Re(z_1...z_nu) (as a real 2x...x2 tensor) is >= nu, attained by the nu-block. The SYMMETRIC analogue is a known
   theorem: Comon & Ottaviani, arXiv 0909.4865, Prop 3.1 "Let f be a real binary form of degree d with d real
   distinct roots. Then the (real) rank of f is equal to d"; Causa & Re, arXiv 1006.5127 (Annali di Matematica,
   per arXiv comment): rank d iff d distinct real roots. Re((x+iy)^nu) has nu distinct real roots, so its real
   Waring rank is nu, with the classical equally-spaced (rotated roots of unity) decomposition. For nu=3 the
   multilinear case is the classical 2x2x2 real-rank-3 / complex-rank-2 example. I did not locate the general
   multilinear statement for nu >= 4 (search budget exhausted, see below). Suggest one sentence: "As r -> 0 the
   lemma says a clean feature needs nu product neurons, the multilinear counterpart of the fact that
   Re((x+iy)^nu) has real Waring rank nu \citep{comon2012typical,causa2011maximum}."
2. Equality case = complex Hadamard matrix. nu unimodular, pairwise orthogonal chain vectors with first entry 1 are
   exactly the rows of a dephased complex Hadamard matrix (Tadej & Zyczkowski, quant-ph/0512154, OSID 13 (2006)).
   Also implied (not stated in the paper): beta_l = beta/nu with EQUAL phases, since H = U D U^* is normal and upper
   triangular, hence diagonal, hence H = beta I. Orthogonality alone is NOT sufficient for interference-freeness:
   at nu=4 the one-parameter family F4^(1)(a) (T&Z eq. 65) is orthogonal for all a, but I computed (numpy) the
   non-interval pattern T={1,3}: sum_l xi_l1 xi_l3 /4 = 0.585+0.493i at a=0.7 (vanishes only at the Fourier point).
   So the lemma is correct as stated (JF is a hypothesis), but the "Weaker cancellation suffices" remark (interval
   patterns only) admits a strictly larger optimum set that is not interference-free. Worth one sentence; also
   Haagerup's uniqueness at order 5 (T&Z line citing [21] Th 2.2) means nu in {2,3,5} optima are Fourier-type up to
   equivalence even before the non-interval constraints.
3. Tools used in the proof are standard and proved inline (Sylvester determinant identity, Hadamard's inequality,
   Bernstein/Stieltjes integral x^r = (r sin(pi r)/pi) int log(1+xt) t^{-1-r} dt). No issue; optionally cite
   Schilling-Song-Vondracek "Bernstein functions" for the representation (not fetched this session, so not listed).
4. Not found anywhere: the margin value with the nu^{-(nu-b)/b} splitting price, the l_r (0<r<1) lower bound
   sum|beta_l|^r >= nu^{1-r}|beta|^r with its equality case, and the frequency-count/structure statements.

## Operational note (for the workflow owner)
WebSearch refused every call: "this session has used its web search budget (200 of 200 WebSearch calls)". The
budget is shared across all verify subagents in this session, so later agents get zero searches. The arXiv API
(export.arxiv.org) then returned HTTP 429 Too Many Requests after two queries. The shared scratchpad holds ~90
PDFs from other agents. These are likely causes of the slow verify stage; worked around via curl + pdftotext.
