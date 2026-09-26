# R9 interference-free definition: novelty_ml verification (task 7)

Verdict: the checker's PARTIALLY_KNOWN label stands, but its evidence does not hold up.
- It missed two NeurIPS 2025 papers that state the n=2 condition exactly.
- It described He et al. 2026 backwards.
- Its Clock/Pizza claim is false.
Upheld: false.

## What I fetched (arXiv PDFs pulled with curl and read via pdftotext, plus arXiv abs metadata)
2311.07568, 2402.09469, 2301.02679, 2602.16849, 2306.17844, 2406.03495, 2410.01779, 2506.06489, 2412.03773, 2509.21519.
The session WebSearch budget was already exhausted (200/200 calls) when this task started, so I checked candidates by pulling full texts directly.

## Confirmed citations
- **Morwani et al. ICLR 2024 (2311.07568), App. F.2: CONFIRMED.**
  - 8 neurons per frequency, with phase pairs (0,0,0)/(0,pi,pi) and so on.
  - The network computes sum_zeta cos_zeta(a+b-c).
  - The text says "f takes the same value on all incorrect labels c', satisfying C.1."
  - For sparse parity (App. G), it says "every term other than Pi x_Si cancels out in the sum."
- **Li et al. AISTATS 2025 (2402.09469), App. E.2: CONFIRMED.**
  - "2^{2k-1} neurons dedicated to each frequency" (2^k terms x 2^{k-1} neurons).
  - The network computes sum cos_zeta(sum a_i - c).
  - The text says "same outcome to all incorrect labels c', thereby fulfilling Condition 3.8."
- **Gromov 2023 (2301.02679), Eqs. 11-15: CONFIRMED.**
  - The sign-flipped waves appear in Eq. 11. Eq. 12 is the constraint phi1+phi2=phi3, giving "constructive interference".
  - The text says "all other waves ... interfere destructively" and calls Eq. 15 the "destructive interference condition". It holds only approximately, via random phases.
- **Doshi, He, Das, Gromov (2406.03495), ICLR 2024 BGPT workshop: CONFIRMED.**
  - The text says "all other terms will vanish due to random phase approximation."

## Incorrect characterisations in the check
1. **He, Wang, Chen, Yang (2602.16849), Def. 4.1(iii) and Prop. 4.2.** The checker says spurious 2x/2y peaks survive, "so their condition is not interference-free in our sense". That is wrong.
   - In the proof of Prop. 4.2 (App. B.1, Eq. B.1), condition (iii) sets sum_m[2cos(w j+2phi_m)cos^2(w(x-y)/2) + cos(w(x+y+j)+4phi_m)] to zero.
     - The iota=2 moment kills the cos(w(x-y-/+j)) terms, i.e. T={1},{2}.
     - The iota=4 moment kills cos(w(x+y+j)), i.e. T={1,2}.
   - What survives is aN/2 cos(w(x+y-j)) + aN/4[cos(w(2x-j)) + cos(w(2y-j))]. The 2x/2y terms sit at modes (2k,0;-k) and (0,2k;-k). These are interference at *other* modes, which def:if explicitly leaves unrestricted.
   - Numerical check (p=13, quadratic neurons, psi=2phi):
     - N=3 neurons with phases {0, pi/3, 2pi/3}: beta_T=0 for all non-empty T, beta_0=1.5.
     - N=1 neuron: |beta_T| = |beta_0| = 0.5, the sign rule.
   - Conclusion: He's "full diversification" is exactly def:if in the equal-magnitude quadratic parametrisation. Its minimal realisation, three phases 0, pi/3, 2pi/3, is the nu-block. It is not a mere cousin.
2. **Zhong et al. NeurIPS 2023 (2306.17844), Clock/Pizza.** The idealised Pizza logit is |cos(w(a-b)/2)| cos(w(a+b-c)). Its a-b modulation lives at modes (k+m, k-m; -k) with m != 0, which are other modes.
   - Numerical check (p=13): beta_{j,T}=0 for every non-empty T and every j. So the idealised Pizza is interference-free per def:if.
   - Citing it as "the kind of label-independent interference that def:if excludes" would be wrong. It fits instead as an example of the other-mode interference that def:if allows, next to "Interference at other modes is unrestricted".

## Missed prior art (both NeurIPS 2025; the venue is printed in each PDF header)
3. **Kunin, Marchetti, Chen, Karkada, Simon, DeWeese, Ganguli, Miolane. "Alternating Gradient Flows" (2506.06489), App. F.2.2, Thm F.7, Eq. (85).**
   - This is the MSE cost minimisation over N harmonically aligned quadratic neurons at one frequency.
   - Equality holds iff sum_i C_i e^{i alpha_i} = 0 for (C, alpha) = (A_w A_u A_v, s_w +/- (s_u - s_v)) and (A_w A_u A_v, s_w + s_u + s_v). Square-term (other-mode) conditions are also required.
   - The proof then says "the only non-vanishing term" is cos(s_x + 2pi xi(c-a-b)/p) (Eq. 92).
   - This is the finite system (JF) at n=2 with arbitrary amplitudes, derived as an *optimality* condition. The checker's statement that (JF) is "not found" is therefore false for n=2.
   - The paper also says, near Eq. 77: "neurons can arrange for square terms to cancel in aggregate ... Such cancellations correspond to directional adjustments that can occur rapidly."
4. **Tian. "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets" (2410.01779), NeurIPS 2025.**
   - Lemma 1 gives zero-loss sufficient conditions with 0-set R_c = {r_{k1k2k}: k1,k2,k not all equal}, where r_{k1k2k} = sum_j z_{ak1j} z_{bk2j} z_{ckj}.
     - Restricted to +/-k, R_c consists exactly of the sign-flipped copies (Table 1 columns "a-bar b c", "a b-bar c", "a b c-bar").
     - R_c also imposes cross-frequency conditions, so it is strictly stronger than def:if.
   - Table 1: u_syn = [w3,w3,w3] has sign-flipped SPs equal to w3, and rho(u_syn) = u^2+u+1 is an order-3 (3 hidden nodes) per-frequency solution that zeroes them all. Its chain vectors are (1, w^s, w^{2s}): three equal-weight neurons with DFT chain vectors, i.e. the n=2 nu-block.
   - Fig. 2 (bottom): when the R* term is dropped, "each frequency has exactly 3 hidden nodes, corresponding to the solution z_syn = rho(u_syn)".
   - Tian gives no lower bound and no margin; the loss is MSE.
   - The repo's sweep (LIT_cite.md) already lists Tian for the nu-block and the cost lemma, but the R9 check did not cite it at def:if.
5. **Checked, not relevant to def:if:**
   - Chan/Yip 2412.03773: ReLU Pizza with an |cos(k(a-b)/2)| factor. This is other-mode and approximate.
   - Tian 2509.21519 (Li_2): nothing on sign-flipped cancellation.

## Novelty assessment
- **Known.** For n=2 (product/quadratic neurons), exact cancellation of the same-frequency sign-flipped copies across the neurons of one frequency appears in three places:
  - as an equation system (AGF Eq. 85, general amplitudes);
  - as part of zero-loss conditions (Tian R_c);
  - as a phase-moment condition for equal-magnitude neurons (He Def. 4.1(iii)).
  - A 3-neuron realisation exists (Tian rho(u_syn); He with three phases).
  - Exact constructions exist for all n (Morwani F.2; Li App. E, with 2^{2n-1} neurons per frequency). Gromov and Doshi give the approximate random-phase version.
- **Apparently new.**
  - Stating the condition for general n on the logit table, rather than per neuron.
  - Separating it from other-mode interference.
  - Using it as the constraint class for L_{2,b} margin maximisation.
  - The general-n sign rule.
  - The nu^{1-r} cost lemma with its equality case, a lower bound that none of the above gives.
- **Corrected verdict:** PARTIALLY_KNOWN. The definition itself (n=2) is known. The general-n class, its use as the margin class, and its pricing are not.

## Recommended text (replaces the checker's)
After def:if: "For two operands this condition is known in other guises:
- it is the equality case of the MSE bound over aligned quadratic neurons in \citet{kunin2025agf} (Thm. F.7);
- it is part of the zero-loss conditions $R_c$ of \citet{tian2025cogs}, whose three-node solution $\rho(u_{\rm syn})$ is a $\nu$-block;
- for equal-magnitude neurons, it is the phase condition of \citet{he2026mechanism} (Def. 4.1(iii)).

Exact constructions for any $n$ appear in \citet{morwani2024feature} (App. F.2) and \citet{li2025nsum} (App. E, $2^{2n-1}$ neurons per frequency), and the approximate random-phase version in \citet{gromov2023grokking}. We state it for general $n$, maximise the margin over it, and price it (Lemma~\ref{lem:cost})."

In the "Interference at other modes is unrestricted" sentence, optionally add: "(e.g. the $a-b$ modulation of the Pizza circuit, \citealp{zhong2023clock}, and the $2x,2y$ peaks of \citealp{he2026mechanism}, Prop. 4.2)."
Do NOT cite Zhong as an example of what def:if excludes.
