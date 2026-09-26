# VERIFY (novelty_ml): R20 cyclic theorem (thm:cyclic)

Verifier task 15, run on 2026-09-25.

## Verdict
The checker's verdict, PARTIALLY_KNOWN, stands. Nobody has published the value nu^{-(nu-b)/b} B_p Gamma_{nu/b}(p). Nobody reduces the margin to a budget programme or prices sparsity by nu-b. Nobody shows that exactly nu equal neurons per frequency is necessary at a norm optimum.

The attaining construction (the nu-block) is known, from Marchetti et al., Lemma B.8 Case 1.

One recommended change is wrong and must be fixed; see the next section.

## The error in the recommended change
The checker wants the paper to say that thm:cyclic at b=nu "recovers Morwani Thm 7 and Li Thm 4.1". This is false.
- thm:cyclic is about product networks: u_1(a_1)...u_n(a_n) w(c).
- Morwani uses the power network (u(a)+v(b))^2 w, and Li uses (sum_t u_t(a_t))^k w.
- The power single-neuron budget carries an extra factor n!. At b=nu the product-class value B_p p/(p-1) is n! below the Morwani and Li values.
- The paper recovers these values in thm:universal / cor:dense, via C_b^{nu/b} = n! B_p. For Morwani this is already done at approach.tex:70 and appendix.tex:32.
- Check for Li: n! B_p p/(p-1) = 2 k!/((2k+2)^{(k+1)/2} p^{(k-1)/2} (p-1)), which is exactly Li Thm 4.1. Numerically the ratio is 1.000 for p in {7, 11, 113} and k in {2, 3, 4}.
- The appendix itself says (appendix.tex:266, 279) that the nu-block does NOT attain the power-network bound.
- Fix: attribute the b=nu recovery of Morwani Thm 7 and Li Thm 4.1 to thm:universal / cor:dense. Optionally add the Li match there.

## Citations checked (all fetched this session)
- **Marchetti, Kunin, Myers, Acosta, Miolane, arXiv 2602.03655** (v2 29 May 2026; comment "Accepted at ICML 2026"). Read from the HTML.
  - Lemma B.8 Case 1: "replicate the previous construction k+1 times ... multiply S_h^i by e^{pi i j/(k+1)} ... s_w^i by e^{-pi i jk/(k+1)}/(k+1)". For real rho_* they take real parts.
  - Eq (71): (1/(k+1)) sum_j e^{-2 pi i j (k-|A|)/(k+1)} vanishes because A is a proper subset.
  - Lemma B.6 / Remark B.7: the Waring decomposition gives 2^{k-1} powers, which is optimal (Carlini et al.).
  - Main text: "the exponential factor is optimal in general".
  - The construction is sufficient for AGF cost (square-loss) minimisation. It has no margin and no norm-optimality or necessity claim. CONFIRMED.
- **Tian, arXiv 2410.01779 v4** (NeurIPS 2025 footer). Read from the PDF.
  - Cor 2: order-6, "each frequency are affiliated with 6 hidden nodes", summed over all k=1..(d-1)/2.
  - Cor 4: order-4 single frequency.
  - Cor 5: mixed order-4/6.
  - Thm 5, "Occam's Razer": low-order solutions are preferred under weight decay. Setting is L2 loss on a quadratic net. CONFIRMED.
- **Gromov, arXiv 2301.02679.** Read from the PDF.
  - Eq (12): phi1 + phi2 = phi3.
  - Eq (15): with random phases the sum is << N. The cancellation is approximate and uses all frequencies. CONFIRMED.
  - Note: gromov2023grokking is in refs.bib but no section cites it.
- **Morwani et al., arXiv 2311.07568.** Thm 7: L_{2,3}, m >= 4(p-1), gamma* = sqrt(2/27)/(p^{1/2}(p-1)), and every frequency is used. CONFIRMED, but it is the endpoint of thm:universal, not thm:cyclic (see above).
- **Li, Liang, Shi, Song, Zhou, arXiv 2402.09469 v4.** Thm 4.1: m >= 2^{2k-1}(p-1)/2 = 2^{2k-2}(p-1), L_{2,k+1}, power neurons (Def D.5), all frequencies. CONFIRMED, with the same attribution caveat.

## Adversarial search (ML lens)
The session's WebSearch budget was exhausted (200/200), and the arXiv export API returned HTTP 429 because parallel agents share one IP. I searched through arXiv listing pages via WebFetch instead:
- abs "modular addition margin"
- "grokking margin"
- abs "grokking frequencies weight decay"
- abs "modular addition" "weight decay"
- "modular addition Fourier features theory"
- two further queries on max-margin sparse polynomial networks and roots-of-unity cancellation, which returned 0 hits

Candidates opened:
- **2609.26679**, Pracher et al., Sep 22 2026, "Spectral theory of grokking: weight decay induces feature learning". NTK and residual dynamics. No frequency count and no margin. Not relevant.
- **2606.08985**, Tan, Gai and Zhang, "Beyond Neural Collapse". Weight-decay surrogates favour a rank-2, single-frequency cyclic geometry, and the paper frames grokking as a "trade-off between separation, symmetry, and complexity". It gives no margin value, no count law in p, and no nu-block. Relevant only to the paper's trade-off framing.
- **2509.21519**, Tian, Li_2 scaling laws. Feature-emergence dynamics. No margin optimum.
- **2506.06489**, Kunin et al., AGF, NeurIPS 2025. Quadratic modular addition; cancellation is by neuron pairs via the polarization identity, not by nu roots of unity. It is not an earlier source for the block.
- **2602.16849** (He et al.) was already audited in a previous session: dense, no arithmetic constraint.
- Also seen and not relevant: 2606.05863, 2608.07436, 2511.23443, 2603.13331, 2607.23967.

Conclusion: no ML-literature source for the theorem's value, the programme reduction, the count, or the necessity. The construction's prior art is Marchetti et al. (for product/sigma-pi-sigma neurons). Gromov is approximate. Tian uses 6 or 4 nodes under L2 loss.
