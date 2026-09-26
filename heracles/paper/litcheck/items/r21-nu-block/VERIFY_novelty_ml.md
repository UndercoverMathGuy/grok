# VERIFY novelty_ml: R21-nu-block (task 17)

## Verdict
PARTIALLY_KNOWN, and more known than the checker said. The checker wrote "None of the ML constructions uses a nu-neuron roots-of-unity block". That is wrong for nu = 3. Tian's CoGS paper has the nu = 3 block, the fact that 3 neurons are needed, and the minimal-norm choice of phases.

## Missed ML prior art: Tian, arXiv 2410.01779 (v4, 30 Sep 2025; NeurIPS 2025)
Title: "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets" (v1 was titled "Composing Global Optimizers..."). I read the full PDF this session.
- Setting: a 2-layer network with quadratic activation and L2 (MSE) loss on Abelian group multiplication, including modular addition. The cross term 2(w_a.e_a)(w_b.e_b)w_c is exactly a degree-3 product neuron. Its "sum potentials" r_{k1 k2 k} = sum_j z_akj z_bkj z_ckj are the table coefficients of a product network.
- Table 1: the generator u_syn = [w3, w3, w3] gives the order-3 block rho(u_syn) = u^2+u+1. That is 3 hidden nodes per frequency with cube-root-of-unity phases, and it cancels every sign-flipped term (abar b c, a bbar c, a b cbar).
- Fig. 2 (bottom): with the R* term dropped from the loss, gradient descent finds exactly 3 hidden nodes per frequency, the z_syn block.
- App., Lemma 5 ("Necessary condition for R_c"): every order-1 and order-2 solution that kills all cross terms has r_kkk = 0. So at least 3 neurons per frequency are needed. This is the nu = 3 case of the "nu neurons are necessary" count.
- App., Lemma 6 and "Minimal Norm solutions" (Eqs. 91-96): the order-3 solutions satisfying R_c have the form [1, +-w3, +-w3bar]. The minimum-L2-norm one has equal magnitudes m_j and phases 2pi/3 and 4pi/3. This is the nu = 3 case of prop:block's equal-amplitude, equally spaced phases, including its norm optimality at a fixed neuron count.
- Differences from our result: n = 2 only (no general nu). MSE with a small L2 penalty, not max L_{2,b} margin. Equal amplitudes across frequencies (no lambda_j^nu proportional to y*_j). No link to Gamma_s. The global solutions in the sum-activation setting are order 6 or order 4/6.
- The r12-cost-of-feature verifier wrote that Tian "proves no lower bound and no norm cost". That is inaccurate. Lemma 5 is a count lower bound and Eqs. 91-95 are a min-norm result, both at nu = 3.

## Other searches
- Huang and Li, arXiv 2511.23443 ("Provable Benefits of Sinusoidal Activation for Modular Addition"). Its width-2 sine networks are a different mechanism (the activation is itself periodic). No product block. Not prior art.
- Tian, arXiv 2509.21519 (Li^2 scaling laws): grepped, no roots-of-unity block.
- arXiv API queries: ("roots of unity" AND "modular addition") returned nothing; ("modular addition" AND margin) returned Morwani, Li et al. and 2511.23443 only.

## Confirmed citations, re-checked by me
- Mohlenkamp and Monzon 2005: Crossref gives Math. Intelligencer 27(2):65-69, March 2005, correct. Thm 2 (sin of a sum, n separable terms with free alpha_j and shifts a_j), the "separation rank n" sentence and Fig. 1 (alpha = 0, pi/3, 2pi/3) are confirmed in the preprint.
- Morwani 2311.07568 App. F.2: "8 neurons for each of the frequencies" is confirmed (line "Our theta* will consist of 4(p-1) neurons: 8 neurons for each...").
- Li et al. 2402.09469v4, Lemma E.3 / Lemma 4.3: "2^{2k-1} * (p-1)/2 neurons", i.e. 2^{2k-1} per frequency, confirmed. Lemma E.1 is the sum-to-product identity. Note the abstract and Thm 4.1 widths use 2^{2k-2}(p-1) for m.
- Gromov 2301.02679: phases are "random, sampled from a uniform distribution", with approximate cancellation for large N. Confirmed.
- He et al. 2602.16849, Def 4.1 (Full Diversification): the condition sum_{m in N_k} exp(i iota phi_m) = 0 for iota in {2,4} is confirmed. Prop 4.2 is confirmed.

## Numerical check
The identity sum_{l=0}^{n} prod_k cos(x_k + pi l/nu) * cos(c + n pi l/nu) = nu 2^{-n} cos(sum x - c) holds to 1e-15 for n = 1..6. It holds exactly as written in prop:block, with no (-1)^l needed.

## Recommended text change (adds to the checker's)
After the sentence "The phases are a rotated copy of the nu-th roots of unity", add: "For nu=3 this block, the fact that fewer than three neurons cannot cancel the sign-flipped terms, and its minimal-norm phases already appear in Tian's analysis of quadratic networks under MSE loss \citep[Table 1, Lemmas 5-6]{tian2025cogs}. The general-nu identity is the equally spaced case of \citet[Thm 2]{mohlenkamp2005trig}." Keep the contrasts with Morwani (8 neurons) and Li et al. (2^{2k-1}). What is claimed as new should be: general nu, the L_{2,b} norm accounting, the lambda_j^nu proportional to y*_j weighting, and attaining Gamma_s. lem:cost should cite Tian Lemma 5 as its nu = 3 count case (see r12).

## Tooling note (why the workflow is slow)
WebSearch is refused with "200 of 200" because the session-wide budget is shared by all workflow agents. So every verifier now has to fall back to WebFetch and curl on the arXiv PDF and API, Crossref and OpenAlex. This item was done with curl and pdftotext.
