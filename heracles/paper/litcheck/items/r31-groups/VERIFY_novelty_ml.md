# VERIFY (novelty_ml) - R31-groups, task 33

Claim (groups.tex, skeleton): the irrep ceiling is priced by the cost of writing the matrix-multiplication tensor as a sum of products (schoolbook at even degree, d^nu neurons). Within the interference-free class the ceiling gives one irrep on S5 and the Galois pair on A5. Structured interference beats the ceiling on D3, S4 and Z12.

Checker verdict: PARTIALLY_KNOWN. **Upheld.** The ingredients are known. The count prediction under L_{2,b}, b<nu, is not found in the ML literature. Three citation details need fixing (listed below).

## Search limits in this session
- WebSearch hit the session cap (200/200) on the first call. The arXiv export API returned HTTP 429 (rate-limited by the parallel verifiers). SSRN returned 403.
- The adversarial search therefore went through the OpenAlex API (10 queries, 2023-06 onward) plus direct arXiv abs/html fetches.

## Adversarial ML search: no scoop found
- **He, Wang, Zhang, Chen, Yang, arXiv 2606.02993** (fetched abs + html v2).
  - Thm 4.3(ii): each neuron locks to a single irrep with rank-one Fourier coefficients ("low-rank compression").
  - It makes no count prediction for non-abelian groups. Thm 5.1 (abelian) gives uniform diversification.
  - No regularisation analysis, and no matmul-tensor, Strassen or nuclear-norm pricing.
  - This is the closest recent work. It is not a scoop, and it is not in refs.bib.
- **Tian, CoGS, NeurIPS 2025, arXiv 2410.01779** (fetched abs). Scope is abelian only. It says weight decay favours simpler, low-order solutions. Nothing on irreps.
- **Kam et al., arXiv 2607.13749** (fetched abs). Abelian, holomorphic monomials, representability. Not relevant.
- **Ladewig et al., arXiv 2608.06597** (fetched abs). Deep linear nets. No groups.
- **Unverified lead:** Chen & Luo, "Scaffold, Not Seed - Except One: Positional Encoding and the Emergent Group Representations of Grokking Transformers" (SSRN 10.2139/ssrn.7331940, 2026). OpenAlex has no abstract and SSRN returned 403. The title concerns grokking transformers and positional encoding, so a norm-pricing scoop is unlikely. Worth one manual look.
- The OpenAlex queries (weight-decay irreps, S5 one irrep, A5 Galois, Schatten weight decay group, max-margin group sparse) surfaced nothing else relevant.

## Checker citations re-verified (all fetched this session)
- **Derksen, FoCM 16:779-811** (Crossref confirms; online 2015, bib year 2016 is fine). Checked in the PDF text:
  - Thm 1.9: the schoolbook is a diagonal SVD, with singular values 1 (x pqr).
  - Cor 1.10: ||M_{p,q,r}||_* = pqr.
  - Remark: Strassen's decomposition has length 2sqrt2+12 > 8, so "minimization of the rank, and minimization of the nuclear norm do not always coincide".
  - Thm 1.11: gives the singular values of T_G.
  - **FIX:** the norm ||T_G||_* = sqrt(n) sum d_i^{5/2} is **Corollary 1.12**, not Thm 1.11. Cite Cor 1.12 for the d^{2.5} link to Morwani Thm 9.
- **Christandl & Zuiddam, Comput. Complexity 28:27-56** (Crossref: online 2018, vol 28 = 2019 issue). The PDF text contains "For bipartite graphs, each inequality in (8) is an equality." Confirmed.
- **Fan, Ding, Yang, Zhang, Udell, TMLR 01/2023.** The PDF text of Thm 1 (Symmetric Regularizer 1) reads ||X||_{S_{q/d}}^{q/d} = inf_{X=CP} (1/d) sum_i sum_j ||x_i^(j)||^q. Confirmed.
- **Cheng et al., arXiv 2506.22134.** Title and authors confirmed (v2 07 Oct 2025, cs.CV, arXiv only). Thm 1 was not re-read; this is an optional cite.
- **Shutman, Louidor, Tessler, arXiv 2509.06931.**
  - Prop 5.2 confirmed verbatim: rank <= m_d / 3m_{d/2} / 8m_{d/2} by type I/II/III.
  - The Strassen sentence is confirmed.
  - Setting: MSE and mostly pure GD, although some experiments use AdamW.
  - It has no mechanism for choosing irreps.
- **Marchetti, Kunin, Myers, Acosta, Miolane, ICML 2026, arXiv 2602.03655** (v2 29 May 2026). **FIX:**
  - What I could verify is **Thm 4.3 (Cost Minimization)**, not Lemma B.8. The theorem gives a lower bound on the restricted loss, L >= 1/2(||x||^2 - C/|G| ||x_hat[rho*]||^2), which is "achievable for N >= (k+1) 2^k n_rho^{k+1}".
  - The (k+1) n_rho^{k+1} count is for sigma-pi-sigma neurons. For plain power neurons a Waring factor 2^k is added.
  - The html was truncated, so I could not see the text of Lemma B.8 itself.
  - Cite "Thm 4.3 and App. B.3". Say their price is width under MSE, not a norm. Do not equate their count with our d^nu without the Waring factor.
- **Morwani et al., ICLR 2024 spotlight, arXiv 2311.07568.** Confirmed. The group result says margin maximisation selects irrep features. I did not re-derive Thm 9's constant here.
- **Chughtai, Chan, Nanda, PMLR 202:6243-6267** (PMLR page confirmed). **FIX: the checker's quote is wrong.**
  - The actual text is: "Representations with more degrees of freedom may also offer better performance for fixed total weight norm, so which the model may prefer, and thus which is least complex, is unclear."
  - It does not say "standard sign offers better performance".
  - S5 over 50 seeds: "the 1d sign and 4d standard representations are most commonly learned, standard_sign (4d), 5d_a and 5d_b are learned approximately equally and less often, and 6d is never learned."
  - Networks there typically use SEVERAL reps. The "as observed" in groups.tex must point to the paper's own runs, and the Chughtai multi-irrep tension must be addressed.
- **Stander, Yu, Fan, Biderman, PMLR 235:46441-46467** (PMLR page confirmed). S5/S6 subgroup/coset circuits. Confirmed.
- **Wu, Jaburi, Drori, Gross, ICLR 2025** (arXiv comments say "ICLR 2025 camera ready"). Confirmed.

## Net
- The verdict stays PARTIALLY_KNOWN.
- Known parts:
  - the r=1 schoolbook optimality (Derksen);
  - the rank/width version of the price (Shutman; Marchetti);
  - the Schatten quasi-norm identity (Fan);
  - the even-cycle rank (Christandl-Zuiddam).
- Novel as far as found: the norm-priced count (one irrep on S5, Galois pair on A5) and the interference-beats-ceiling finding.
- Suggested addition: He et al. 2606.02993, cited as the dynamics-side single-irrep, rank-one result with no count prediction.
