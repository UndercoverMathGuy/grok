# VERIFY novelty_ml: R29-kstar-table (task 31)

Verdict: the checker's NOVEL_AS_FAR_AS_FOUND stands. All 4 confirmed citations hold.

## Novelty search (ML lens)
- WebSearch was unavailable: the session's 200-search budget was used up, and the arXiv export API returned HTTP 429. I searched via the Hugging Face papers index instead (6 queries) and fetched PDFs directly.
- I found no paper that computes the optimal (max-margin or weight-decay-optimal) number of frequencies at a specific p, or that locates switch points in the regularisation exponent.
- Closest hits, none of which is prior art:
  - Tian, arXiv 2410.01779 (NeurIPS 2025): global optimisers for modular addition under MSE + weight decay. They use every frequency; "order-6" counts hidden nodes per frequency, not frequencies. The paper is dense in frequencies and gives no sparse count.
  - He et al. 2602.16849: diversification is dense, as already recorded.
  - Gromov 2301.02679: analytic solution over all frequencies.
  - Tian, Li2 2509.21519: dynamics and scaling. From its HF summary, it gives no count table.
- McCracken 2505.18266 Cor 4.8 is a sufficiency result: O(log n) frequencies give margin Omega(log n). Fig 3 is an empirical log fit. Neither gives the optimum count.

## Citation checks (PDFs downloaded and grepped)
- nanda2023progress (2301.05217): the header reads "Published as a conference paper at ICLR 2023". Line ~239 gives k in {14,35,41,42,52}. "Of the six non-zero frequencies, five key frequencies ...". OK.
- chughtai2023toy (PMLR 202:6243-6267): title, authors, volume and pages confirmed on the PMLR page.
  - Table 5 (MLP, C113) has 13/11/15/10 key reps.
  - Table 6 (Transformer, C113) has 3/4/4/3.
  - C118/D59/D61 MLP rows have 9-15. OK.
- zhong2023clock (2306.17844): authors are Zhong, Liu, Tegmark, Andreas. Model A at p=59 has circles k=17,3,44; keeping three circles gives 91.4% accuracy. OK. Model A is the pizza model.
- mccracken2025 (2505.18266): the real title is "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks". Authors: Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love. Sec 4.3 has "handfuls of frequencies (3-7)", and Fig 4 covers moduli 59-66. Cor 4.8 is confirmed.
  - The refs.bib entry is WRONG: it reads "Emergence of sparse and dense circuits...", with author "McCracken, Andrew". It must be fixed.

## Table notes
- The caption now states exhaustive K<=8,7,5 and heuristic above. The (113, nu=3)=7 caveat is only a LaTeX comment and should go into the caption.
- The equal-share argmax disagrees at (59,3)->6 and (113,3)->5 (appendix l.260).
- The p=13 switch at 1.285 matches the dense-up-to value at p=13 (cyclic.tex l.135), so the paper is internally consistent.
- None of the empirical counts (transformer/ReLU MLP) is in the L_{2,b} class. The comparison is only qualitative, as the checker said.
