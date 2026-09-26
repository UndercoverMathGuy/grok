# R29-kstar-table: literature check

Item: cyclic.tex tab:kstar. K*(nu/b,p) at b=2 is 5,4,3 at p=59 and p=71, and 7,4,4 at p=113. Shape switches at p=13 (s=1.285/1.407/1.678) and p=23 (8 switches, s=1.955 ratio-only).

## Verdict: NOVEL_AS_FAR_AS_FOUND (these exact numbers are our computation of our programme). There is related empirical context that reviewers will expect to see cited.

No paper computes the maximiser of the L_{1/s}-budget coverage max-min at specific p, and no paper reports shape switch points in s.
Morwani et al. (ICLR 2024) Thm 7 and Li et al. (AISTATS 2025) Thm 4.1 give dense optima (b = nu). McCracken et al. (2505.18266) Cor 4.8 is a sufficiency bound for random frequencies, with no count at the optimum.

## Empirical counts in the literature, all verified from the PDFs this session
| Source | Architecture | p | Count |
|---|---|---|---|
| Nanda et al. 2023, Sec 4.1 / Fig 3 | 1-layer transformer, AdamW wd=1 | 113 | "five key frequencies ... k in {14,35,41,42,52}". WE has 6 non-negligible frequencies. |
| Chughtai et al. 2023, App. Table 6 | 1-layer transformer | C113 | 3, 4, 4, 3 key frequencies (seeds 1-4) |
| Chughtai et al. 2023, App. Table 5 | 1-hidden-layer MLP with embeddings | C113 | 13, 11, 15, 10 key frequencies (seeds 1-4) |
| Chughtai et al. 2023, App. Table 5 | MLP | C118 / D59 / D61 | ~9-15 reps |
| Zhong et al. 2023, Sec 4 | transformers | 59 | Model A: the top 3 circles (k=17,3,44) give 91.4% accuracy; the full count is not tabulated |
| McCracken et al. 2025, Sec 4.3, Fig 3/Fig 4 | clocks/pizzas/MLPs with embeddings | many n | "handfuls of frequencies (3-7)"; log fit of count vs n. One-hot 1-layer MLPs use floor(n/2) |

Reading: our p=113 values (7/4/4) sit inside the empirical range (3 to 15, depending on architecture). But the empirical models are transformers or ReLU MLPs, not the product/power neurons with L_{2,b} that we study, so this is context and not a test. Chughtai's MLP counts (10-15) are above the nu=3 value of 7. If a reviewer maps "ReLU MLP" onto some nu, they could read this as tension. Do not claim that Nanda's 5 matches our table.

## Heuristic cells (flag)
1. The caption says enumeration is exhaustive only for K <= 8, 7, 5 at p = 59, 71, 113. Exhaustive-up-to-K certifies only the best support of size at most Kmax. It does not exclude a larger support beating it. Theorem thm:scaling's upper bound on K is 9*4^{1/(s-1)} ln p, which is about 587 at s=3/2 and p=59, so it does not close the gap. So *every* cell is certified only among supports of size at most Kmax. The cells with K* < Kmax still depend on the heuristic search above Kmax to rule out larger K. The caption wording is accurate, but the text ("Table gives the counts") reads as exact.
2. (113, nu=3) = 7 is above the exhaustive range (the source comment at line 104 says so). The equal-share argmax gives 5 there (appendix.tex line 260), and 6 at (59,3). Report 7 as a heuristic value or a band.
3. (113, nu=5) = 4 and (113, nu=4) = 4 are at or below Kmax=5, so they are exhaustive within K<=5 only.
4. The p=13 and p=23 switch points come from an exact computation at small p. No literature compares with them.

## Recommended citations
All are already in refs.bib, except that mccracken2025 has a wrong title/author (see sweep item 1).
- nanda2023progress (arXiv 2301.05217; ICLR 2023)
- chughtai2023toy (PMLR 202:6243-6267)
- zhong2023clock (arXiv 2306.17844, NeurIPS 2023)
- mccracken2025 -> fix to McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266
