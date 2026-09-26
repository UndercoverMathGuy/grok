# VERIFY (citation lens) — U2-grokking-testbed (task 90)

Verdict: UPHELD. CITATION_FOUND stands. All four recommended citations exist, the bib fields are right, and each supports the sentence it is attached to.

## Local check
- refs.bib contains power2022grokking and gromov2023grokking. `grep` over sections/*.tex finds no citation of either, and the word "grokking" appears in no section. Intro l.4-6 cites only morwani2024feature, nanda2023progress, chughtai2023toy and mccracken2025.
- liu2022towards and zhong2023clock are not in refs.bib.

## Sources fetched this session
1. arXiv 2201.02177 (abs + PDF text). Power, Burda, Edwards, Babuschkin, Misra. The title matches. There is no journal-ref (arXiv only). The PDF says "We train on the binary operation of division mod 97" and "binary operation tables of the form a o b = c". App. A.1.1 lists "the binary operations that we have tried (for a prime number p = 97)". It also reports that weight decay is "particularly effective". The bib entry is correct.
2. arXiv 2301.02679 (abs + PDF text + S2 API). Gromov, sole author. arXiv only (S2 venue arXiv.org, DBLP corr). Sec. 3.1 "Modular addition", Claim I: for network form (4) (quadratic activation, f = (1/DN) W2 (W1 x)^2), the weights are given by Eqs. (6)-(7), cos(2pi k n/p + phi). The phases are random and uniform and satisfy constraint (12). Width adds redundancy: "each frequency appears several times with different random phases". Caveats: the loss is MSE (CE optional), the parametrization is mean-field, and the result is a constructed solution, not a max-margin one. Citing it "next to Morwani for the analytic Fourier construction in quadratic nets" is accurate. The bib entry is correct.
3. arXiv 2205.10343 (abs + S2 API). Liu, Kitouni, Nolte, Michaud, Tegmark, Williams. Journal-ref "Accepted by NeurIPS 2022". S2 lists venue NeurIPS 2022 with DBLP key conf/nips/LiuKNMTW22. The phase diagram (comprehension / grokking / memorization / confusion) and the arithmetic toy tasks are confirmed. The recommended bibtex is correct.
4. arXiv 2306.17844 (abs). Zhong, Liu, Tegmark, Andreas. Journal-ref and comments both say "Accepted by NeurIPS 2023". The abstract uses modular addition and says "small changes to model hyperparameters and initializations can induce the discovery of qualitatively different algorithms" (Clock vs Pizza). The recommended bibtex is correct.

## Could not fetch
- Proceedings pages: dblp returned Anubis "Access Denied", OpenReview showed a bot-check page, and S2 returned 429 for 2306.17844 and 2201.02177. WebSearch is exhausted for this session (200/200).

## Notes
- The sentence "canonical testbed for grokking" leans on the grokking framing, which the current intro avoids entirely. The citation is correct either way; the framing is the authors' call.
- A minor imprecision in the checker's summary: in Gromov Eq. (6) the neuron index k enters as the frequency k/p, so no explicit range "k = 0..(p-1)/2" is stated at Claim I. This does not affect the recommended text.
