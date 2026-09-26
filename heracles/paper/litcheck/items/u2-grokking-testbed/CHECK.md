# U2-grokking-testbed — CITATION_FOUND

Status in paper: refs.bib has power2022grokking and gromov2023grokking, but no section cites them. The only grokking/mod-add
background citations are nanda2023progress, chughtai2023toy, morwani2024feature, and mccracken2025 (that entry is itself wrong; see LIT_ml.md).
Liu 2022 and Zhong 2023 are not in refs.bib at all.

## Verified (fetched this session)
- Power, Burda, Edwards, Babuschkin, Misra. "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets".
  arXiv:2201.02177 (6 Jan 2022). From the PDF: "We train on the binary operation of division mod 97"; the appendix says
  "binary operations that we have tried (for a prime number p = 97)". This is the origin of the grokking-on-modular-arithmetic testbed.
- Gromov. "Grokking modular arithmetic". arXiv:2301.02679 (2023). Sec. 3.1, Eqs. (6)-(7): an analytic solution for
  modular addition by a two-layer network with QUADRATIC activation. The weights are periodic in n1, n2, q, and the
  frequencies k range over 0..(p-1)/2. The phases are random and satisfy constraint (12). Each frequency appears several times
  with different random phases. This is the direct precursor of our constructed Fourier/product nets, and it is the
  quadratic-network setting that Morwani generalises. It should be cited in prelim/approach next to Morwani as well.
- Liu, Kitouni, Nolte, Michaud, Tegmark, Williams. "Towards Understanding Grokking: An Effective Theory of Representation Learning".
  arXiv:2205.10343; the arXiv page lists NeurIPS 2022. It covers phase diagrams (comprehension/grokking/memorization/confusion)
  and the role of weight decay. It is background only.
- Zhong, Liu, Tegmark, Andreas. "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks".
  arXiv:2306.17844; the arXiv page lists NeurIPS 2023. It uses modular addition, and different hyperparameters/inits give
  different algorithms (Clock vs Pizza). This is relevant to our framing that the solution depends on architecture/regularisation.

I could not check venue details on the proceedings pages: the search budget was exhausted and Semantic Scholar returned 429.
The venues above come from the arXiv comment fields.

## Suggested edit (intro para 2)
"Modular addition is the standard testbed for grokking \citep{power2022grokking,liu2022towards} and for mechanistic
interpretability: trained networks implement Fourier circuits \citep{nanda2023progress,gromov2023grokking,zhong2023clock} ..."
