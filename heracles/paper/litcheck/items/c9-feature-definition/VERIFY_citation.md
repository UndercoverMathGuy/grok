# VERIFY (citation lens): C9-feature-definition, task 83

Claim (approach.tex:27): "a frequency with $A_j>0$ is a \emph{feature}, as in \citet{nanda2023progress,morwani2024feature}".

## Fetched this session
- arXiv 2301.05217 (abs + PDF v3). The PDF header reads "Published as a conference paper at ICLR 2023". Title and authors (Nanda, Chan, Lieberum, Smith, Steinhardt) match the bib.
  - Sec. 4.1, verbatim: "The embedding matrix WE is sparse in the Fourier basis–it only has significant nonnegligible norm at 6 frequencies. Of these frequencies, only 5 appear to be used significantly in later parts of the model (corresponding to k ∈ {14, 35, 41, 42, 52}). We dub these the key frequencies of the model."
  - The words "feature"/"features" appear once in the extracted text: "some feature of the loss landscape". The paper never calls a frequency a feature.
  - Its criterion is thresholded and based on use: it excludes a 6th frequency that has non-negligible embedding norm. So it is not "nonzero means feature". The closest analogue to A_j is W_L, which is "well approximated by a sum of sinusoidal functions of the key frequencies".
- arXiv 2311.07568 (abs + HTML v2). ICLR 2024 (Spotlight). Title and author order (Morwani, Edelman, Oncescu, Zhao, Kakade) match the bib.
  - Abstract: "the principle of margin maximization alone can be used to fully specify the features learned by the network ... the trained networks utilize Fourier features to perform modular addition". Here Fourier frequencies are called features.
  - Thm 7 / informal theorem: for every neuron, "the Fourier spectra of the input and output weight vectors are supported only on that frequency", and "for every frequency ... at least one neuron in the network uses this frequency". A neuron "uses" a frequency when its weight spectrum is supported there. That is defined per neuron, not by a network-level logit amplitude.

## Verdict
Upheld: PARTIALLY_SUPPORTED. Both citations exist and their bib fields are correct.
- Morwani supports calling frequencies "features".
- Nanda supports the idea of frequencies as the units, but gives no "feature" terminology and uses a thresholded criterion.
- Neither paper defines a feature as "nonzero amplitude A_j of the logit table". "As in" overstates this.

The checker's rewording is appropriate: an exact counterpart of Nanda's "key frequencies" and of the frequencies neurons "use" in Morwani. Adding the footnote about ablation thresholds in trained nets is worthwhile.

No better or more canonical citation is needed. Chughtai et al. 2023 is already in the bib and could be added for the group case (irreps).

## Side finding (outside this item)
refs.bib has `mccracken2025` = "Emergence of sparse and dense circuits in modular arithmetic networks", author "McCracken, Andrew and others", arXiv:2505.18266. arXiv 2505.18266 (v1 is the only version, fetched) is actually "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks" by Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup and Jonathan Love. The title and the first name are both wrong. Check this under item c6-mccracken-transition.
