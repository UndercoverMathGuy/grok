# C9-feature-definition: check of a cited claim

The sentence is at approach.tex:27: "We call $A_j:=|\beta_j|$ the amplitude of frequency $j$; a frequency with $A_j>0$ is a feature, as in \citet{nanda2023progress,morwani2024feature}."

## Verdict: PARTIALLY_SUPPORTED

Both papers treat Fourier frequencies as the units of the learned algorithm. Neither paper defines a feature as a frequency with nonzero logit amplitude, so "as in" is too strong.

## Nanda, Chan, Lieberum, Smith, Steinhardt (ICLR 2023, arXiv 2301.05217)
Source: PDF fetched from https://arxiv.org/pdf/2301.05217. The header reads "Published as a conference paper at ICLR 2023".
- Sec. 4.1 says: "The embedding matrix W_E is sparse in the Fourier basis -- it only has significant nonnegligible norm at 6 frequencies. Of these frequencies, only 5 appear to be used significantly in later parts of the model (corresponding to k in {14,35,41,42,52}). We dub these the key frequencies of the model."
- Fig. 3 caption: "W_L is well-approximated by the 5 key frequencies". Sec. 4.4 ablates the key frequencies and reports the restricted and excluded losses.
- The word "feature" never appears in the paper with this meaning (its only occurrence, in an appendix, is "some feature of the loss landscape").
- The definition is empirical and uses a threshold ("significant", "used ... in later parts"). One frequency has nonzero embedding norm but is not a key frequency. So "exactly nonzero means feature" is not Nanda's criterion. The notion closest to ours is being present in the neuron-logit map W_L, and that matches our logit-level A_j.

## Morwani, Edelman, Oncescu, Zhao, Kakade (ICLR 2024, arXiv 2311.07568)
Source: PDF fetched from https://arxiv.org/pdf/2311.07568v2 and HTML from https://arxiv.org/html/2311.07568.
- Abstract: "analytically characterize the features learned ... the trained networks utilize Fourier features to perform modular addition".
- Informal theorem and Theorem 7: "For every neuron, there exists a frequency such that the Fourier spectra of the input and output weight vectors are supported only on that frequency" and "at least one neuron in the network uses this frequency".
- Here "feature" means the Fourier or sinusoidal structure of the weights. A frequency is "used" by a neuron whose spectrum is supported on it. That is a notion at the level of neurons and supports, not an amplitude of the network's logits. In spirit it is close to "the frequency has A_j>0", but it is not the same definition.

## Fix
Replace "a frequency with $A_j>0$ is a \emph{feature}, as in \citet{nanda2023progress,morwani2024feature}" with:
"a frequency with $A_j>0$ is a \emph{feature}. This is the exact counterpart of the empirical `key frequencies' of \citet{nanda2023progress} and of the frequencies that neurons `use' in \citet{morwani2024feature}."
Optionally, a footnote could say that trained networks have small nonzero amplitudes at every frequency, so the empirical count uses an ablation threshold (Nanda's protocol).
No new citations are needed.
