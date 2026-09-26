# VERIFY (citation lens), task 96, U8-nuclear-norm

Sentence: appendix.tex l.285, "the shared layer is priced by a nuclear norm that rewards carrying many frequencies in one neuron". refs.bib has no Srebro, Ongie, Parkinson or Gunasekar entry (checked with grep).

## Checker's citations, re-checked
1. srebro2004mmmf. EXISTS; the bib is correct. NeurIPS page: "Maximum-Margin Matrix Factorization", Srebro, Rennie, Jaakkola, Advances in NIPS 17 (NIPS 2004). I read the PDF (pdftotext). Its **Lemma 1** is ||X||_Sigma = min_{X=UV'} ||U||_F ||V||_F = min_{X=UV'} 1/2(||U||_F^2 + ||V||_F^2). This is the actual pricing fact: Frobenius (L2) cost on the two factors of a linear product equals the trace (nuclear) norm. The item's "2005" is the print year. The NeurIPS site gives 2004, so keep 2004.
2. ongie2022linear. EXISTS: arXiv 2202.00856, v1 only (2 Feb 2022), with no journal-ref on the abs page. The intro reads verbatim: "adding linear layers to a ReLU network with weight decay regularization is akin to using a two-layer ReLU network with nuclear or Schatten norm regularization on the weight matrix". Lemma 3.2 gives R_L(f) = min (1/L)||a||^2 + ((L-1)/L)||W||_{S_q}^q with q = 2/(L-1). Before Lemma 3.8 it says "the Phi_3 loss involves minimizing over the nuclear norm of a matrix". Architecture (eq. 3): the input layer W is replaced by a product W_{L-1}...W_1 in front of the ReLU units. That is exactly "a shared linear layer in front of the neurons". Confirmed.
3. gunasekar2017implicit. EXISTS; the bib is correct (NIPS 30, 2017, five authors). The abstract is about gradient descent's implicit bias toward minimum nuclear norm. It is tangential here, because the sentence is about the explicit weight-decay price. Optional, as the checker said.

## Better or more canonical citation (new)
- **Parkinson, Ongie, Willett, "ReLU Neural Networks with Linear Layers Are Biased towards Single- and Multi-index Models", SIAM J. Math. Data Sci. 7(3):1021-1052, 2025, DOI 10.1137/24M1672158**. I checked this with Crossref metadata and with the arXiv 2305.15598v4 PDF. It is the peer-reviewed successor of Ongie-Willett 2022.
  - Lemma 2.1 is the same statement (q = 2/(L-1)).
  - The intro says "adding linear layers to a ReLU network while using l2-regularization (weight decay) is equivalent to fitting a two-layer ReLU network with nuclear or Schatten norm regularization on the innermost weight matrix".
  - It attributes the P=2 variational identity to Srebro et al. 2004 (its ref [70]).
  - Recommendation: cite this instead of, or alongside, the unpublished arXiv preprint.
- Recht, Fazel, Parrilo, SIAM Review 52(3):471-501, 2010 (arXiv 0706.4138). Problem (5.1) is min 1/2(||L||_F^2+||R||_F^2) s.t. A(LR')=b, and Lemma 5.1 says it is equivalent to nuclear-norm minimization. A canonical alternative to Srebro; optional.

## Caveats
- **Scope.** The Ongie/Parkinson results are for 1-homogeneous ReLU units. The rebalancing against the outer weights (the 1/L, (L-1)/L split) does not carry over to degree-nu product or power neurons. The part that does carry over is pure linear algebra: the shared layer times the neuron input weights, priced by Frobenius cost, gives a nuclear norm (Srebro Lemma 1). So **Srebro is the load-bearing citation**, and Parkinson et al. is the network-context citation. Do not quote Lemma 2.1's formula as applying to our neurons.
- **Gloss.** No source states "rewards carrying many frequencies in one neuron". Also, the nuclear norm is invariant under left-orthogonal mixing, so it is *indifferent to* how frequency content is spread across neurons. It does not actively reward multi-frequency neurons. It removes the per-neuron penalty that L_{2,b}, b>2, puts on them. "Rewards" may be an overstatement. Consider "does not penalize" or "is blind to", or keep it and justify it in the text.

## Tooling note, relevant to workflow slowness
In this session WebSearch returned "used its web search budget (200 of 200 WebSearch calls)". Every verifier in this run now has search disabled and has to fall back to WebFetch or curl. Other failures:
- dblp WebFetch timed out after 60 s.
- Semantic Scholar returned 429.
- WebFetch cannot parse PDFs (binary), so each PDF needs a pdftotext round trip.
- Verifiers share one scratchpad with generic file names (ongie.txt and srebro.txt already existed from other agents), so they can overwrite each other.
