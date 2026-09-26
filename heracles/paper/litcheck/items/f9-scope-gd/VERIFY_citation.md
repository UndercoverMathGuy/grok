# VERIFY (citation lens): F9-scope-gd, task 73

Claim (approach.tex l.3): maximum-margin solutions are studied as a proxy for regularized global minimizers (Wei et al. 2019, Thm 4.1). "Nothing here concerns whether gradient descent finds them."

Checker's verdict: SUPPORTED. **Upheld.** The sentence only sets scope and asserts nothing about GD, so it needs no citation. Adding GD-to-KKT citations is optional context. The checker's two citations are real and correct. The proposed replacement sentence overstates them slightly (see "Wording fix").

## Citations fetched this session

1. **lyu2020gradient.** arXiv 1906.05890. The abs page gives the journal-ref as ICLR 2020; authors Kaifeng Lyu, Jian Li. The refs.bib entry is correct. I read the PDF text directly.
   - Thm 4.4 (Cor. of A.8/E.3): gradient flow under (A1)-(A4), or GD under (S1),(A2)-(A4),(S5). Any limit point of theta(t)/||theta||_2 is along the direction of a KKT point of (P): min 1/2||theta||_2^2 s.t. q_n(theta) >= 1. The result is for unregularized loss only.
   - Sec 4.3: "it is easy to construct examples that gradient descent does not lead to a normalized margin that is globally optimal. Thus, proving the convergence to KKT points is perhaps the best we can hope for". This is verbatim.
   - Sec 4 and App G cover multi-class cross-entropy, with q_n = true score minus the max other score. This is the same margin as g(theta,x,y) in the paper. Thm 4.1 and 4.4 still hold "with a slightly different definition of (P)".
   - Product/power networks are homogeneous and C^2, so they meet (S1) and (A2).
   - Caveat: the result covers the L2 norm (b=2) with no weight decay. It says nothing about the L_{2,b} problem for b != 2.
2. **ji2020directional.** arXiv 2006.06657. Comment: "To appear at NeurIPS 2020". OpenAlex lists it under Neural Information Processing Systems 2020. I read the PDF.
   - Sec 2 is binary only: y_i in {-1,+1}, Phi: R^d -> R.
   - Thm 3.1: under Assumption 2.1 (locally Lipschitz, L-homogeneous, definable) and 2.3 (L(W0) < l(0)), for l_exp and l_log, W_t/||W_t|| converges. This is gradient flow only.
   - Sec 5 says: "We also handled only the binary classification case, however our tools should directly allow for cross-entropy", and it lists finite-step GD as future work.
   - The bibtex is correct.
3. **Vardi, Shamir, Srebro, "On Margin Maximization in Linear and ReLU Networks".** arXiv 2110.02732.
   - **Venue now confirmed.** Semantic Scholar gives DOI 10.52202/068431-2683. That DOI resolves (CSL-JSON via doi.org) to *Advances in Neural Information Processing Systems 35* (2022), pp. 37024-37036, by Vardi, Shamir and Srebro.
   - Abstract: "in many cases, the KKT point is not even a local optimum of the max margin problem".
   - This is safe to cite as NeurIPS 2022.

## Better or additional citation (optional)

4. **Lyu, Jin, Li, Du, Lee, Hu, "Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking".** ICLR 2024 (arXiv comment), arXiv 2311.18817. I read the PDF.
   - Thm 3.5 is gradient flow **with weight decay lambda** from large initialization alpha -> inf. At some time t_k in [log(alpha)/lambda, (1+c)log(alpha)/lambda], every limit point of the normalized iterate is along a KKT point of (R1): min 1/2||theta||_2^2 s.t. y_i f_i(theta) >= 1 (binary).
   - The authors add that the KKT condition "is not sufficient in general".
   - This is the weight-decay, grokking-setting analogue of Lyu-Li. It fits better than Lyu-Li alone because the paper's proxy is *regularized* minimizers and its testbed is grokking. The caveats are binary labels and L2 only.

## Wording fix to the checker's recommended sentence

The checker proposed: "Gradient flow and descent on cross-entropy converge in direction to KKT points ... [lyu2020gradient, ji2020directional]". Neither paper proves this as worded:
- Lyu-Li gives "every limit point is along a KKT direction", not convergence.
- Ji-Telgarsky proves convergence only for binary gradient flow.
- Both are unregularized, and both are L2 (b=2) only.

Suggested replacement:

> "For $b=2$, every limit direction of unregularized gradient flow or descent on cross-entropy is a KKT point of the $L_2$ maximum-margin problem \citep[Theorem~4.4 and Appendix~G]{lyu2020gradient} (directions converge for binary gradient flow \citep{ji2020directional}; with small weight decay, see \citealp{lyu2024dichotomy}), and such KKT points need not be global or even local maximizers \citep{vardi2022margin}. Nothing here concerns whether gradient descent finds them."

A shorter alternative keeps the original sentence and adds "(for $b=2$, gradient descent is only known to reach KKT points of the margin problem \citep{lyu2020gradient,vardi2022margin})".

## Operational note
The session-wide WebSearch budget is exhausted: 200 of 200 calls, and WebSearch now refuses. I used direct fetches and APIs instead: arXiv, doi.org, OpenAlex and Semantic Scholar. DBLP returned non-JSON, which looks like rate limiting under parallel load.
