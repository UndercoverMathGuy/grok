# F9-scope-gd: "Nothing here concerns whether gradient descent finds them"

Location: approach.tex, par. "Maximum margin" (line 3). The paragraph cites Wei et al. 2019 Thm 4.1 for the claim that regularized global minimizers tend to the maximum margin. It does not discuss GD. refs.bib contains `lyu2020gradient`, but no section cites it (checked with grep).

## Verdict: SUPPORTED (the scoping is accurate); CITATIONS FOUND to back it

### Lyu & Li, ICLR 2020, arXiv 1906.05890. Fetched the PDF; the ICLR 2020 header is confirmed.
- Thm 4.1 (Cor. of Thm A.7): gradient flow on homogeneous nets. A smoothed normalized margin is non-decreasing once the loss is below a threshold.
- Thm 4.4 (Cor. of Thm A.8, E.3): for gradient flow, or GD under (S1),(A2)-(A4),(S5), "any limit point of theta(t)/||theta(t)||_2 is along the direction of a KKT point of (P): min 1/2||theta||_2^2 s.t. q_n(theta) >= 1". App. G extends this to multi-class cross-entropy with q_n = the logit-difference margin, which is the same margin definition we use.
- Sec. 4.3, a direct caveat: "if we do not make further assumptions on q_n(theta), then it is easy to construct examples that gradient descent does not lead to a normalized margin that is globally optimal. Thus, proving the convergence to KKT points is perhaps the best we can hope for".
- Scope caveat for our paper: this result is for the plain L2 norm, i.e. b = 2, where the parameter norm equals the L_{2,2} norm. For b != 2 it gives KKT points of the L2 problem, not of the L_{2,b} problem (see Tsilivis et al. ICLR 2025 for algorithm-dependent norms, already in the sweep).

### Ji & Telgarsky, NeurIPS 2020, arXiv 2006.06657. Fetched the abstract and PDF; the arXiv page gives the venue as NeurIPS 2020.
- Thm 3.1: under Assumptions 2.1 and 2.3 (homogeneous, definable network; loss below the initial threshold), for the exp and logistic losses, W_t/||W_t|| has finite length and so converges. The paper covers binary classification and gradient flow only (it says "We also handled only the binary ..."). Directional convergence therefore covers only the binary case. The multi-class GD-to-KKT citation is Lyu-Li App. G.

### Vardi, Shamir & Srebro, "On Margin Maximization in Linear and ReLU Networks", arXiv 2110.02732. Fetched the abstract.
- Abstract: "in many cases, the KKT point is not even a local optimum of the max margin problem". This directly backs the choice not to claim that GD reaches the global maximum-margin solutions we study.
- Venue: I believe it is NeurIPS 2022, but this is UNCONFIRMED. The proceedings URL returned 404, S2 was rate-limited, and the arXiv page has no journal-ref. No bib entry is recommended until the venue is confirmed; cite the arXiv version if needed.

## Recommended change
Append to approach.tex line 3:
"Gradient flow and descent on cross-entropy converge in direction to KKT points of the $L_2$ maximum-margin problem \citep{lyu2020gradient,ji2020directional}, which need not be global maximizers; nothing here concerns whether gradient descent finds them."
Add `ji2020directional` to refs.bib. Keep in mind that the Lyu-Li KKT statement is specific to the L2 norm (b=2).
