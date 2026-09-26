# C3-wei-thm41: check of the Wei et al. [Theorem 4.1] citation in approach.tex

Verdict: PARTIALLY_SUPPORTED. The mathematics carries over. The theorem number is wrong for the multi-class setting.

Source: Wei, Lee, Liu, Ma, "Regularization Matters: Generalization and Optimization of Neural Nets v.s. their Induced Kernel", NeurIPS 32 (2019). I read the full text from arXiv 1810.05369 (PDF) and confirmed the venue on proceedings.neurips.cc.

## What Wei et al. state
- Setup (Sec. 4): f(.;Theta) is a-positive-homogeneous in Theta, f(x;cTheta)=c^a f(x;Theta) for c>0, and continuous in Theta. The regulariser is "some general norm ||.||", with L_lambda = logistic loss + lambda ||Theta||^r for a fixed r>0 (eq. 4.1). The normalized margin is gamma_lambda = min_i y_i f(x_i; Theta_lambda/||Theta_lambda||), and gamma* = max_{||Theta||<=1} min_i y_i f.
- Theorem 4.1 is binary logistic only: "Assume the training data is separable by a network f(.;Theta*) in F with an optimal normalized margin gamma*>0. Then the normalized margin of the global optimum of the weakly-regularized objective (equation 4.1) converges to gamma* as the regularization goes to zero."
- The paper says: "Though the result in this section is stated for binary classification, it extends to the multi-class setting with cross-entropy loss ... Section C."
- Theorem C.1 (App. C, "Multi-class Setting") uses eq. C.1 = softmax cross-entropy + lambda||Theta||^r. The margin is f_y - max_{j!=y} f_j of the normalized Theta, and gamma* is taken over ||Theta||<=1: "Assume gamma*>0 in the multi-class setting with cross entropy loss. Then as lambda -> 0, gamma_lambda -> gamma*." Theorem 4.1 is proved by reduction to C.1.

## Hypotheses matched against our setting
- Homogeneity: our networks are nu-positive-homogeneous in theta, so a = nu. They are polynomial, hence continuous. OK.
- Norm: ||theta||_{2,b} = (sum_i ||w_i||_2^b)^{1/b} with 2 <= b <= nu (prelim.tex). This is an l_b norm of l_2 norms with b >= 1, so it is a genuine norm and Wei's "general norm" covers it for every b in range, including b != nu. The result would fail to apply only for b < 1, where L_{2,b} is a quasi-norm, and that range is excluded. OK.
- Exponent: our regulariser is lambda ||theta||_{2,b}^b, so r = b > 0. OK.
- Normalized margin: Wei's margin of Theta/||Theta|| equals h(theta)/||theta||^nu by homogeneity, which is our definition. gamma* is defined over the unit ball, as ours is. OK.
- Multi-class cross-entropy: modular addition has p classes, so the right statement is Theorem C.1, not 4.1. MISMATCH in the pointer only.
- Separability: gamma*_b > 0 holds for modular addition (Morwani et al. construct positive-margin nets). OK. The paper could say this in one clause.
- "Global minimizers": Wei's statement is for any Theta_lambda in argmin L_lambda, and Claim C.3 shows a minimizer exists. OK.
- The paper's disclaimer "Nothing here concerns whether gradient descent finds them" matches Wei's scope, which is statics only. OK.

## Fix
In approach.tex, replace "By \citet[Theorem~4.1]{wei2019regularization}" with
"By \citet[Theorem~4.1, and Theorem~C.1 for multi-class cross-entropy]{wei2019regularization}, which applies to any norm and any positively homogeneous network,"
or simply use [Theorem~C.1].
Optionally add "(here $\gamma^*_b>0$)" to state the separability hypothesis.

Bib: add volume 32. The title in the bib already matches the proceedings page.
