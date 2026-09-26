# Explaining Feature Sparsity Through the Margin–Regularization Trade-off

ICLR 2027 — abstract as submitted 2026-09-19 (editable until the full-paper deadline, 2026-09-25).

**Keywords:** feature learning, margin maximization, mechanistic interpretability, regularization

**TLDR:** We explain why networks trained with weight decay have sparse features from the trade-off between margin and regularization cost.

## Abstract

Explaining the number of features in a network has been a long-desired result in mechanistic interpretability. [Morwani et al. (2024)](https://arxiv.org/abs/2311.07568v2) laid the baseline with their principle of margin-maximization, showing provably that in networks where the regularization ($L_{2,\nu}$) matches its homogeneity degree ($\nu$), all features would be used for modular addition or group composition tasks. Empirically, however, networks trained with ordinary weight decay consistently show a sparse selection of features. This paper investigates why.

Our inquiry focuses specifically on modular addition and group composition tasks in polynomial and multilinear models. Our primary theoretical finding centers on the cost of new features - a tradeoff between the logit margin of the solution and regularization. While more features improve margin, they cost the model in regularization. Hence we show that an optimal number of features can be calculated for networks based on the task, their homogeneity degree and the regularization. Further, our theory explains the transition in the number of features from dense to sparse - as seen empirically in [McCracken et al. (2025)](https://arxiv.org/abs/2505.18266) - by showing that this transition arises on changing either the homogeneity degree or regularization of the network.

Specifically, we empirically show that the normalized margin attained by these networks peaks at the predicted optimum for the network, matching the result of [Morwani et al. (2024)](https://arxiv.org/abs/2311.07568v2) for their specific case. The optimums are also consistent with the empirical results seen in [Chughtai et al. (2023)](https://arxiv.org/abs/2302.03025).
