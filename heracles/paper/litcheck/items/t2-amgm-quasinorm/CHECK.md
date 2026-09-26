# T2-amgm-quasinorm: AM-GM prices amplitude A at A^{b/nu} (weight decay on degree-nu products = l_{b/nu} quasi-norm)

Verdict: KNOWN. This is a standard, published technique. The paper cites nothing at approach.tex:45 and must add citations.

## Our statement
approach.tex:45 says "The exponent comes from homogeneity: a feature is a product of nu weight factors, and at fixed product their total L_{2,b} cost is least when they are equal (AM-GM), so amplitude A costs a multiple of A^{b/nu}=A^{1/s}." It is used in Thm universal (approach.tex:64) and cyclic.tex:49.

## Prior art (all fetched this session)
1. **Tibshirani 2021**, "Equivalences Between Sparse Models and Neural Networks", working notes, https://www.stat.cmu.edu/~ryantibs/papers/sparsitynn.pdf. Sec 2 (two-layer case): min_{ab=c}(a^2+b^2)=2|c|, "this is just saying that the arithmetic mean-geometric mean (AM-GM) inequality ... is tight when a = b". Next section (k-layer case): min_{prod a_l=c} sum a_l^2 = k|c|^{2/k}, again via AM-GM, which gives the l_p penalty with p = 2/k and is "nonconvex" for k>2 (eq. (8) <-> (9)). This is exactly our step at b=2, stated with AM-GM by name. The note itself credits Neyshabur et al. 2015 Thm 1 for the two-layer ReLU group version.
2. **Hoff 2017**, CSDA 115:186-198, arXiv 1611.00040. "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties" through a Hadamard (multiplicative) reparametrisation. This is the earliest statistics source for the l_{2/k} bridge.
3. **Gunasekar, Lee, Soudry, Srebro, NeurIPS 2018**, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", arXiv 1806.00468. Thm 2a: for depth-L linear convnets, the GD limit direction is a first-order stationary point of min ||beta_hat||_{2/L} s.t. y_n<beta,x_n> >= 1, where beta_hat is the discrete Fourier transform. This is the CLOSEST analogue to our setting: max-margin, an l_{2/L} quasi-norm on FOURIER coefficients, and sparsity in frequency. It is implicit bias (GD), not explicit wd, and uses linear predictors, not modular addition.
4. **Dai, Karzand, Srebro, NeurIPS 2021**, "Representation Costs of Linear Neural Networks". Covers the representation cost induced by l2 on parameters, including l_p costs and which parametrisations produce them.
5. **Kolb, Weber, Bischl, Rugamer, ICLR 2025**, arXiv 2502.02496, Thm 1: L2 regularisation on D factors has the same global and local minima as L_{2/D} regularisation of the collapsed weights w = omega_1 ⊙ ... ⊙ omega_D. They credit Hoff 2017, Grandvalet 1998, Tibshirani 2021 and Ziyin & Wang 2023.
6. (from the sweep, not re-fetched here) Ongie-Parhi 2026 arXiv 2606.14954: the ReLU representation cost is a quasi-seminorm.
- Woodworth et al. 2020 (arXiv 2002.09277) is implicit bias from init scale on depth-D diagonal nets. It is only related, so do not cite it for this step.
- Grandvalet 1998 (ICANN, adaptive ridge = lasso) was found by search only. Its bib is unconfirmed, so no bib is given.

## What is ours
- Using the general L_{2,b} per-neuron cost (exponent b/nu in place of 2/nu). This is a one-line extension of the same AM-GM step and does not count as novelty.
- Applying the step to the Fourier amplitudes of a multilinear/power neuron, with the single-neuron constant C_b (approach.tex:56), and summing via subadditivity so the bound holds for arbitrary networks (Thm universal).
- Coupling the l_{1/s} budget to the worst-offset coverage programme Gamma_s. The sparsity count K* and the log p law come out of this coupling, not out of the quasi-norm itself.

## Recommended edit
At approach.tex:45, after "...costs a multiple of A^{b/nu}=A^{1/s}." add:
"This is the familiar fact that $\ell_2$ weight decay on a product of $k$ factors is an $\ell_{2/k}$ quasi-norm penalty on the product \citep{hoff2017lasso,tibshirani2021equivalences,dai2021representation,kolb2025deep}. \citet{gunasekar2018implicit} find the same bridge penalty on Fourier coefficients as the implicit bias of deep linear convolutional networks. What is new here is pairing it with the worst-offset coverage of \eqref{eq:programme}."
