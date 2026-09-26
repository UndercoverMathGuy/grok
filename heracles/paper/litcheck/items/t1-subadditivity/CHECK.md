# T1-subadditivity: check

Item: in the proof of thm:universal (approach.tex, around l.64-67), subadditivity of t^r for r=b/nu<=1 gives
sum_j A_j^r <= sum_{i,j} |beta_ij|^r, i.e. splitting a feature over neurons never makes it cheaper.

Verdict: KNOWN (standard). The inequality (a+b)^r <= a^r + b^r for 0<r<=1 is the p-subadditivity that defines
a p-norm (p-normed / p-Banach space). This is textbook material (Kalton-Peck-Roberts, An F-space Sampler, LMS LNS 89).
In ML it is exactly the statement that the weight-decay representation cost of a product/deep parametrisation is subadditive
("concatenating neurons"), so that its root is an l_{2/L}-type p-(semi)norm.

## Sources fetched
- Ongie & Parhi, arXiv 2606.14954 (2026), "Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks".
  Def 3.5: Q is a p-seminorm if Q(f+g)^p <= Q(f)^p + Q(g)^p, 0<p<=1. Prop 3.6: K absolutely p-homogeneous and subadditive
  gives K^{1/p} a p-seminorm. Prop 3.13: parameter q-subadditive + (q,p)-homogeneous costs give subadditive representation cost.
  Section 8 (after 8.30): "the subadditivity follows by concatenating neurons". Thm 9.3: for depth L>=2 the cost R_L is a (2/L)-seminorm.
  This is the same mechanism: a merged feature costs at most the sum of its pieces.
- Gunasekar, Lee, Soudry, Srebro, arXiv 1806.00468 (NeurIPS 2018), "Implicit Bias of Gradient Descent on Linear Convolutional Networks".
  Abstract: depth-L linear conv nets converge to a predictor related to the l_{2/L} bridge penalty in the frequency domain.
  This is the closest precedent: product parametrisation plus l2 gives an l_{2/L} quasi-norm on Fourier coefficients.
- Kalton, Peck, Roberts, An F-space Sampler, LMS Lecture Note Series 89, Cambridge UP (abebooks lists 1985; the usual citation is 1984). This is the canonical reference for p-normed spaces.
  Bibliographic data confirmed; the book text was not read.

## Our contribution
The inequality itself is not ours. What is ours is using it to show that the universal bound needs no hypothesis on the neurons.
Mixed-frequency neurons, interference and inactive neurons all pay at least the price A^r of an unsplit feature, and the neuron-level cost
is taken from our AM-GM/C_b step.

## Recommended change
approach.tex l.64: after "As t -> t^r is subadditive", add "(the p-norm inequality for 0<r<=1; cf. \citealp{kalton1984fspace})".
At l.67 add: "This is the same subadditivity that makes weight-decay representation costs of deep and product parametrisations
$\ell_{2/L}$-type $p$-norms \citep{gunasekar2018implicit,ongie2026representation}." Do not claim the technique as new.
