# The convex case plus an explicit correction (exact, every p, n, b, every network)

## The identity

Notation:
- a_i = ||omega_i||^nu and m_i = a_i / sum_j a_j are the neuron masses; they do not depend on s.
- Gamma_1 = p/(p-1) is the programme value at s = 1.
- beta_l = sum_i beta_il are the clean coefficients; beta_il are the per-neuron atoms (`Net.atoms()`).

For every product network with any margin h (even negative):

    gamma(theta; s) = B_p Gamma_1 * [ 1 - E_m(sigma) - d_prog - d_int ] * exp( -(s-1) H_{1/s}(m) ).

The four ingredients are:
- sigma_i = 1 - sum_l Re(beta_il) / (B_p a_i), in [0, 2]. This is neuron i's clean misalignment. It is positive
  through mixed frequencies (non-own mode mass), imbalance of the factors, or atom phase off the feature axis
  (cancellation). It is charged per neuron, so it cannot be compensated.
- d_prog = [Gamma_1 sum_l Re beta_l - min_delta sum_l Re(beta_l) kappa_l(delta)] / (B_p Gamma_1 sum_i a_i) >= 0.
  This is the programme deficit of the clean profile at the convex end, i.e. how far the allocation is from
  uniform coverage.
- d_int = [min_delta sum_l Re(beta_l) kappa_l(delta) - h] / (B_p Gamma_1 sum_i a_i) >= 0. This is the interference
  deficit: x-mean of the gap minus its minimum. It is the ONLY term where compensation between features acts.
- H_{1/s}(m) is the Renyi entropy of order 1/s = b/nu of the neuron masses.

## Proof

1. Use the convex-end dual q* = uniform(x) x uniform(delta). With E_delta(1 - rho^{-l delta}) = p/(p-1) = Gamma_1:
   E_{q*} g = Gamma_1 sum_l Re beta_l = Gamma_1 sum_i sum_l Re beta_il.
2. Hence B_p Gamma_1 sum_i a_i - h = Gamma_1 sum_i (B_p a_i - sum_l Re beta_il) + (E_{q*} g - h).
3. The sign facts:
   - sigma_i >= 0 is the single-neuron budget sum_l |beta_il| <= B_p ||omega_i||^nu (SPLIT E7(i), with r = 1).
   - d_prog >= 0 because a minimum over delta is at most the mean over delta.
   - d_int >= 0 by Lemma 2: pairing delta with -delta, the x-average of the gap is sum_l Re(beta_l) kappa_l(delta).
4. Divide by B_p Gamma_1 sum_i a_i and apply the Renyi temperature identity (E_width W1).

**Check** (`ident_check.py`, `ident_check.out`).
- Exact to machine precision on random networks: (5,2), (7,2), (5,3) and (3,2).
- Exact on IF optima. At (5,2) and (7,2) every deficit is 0. At (11,2,2), d_prog = 0.316 and the other deficits are 0.
- Exact on the certified mixed networks of SPLIT.

## The same structure one level up

The programme has exactly the same form at the feature level (E_dual D1(iv)):

    Gamma_s = Gamma_1 * max over yhat in the simplex of [ 1 - d_prog(yhat) ] * exp( -(s-1) H_{1/s}(yhat) ).

The single-feature price pi is the within-feature version: the best value of
[1 - E sigma - d_int] * exp(-(s-1) H_{1/s}) over single features, relative to kappa_min.

**So RD is a factorisation statement.** The neuron-level trade-off (convex deficits against Renyi concentration)
never beats the product of two separate trade-offs:
- the feature-level one, which is the programme;
- the within-feature one, which is the price.

Decoupled networks are product distributions m = z (x) m^(1). For them the Renyi entropy is additive and the trade-off
factorises exactly.
