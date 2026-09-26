# Can the dense theorem simply be continued down to q=2?

This note gives an additional local theorem at nu=3 and a precise limitation
of continuation. The explicit unrestricted interval is in `EXPLICIT_ENDPOINT.md`.

## 1. Dense local optimality for every 2<=q<3

For every odd prime p, the dense three-neuron-per-frequency construction is
a strict local maximum of normalized margin at EVERY 2<=q<3, modulo scaling,
phase motions, neuron permutations, and signs that leave each tensor unchanged.
Extra zero neurons and all mixed-frequency perturbations are allowed.

The q=2 theorem, including its exact quadratic certificate, is already proved
in `LOCAL_THEOREM.md`. The extension below uses only the power-mean inequality.

Let N0=3(p-1)/2, and first consider exactly N0 neurons. Write rho_i=||theta_i||_2.
For q>=2, convexity gives

    P_q=sum rho_i^q >= N0^(1-q/2) P_2^(q/2),
    gamma_q(theta) <= N0^(3/2-3/q) gamma_2(theta).           (1.1)

The dense candidate has all rho_i equal, so equality holds in (1.1) there.
We work in a positive-margin neighborhood, so multiplication by the margin
preserves the direction of this inequality.
Its strict q=2 local optimality immediately implies strict q local optimality
in the same neighborhood of its scaling/phase family. This argument handles
all main-neuron perturbations, including impurities.

### Additional neurons

Normalize a nearby network to the dense candidate's P_q value P0(q).
Let s be the penalty used by additional small neurons. The main N0 neurons
then have penalty P0-s. Their local bound gives margin at most

    gamma_dense(q) (P0-s)^(3/q).

A neuron of penalty P_i has unit tensor amplitude at most 3^(-3/2)P_i^(3/q),
by balancing/AM-GM. Its contribution to any gap has magnitude at most twice
that amplitude. Since 3/q>1, summing over the extra neurons gives gap change
at most 2*3^(-3/2)*s^(3/q). Thus the full margin is at most

    gamma_dense(q) (P0-s)^(3/q) +2*3^(-3/2)*s^(3/q).

For sufficiently small positive s, this is strictly below the dense margin:
the first term loses a positive multiple of s, while the second is o(s).
At s=0 the main-neuron theorem applies. This also proves uniform local
optimality as q varies in any compact subinterval of [2,3).

## 2. The endpoint minimal frames at nu=3 are the same phase family

To apply the local theorem to every minimal endpoint representation, no
unproved phase classification is needed. At a fixed frequency such a
representation has three equal coefficients and phases v_i,w_i satisfying

    sum v_i=sum w_i=sum v_i*w_i=0,    |v_i|=|w_i|=1.

Three unit complex numbers summing to zero are a rotated equilateral triple.
After relabeling, v_i=a*zeta^i with zeta=exp(2*pi*i/3). Similarly w_i=b*zeta^{pi(i)}.
Every permutation of three residues is affine, pi(i)=s*i+t mod 3 with s=+1 or -1.
The product cancellation excludes s=-1, since then all v_i*w_i are equal.
Thus w_i=b'*zeta^i. Taking half phases recovers exactly the standard block,
up to slot signs whose product is absorbed by the aligned readout.

Consequently the compact minimal endpoint set in `ENDPOINT_NEIGHBORHOOD.md`
is covered by the strict local theorem above.

## 3. If global optimality is lost, a distant competitor must tie it first

Suppose the dense construction fails to be globally optimal somewhere in
2<=q<3. Starting from the interval proved in `EXPLICIT_ENDPOINT.md`, move q
downward, and let q_c be the lower boundary of the connected interval of
global dense optimality adjoining 3. If q_c>2, then at q_c:

* the dense construction is still globally optimal;
* there is another global optimizer outside a fixed neighborhood of the
  dense minimal-frame family;
* the dense construction remains a strict local optimum.

Proof. Width compression gives a fixed finite width d. For q in a compact
interval containing q_c, bounded objective values give a common compact
coefficient/atom domain. The global minimum is continuous in q: the upper
bound follows by retaining a fixed feasible optimizer, and the lower bound
by subsequential compactness of minimizing representations.

Choose q_j<q_c approaching q_c at which a strictly better competitor exists,
and take global minimizers. A subsequence converges to an optimizer at q_c,
whose value equals the dense value by continuity. It cannot converge to the
dense family, because section 1 supplies a uniform local comparison for all
nearby q_j; that would contradict strict improvement. The limiting competitor
therefore remains a positive distance from that family.

This is a statement about representations in the fixed compact parametrization,
allowing permutations, phase motions, signs, and arbitrary directions on zero
coefficients. It need not imply a positive distance between the output tensors.

This does NOT exclude a global transition. The distant competitor might be
a different clean spectrum, or might carry junk. It explains why local
stability cannot bootstrap the endpoint theorem through the whole dense
ansatz regime: it is already present throughout 2<=q<3, including primes
where the dense construction is globally beaten by a sparse ansatz.

## 4. The remaining promising target

For p=5 the clean ansatz is dense throughout 2<=q<=3 (its ansatz transition
is below q=2; see `ARGUMENT.md`, section 2.4). A failure of unrestricted dense
optimality in that range would therefore require a distant, junk-carrying
competitor to tie the dense solution at some exponent below the explicit
interval. Excluding that specific type of tie is a sharper target than
another local Hessian calculation. No exclusion is proved here.
