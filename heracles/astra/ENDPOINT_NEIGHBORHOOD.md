# An unrestricted interval below the homogeneous endpoint

Follow-up: `EXPLICIT_ENDPOINT.md` now gives an explicit interval and a shorter
direct global proof, without compactness or entropy selection. The nonquantitative
argument below is retained as a separate derivation; its comments about unquantified
width refer to this proof, not to the current overall state of the project.

## Result and scope

Fix an odd prime p and a homogeneity degree nu >= 3; put n=nu-1 and h=(p-1)/2.
There exists epsilon(p,nu)>0 such that, for

    1-epsilon(p,nu) < r=q/nu < 1,    q>=2,

the dense Fourier ansatz is globally margin maximal among ALL real gated-chain
networks of arbitrary finite width, with DC, mixed frequencies, and junk allowed.
The endpoint r=1 is included in the value statement.

For r<1 in this interval, every margin-one minimum-cost network, after deleting
zero neurons, has exactly N0=nu*h neurons. Every neuron is balanced, centered in
the readout, and pure at a nonzero Fourier frequency. Each frequency has exactly
nu neurons, all with the same balanced tensor amplitude. Their same-frequency
junk cancels, and the complete centered logit tensor is

    Z0(x,c) = 1[c=x1+...+xn mod p] - 1/p.

Thus K*=h throughout this unrestricted interval. We do not classify the phase
frames beyond their cancellation and orthogonality properties.

The proof establishes existence of the interval; it does NOT provide a numerical
lower bound on its width, an interval uniform in p or nu, or the full range
2<=q<=nu. In particular it does not place ordinary q=2 inside this interval.

This is a logical proof, followed by exact checks in `endpoint_verify.py`.
The checks verify identities and representative finite Fourier constructions;
the all-parameter theorem rests on the argument below, not on those examples.

## 1. Remove the irrelevant scaling constants

A centered unit atom is a real rank-one tensor

    T(x,c)=u0(c) product_{j=1}^n uj(xj),
    ||uj||_2=1,    sum_c u0(c)=0.

Any network can be centered in its readouts and balanced across its nu slots
without changing its gaps or increasing its cost. If a_i is the product of
the slot norms, its least neuron cost is nu^(q/2) a_i^r. Therefore the margin-one
problem is equivalent to minimizing sum a_i^r over

    Z=sum_i a_i T_i,    a_i>=0,    Z_y(x)-Z_c(x)>=1 for c!=y.

Define the dense-block atom amplitude and the number of its atoms by

    a0 = (2p)^(nu/2)/(p*nu),    N0=nu*h,
    b_i=a_i/a0,    F_r=sum_i b_i^r,    B=sum_i b_i.

The candidate dense network has N0 atoms, each b_i=1, hence F_r=N0 for EVERY r.
Explicitly, for each k use the nu atoms (2.5) with phi_j=pi*ell/nu in
every input slot, ell=0,...,nu-1. A flipped nonempty sign pattern T contributes
the root sum sum_ell exp(2*pi*i*ell*|T|/nu)=0 because 1<=|T|<=nu-1.
The clean coefficient at each k is a0*nu*(2p)^(-nu/2)=1/p, giving Z0.
Its penalty and normalized margin are

    P_dense(q)=nu^(q/2) N0 a0^r,
    gamma_dense(q)=1/[nu^(nu/2) N0^(1/r) a0]
                  =p*nu / [(2p*nu)^(nu/2) (nu*h)^(nu/q)].       (1.1)

We will prove F_r>=N0 for all feasible decompositions when r is sufficiently
close to 1. This also proves that the dense vertex solves the ansatz programme
there; no separate ansatz phase-transition result is needed.

### Finite width and existence

There is no width compactness assumption hidden below. The centered tensor
space has dimension d=p^n(p-1). For fixed atom directions, minimize sum a_i^r
over the nonnegative coefficient fiber representing the same tensor. For r<1,
strict concavity forces the positive columns of a minimizing representation
to be linearly independent; otherwise a small two-sided coefficient variation
improves one endpoint. For r=1, a minimum-cost extreme point of the coefficient
fiber gives the same width bound. Thus width d suffices, as proved in detail
in `ARGUMENT.md`, section 4.

At fixed width d, atoms lie in a compact product of unit spheres, and bounded
F_r bounds every coefficient. Feasibility is closed. The minimum is attained.
Every global optimizer at r<1 already has at most d nonzero atoms, since a
linear dependence would give strict improvement. Pad with zero coefficients
when taking limits.

## 2. A supporting functional with an exact endpoint equality class

Use the normalized Fourier transform on the full nu-dimensional table:

    Zhat(k1,...,kn,k0)
      =p^(-nu) sum_{x,c} Z(x,c) exp[-2*pi*i*(sum kj*xj+k0*c)/p].

Let G_k=Zhat(-k,...,-k,k), for k=1,...,h, and define the real linear functional

    L(Z)=(2p)^(nu/2) Re sum_{k=1}^h G_k.

For a real unit slot u_j, set z_jk=sqrt(2p)*uhat_j(k). Parseval gives
sum_k |z_jk|^2<=1, with equality exactly when its DC component vanishes.
For the centered readout equality always holds. For a unit atom,

    L(T)=Re sum_k z_0k product_{j=1}^n conjugate(z_jk) <=1.    (2.1)

Indeed, with x_jk=|z_jk|, Holder with exponent nu gives
sum_k product_j x_jk <= product_j ||x_j||_nu <=1.

Let gbar(Z) be the average gap, uniformly over all inputs and all p-1 wrong
offsets. Fourier orthogonality gives

    gbar(Z)=[2p/(p-1)] Re sum_k G_k,
    L(Z)=a0*N0*gbar(Z).                                    (2.2)

For a feasible network Z=a0 sum b_i T_i, put E=B-N0. Then

    E>=0,
    sum_i b_i[1-L(T_i)] = B-L(Z)/a0 <= E,
    0 <= L(Z)/a0-N0 <= E.                                  (2.3)

All gaps are at least one, so their nonnegative excesses have sum controlled
by E. Centering each row recovers its logits uniquely from its gaps. Consequently

    ||Z/a0-Z0/a0||_F <= C_gap E,                            (2.4)

where C_gap depends only on p,nu. For example, the sum of all gap excesses is
at most p^n(p-1)E/N0; each row's centered logit perturbation is a linear map
of those excesses. More explicitly, append zero excess at the correct class;
the logit perturbation is the negative orthogonal projection of this excess
vector onto the centered subspace. Its Euclidean norm is at most the sum
of its nonnegative entries. We may therefore take

    C_gap=p^n(p-1)/(a0*N0).

### Equality in the atom bound

Call an atom exposed if L(T)=1. Since nu>2, equality between the ell_nu and
ell_2 norms forces each slot's positive-frequency amplitudes to be supported
at a single frequency, with no DC. Equality in Holder forces those frequencies
to agree. Equality in the real-part bound forces the clean phase to be zero.
Conversely these conditions give L(T)=1.

Thus exposed atoms have the form

    uj(x)=sqrt(2/p) cos(2*pi*k*x/p+phi_j),
    u0(c)=sqrt(2/p) cos(2*pi*k*c/p+sum_j phi_j),              (2.5)

including equivalent choices of signs and phases.

### All endpoint optimizers and the atom cap

At r=1, the candidate attains B=N0. For any optimizer, (2.3) implies that
every positive atom is exposed; (2.4) implies Z=Z0. At each frequency the
clean Fourier coefficient of Z0 is 1/p. Since a0*(2p)^(-nu/2)=1/(p*nu),
the coefficients in that frequency obey

    sum_i b_i=nu,
    sum_i b_i product_{j in T} v_ij=0 for every nonempty T,
    v_ij=exp(2*i*phi_ij).                                  (2.6)

Map each atom to its chain vector

    m_i=(1,v_i1,v_i1*v_i2,...,product_{j=1}^n v_ij),
    ||m_i||^2=nu.

The interval equations in (2.6), and the positivity of b_i, give

    sum_i b_i m_i m_i^* = nu I_nu.                         (2.7)

Each summand is positive semidefinite, so its only nonzero eigenvalue nu*b_i
is at most nu. Therefore

    0<b_i<=1 for every endpoint atom.                      (2.8)

Conversely, every exposed decomposition of Z0 is an endpoint optimizer.
No uniqueness of that decomposition is asserted.

## 3. Moving below the endpoint selects the minimal frames

Consider any sequence r_l=1-epsilon_l tending to 1 from below, and choose
arbitrary unrestricted global minimizers at these exponents. Compress/pad
them to width d. Since F_{r_l}<=N0, their coefficients are uniformly bounded
and a subsequence of the coefficients and atom directions converges.

For H(b)=sum_i b_i log b_i, with 0 log 0=0, Taylor expansion in the exponent
is uniform on any fixed bounded coefficient interval:

    F_{1-epsilon}=B-epsilon H(b)+O(epsilon^2).              (3.1)

To justify uniformity even at zero: for r>=1/2, the second derivative
b^r (log b)^2 is uniformly bounded on [0,M], after extending it by zero
at b=0. There are only d terms.

The first consequence is B->N0, so every limiting representation is an
endpoint optimizer. Its coefficients are at most one by (2.8), hence H<=0.
On the other hand F_{1-epsilon}<=N0 and E>=0 give

    E-epsilon H(b)<=C epsilon^2,
    H(b)>=-C epsilon.

Continuity of H forces H=0 at the limit. Since each positive coefficient
is at most one, equality is possible only if every positive coefficient
equals one. Their sum is N0, so precisely N0 atoms survive. Equation (2.6)
then gives precisely nu atoms per frequency; (2.7) says their chain vectors
are orthogonal.

Let M denote this compact set of minimal endpoint representations, padded
to width d and allowing permutations and all admissible phase frames.
The directions attached to its zero coefficients are arbitrary unit atoms.
We have proved:

    Every global minimizer approaches M as r approaches 1 from below. (3.2)

More formally, failure of this uniform neighborhood statement supplies a
sequence outside one fixed neighborhood, and compactness gives a limiting
member of M, a contradiction. So (3.2) holds for ALL minimizers, not just
one favorable subsequence.

This alone is not the theorem: a network with small but nonzero junk could
still beat the dense solution. The next two sections exclude that possibility.

## 4. Quantitative control of the main amplitudes

### 4.1 Exposure deficit controls distance to a pure aligned atom

There is a constant C_exp(p,nu) such that every unit atom T has an exposed
unit atom V satisfying

    ||T-V||_F <= C_exp sqrt(1-L(T)).                        (4.1)

Here is a direct proof. Write delta=1-L(T), s=sum_k product_j x_jk.
Then 1-delta<=s<=1. For EACH slot j,

    s <= max_k x_jk * sum_k product_{ell!=j} x_ell,k
      <= max_k x_jk.

The final inequality uses Holder on the nu-1>=2 remaining slots and their
ell_2 norm bounds. Thus each slot has an entry of magnitude at least 1-delta.
For sufficiently small delta, these large entries occur at the same frequency.
Indeed, if two slots have different dominant frequencies, their inner product
is at most 2*sqrt(2*delta)+2*delta, while s is at most that inner product and
at least 1-delta. These inequalities contradict each other for small delta.

At the common frequency each magnitude is at least 1-delta. Replacing every
slot by its unit pure sinusoid with the same phase costs O(sqrt(delta)) in
Euclidean distance. If alpha is the clean phase and rho is the product of
the dominant magnitudes, then

    rho*(1-cos alpha) <= s-L(T) <= delta.

For small delta, rho>=1/2; rotating the readout to make alpha zero costs
O(sqrt(delta)) too. The norm of a difference of unit rank-one tensors is
at most the sum of the slot differences, by telescoping. This proves (4.1)
for small delta. For larger delta use ||T-V||<=2 and increase the constant.
For an explicit choice, set delta_0=min(1/100,1/(2*nu)). Each slot replacement
costs at most sqrt(2*delta), the phase correction at most 2*sqrt(delta), and
(1-delta)^nu>=1-nu*delta>=1/2. Thus one may take

    C_exp=max(nu*sqrt(2)+2, 2/sqrt(delta_0)).

Near a specified exposed atom, the selected frequency is that atom's frequency.
All constants are uniform over its phases and over the finitely many frequencies.

### 4.2 Separate the N0 main atoms from the small remainder

In a sufficiently small neighborhood of M, label N0 main atoms so their
b_i are near one and their directions are near the corresponding exposed
atoms. There are nu main atoms at each frequency. All other atoms form the
remainder, with total coefficient

    t=sum_{remainder} b_i.

Both E and t are small in this neighborhood. Project each main atom to an
exposed atom V_i using (4.1), preserving its amplitude b_i and its frequency.
By Cauchy-Schwarz, (2.3), and the local boundedness of B,

    sum_main b_i ||T_i-V_i||
       <= C_exp sqrt[(sum_main b_i)*(sum_main b_i[1-L(T_i)])]
       <= C sqrt(E).

Together with (2.4) and ||sum_remainder b_i T_i||<=t, this gives

    ||sum_main b_i V_i-Z0/a0||_F <= C sqrt(E)+t.             (4.2)

The projection here is only an estimate inside the proof. We do NOT assert
that replacing the network by these atoms preserves feasibility or improves cost.

### 4.3 Square frame inversion controls every main coefficient

For each frequency form the nu-by-nu Hermitian chain matrix of its nu
projected main atoms:

    M_k=sum_{i at k} b_i m_i m_i^*.

Its diagonal entries are the clean amplitude divided by (2p)^(-nu/2), and
its off-diagonal entries are interval junk amplitudes or their conjugates.
Those are linear Fourier coefficients of the tensor on the left of (4.2).
At Z0/a0 the matrix is nu I. Hence

    ||M_k-nu I||_op <= C [sqrt(E)+t].                       (4.3)

For small E,t, let eta=||M_k-nu I||<nu/2. The matrix is positive definite.
There are exactly nu vectors in it, so their square column matrix is
invertible. The elementary inverse identity gives

    b_i = 1/(m_i^* M_k^(-1) m_i).

The eigenvalues of M_k lie between nu-eta and nu+eta and ||m_i||^2=nu.
It follows, without any phase-frame classification, that

    1-eta/nu <= b_i <= 1+eta/nu.

Summing over the N0 main atoms proves the key estimate

    sum_main (b_i-1)^2 <= C_* (E+t^2).                     (4.4)

The constants and the neighborhood are independent of r. This is the step
that controls small improvements from moving away from the endpoint set.

For completeness these constants can be explicit. In a neighborhood with
E<=1, put C_delta=C_gap+C_exp*sqrt(N0+1). The right side of (4.2) is at most
C_delta*sqrt(E)+t. Every normalized Fourier coefficient of a tensor X has
modulus at most p^(-nu/2)||X||_F. Hence

    eta <= nu*2^(nu/2) [C_delta*sqrt(E)+t],
    C_* = N0*2^(nu+1)*max(C_delta^2,1)

suffices in (4.4), provided the displayed bound on eta is less than nu/2.

## 5. A uniform cost comparison closes the neighborhood

Choose the neighborhood smaller if necessary so every main b_i lies in
[1/2,3/2], and restrict r to [1/2,1). Put epsilon=1-r. Since

    d^2/db^2 b^r = -r*epsilon*b^(r-2),

Taylor's theorem on this fixed interval gives a constant C_T, independent
of r (C_T=2 suffices), such that

    b^r >= 1+r(b-1)-C_T*epsilon*(b-1)^2.

The main coefficients sum to N0+E-t. Using (4.4) and increasing a fixed C,

    sum_main b_i^r >= N0+r(E-t)-C*epsilon*(E+t^2).

Concavity/subadditivity gives sum_remainder b_i^r>=t^r. For 0<=t<=1,

    t^r-r*t >= t-r*t = epsilon*t.

Consequently every feasible network in this neighborhood obeys

    F_r-N0 >= (r-C*epsilon)E + epsilon*t-C*epsilon*t^2.     (5.1)

Choose a fixed neighborhood with t<=min(1,1/(2C)), and then choose a fixed
epsilon_0>0 small enough that r-C*epsilon>=1/2. For example take C=2*C_*
and epsilon_0<=1/[2(C+1)]. We obtain

    F_r-N0 >= E/2 + epsilon*t/2 >=0                         (5.2)

throughout that neighborhood whenever 0<epsilon<epsilon_0.

If equality holds, E=t=0. Equation (2.3) then makes every atom exposed,
(2.4) gives Z=Z0, and (4.4) gives b_i=1 for every main atom. Thus the
network belongs to M. This is a uniform exact local comparison, including
arbitrary additional small atoms, rather than merely strict local optimality
at each separately fixed exponent.

## 6. Global conclusion

If the theorem failed arbitrarily near r=1, choose corresponding unrestricted
global minimizers. By (3.2), all sufficiently late members lie in the fixed
neighborhood used in section 5. But (5.2) forces F_r>=N0, while the dense
candidate gives F_r<=N0. Equality forces membership in M, a contradiction
to either a strictly better value or a non-minimal optimizer.

Equivalently, (3.2) supplies a positive epsilon_1 for entry into that fixed
neighborhood. Take epsilon(p,nu) less than epsilon_0, epsilon_1, and
1-2/nu if we want the entire stated interval to lie in q>=2. This proves
the result for every fixed p and nu>=3.

No quantitative epsilon_1 is extracted from the compactness argument.
The proof does not imply that the interval reaches the first sparse ansatz
transition. Extending it that far would require new estimates.

## 7. Checks on the logical dependencies

* The unrestricted endpoint is rederived above, including its full exposed
  atom class and its unique centered output tensor. We do not assume endpoint
  representations are already minimal or unique.
* The entropy argument is applied to global minimizers over a fixed compact
  width bound, with a uniform Taylor remainder at zero coefficients.
* The arbitrary-depth chain frame is used only AFTER projection to pure
  aligned atoms, where its weights are real and nonnegative. No uncontrolled
  complex mirror entries appear.
* Exact cancellation is not inferred from convergence. Inequality (5.2)
  is the separate step that establishes exact cancellation on an interval.
* The remainder includes arbitrary mixed-frequency atoms and new neurons.
  Its full cost is retained as t^r; it is not discarded in a differential
  argument at zero amplitude.
* The projection in section 4 is not the spectrum-preserving cleanup refuted
  in `PROJECTION_OBSTRUCTION.md`. It estimates amplitudes of a near-endpoint
  competitor; (5.1) compares the original competitor's own cost.
* No tensor-norm concavity, global symmetry of optimizers, fixed linear dual
  certificate below the endpoint, or unproved phase-frame rigidity is used.

## 8. What the paper can now say

For each fixed odd prime p and degree nu>=3, there exists a nonempty interval
of penalty exponents immediately below q=nu on which the dense Fourier
construction is globally normalized-margin optimal without any junk-cancellation
assumption. For q<nu in this interval, every optimizer has zero junk and exactly
nu neurons at each of the (p-1)/2 nonzero real Fourier frequencies.

The weak-regularization global-minimizer limit can therefore be applied
unrestrictedly in this interval at widths containing the dense construction.
This is a statement about global minimizers; it does not assert that training
finds them, or that finite-regularization models have exactly the limiting support.
