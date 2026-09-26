# An analytic attack on a hypothetical global optimizer

Status: the unrestricted ceiling is still open. This note contains analytic proofs of
necessary optimizer conditions and a finite-replacement criterion. It does not establish
that the required replacement exists. No numerical search, fitted constant, or numerical
certificate is used in this note.

Follow-up: `P3_ANALYTIC.md` subsequently closes the unrestricted optimizer-counterexample
problem at p=3, nu=3, for every 2<=q<=3 with a direct analytic inequality. It also closes
the arbitrary-width pure one-frequency case at every odd prime. The limitations recorded
below still apply to the general-prime unrestricted multifeature problem.

The central distinction is between a global replacement and a local perturbation. The
former is the proposed route. First-order stationarity, even with optimizer-dependent
row weights, is not a global certificate.

## 1. Set up an attained counterexample

Let nu=n+1, 2<=q<nu, and r=q/nu. Center each readout and balance the slot norms of every
nonzero neuron. Write the centered logit tensor as

    Z = sum_{i=1}^N a_i T_i,
    T_i = u_i0 tensor ... tensor u_in,
    a_i>0, ||u_ij||_2=1, sum_c u_i0(c)=0.

The least penalty for this representation is nu^(q/2) C, where C=sum_i a_i^r.
For a row/wrong-class pair alpha let ell_alpha(Z) be its gap. All constraints are
ell_alpha(Z)>=1. Write g_alpha=ell_alpha(Z), and I={alpha:g_alpha=1}.

ARGUMENT.md section 4 proves attainment at some N<=p^n(p-1). Thus, if the ansatz ceiling
fails, take an actual global minimum-cost margin-one representation with

    C < C_A,
    C_A = nu^(1-r) 2^(q/2-r) p^(q/2) C_r,
    C_r = min_{v>=0, Av>=1} sum_k v_k^r.

Its minimum gap is exactly one: otherwise scaling every a_i down improves cost.
No purity, DC-free input, or cancellation hypothesis is imposed.

## 2. A finite-replacement criterion that retains global optimality

Replace the existing atoms by arbitrary unit real atoms S_i, give them amplitudes b_i>=0,
and optionally add new atoms R_j with amplitudes c_j>=0. Suppose

    Z' = sum_i b_i S_i + sum_j c_j R_j

obeys EVERY margin-one constraint. Define

    E = r sum_i a_i^(r-1)(b_i-a_i) + sum_j c_j^r.                 (2.1)

Then:

* E<0 contradicts global optimality.
* E=0 also contradicts it if any b_i differs from a_i.

Proof. Strict concavity of x^r on [0,infinity), with its tangent taken at a_i>0, gives

    b_i^r <= a_i^r + r a_i^(r-1)(b_i-a_i),

with equality exactly when b_i=a_i. Sum and include the exact costs of the new atoms.
The resulting feasible representation costs at most C+E, strictly less if an old amplitude
changes and E=0. This proves both claims.

The replacement may change the clean spectrum, all junk coefficients, and the width.
There is no requirement that it preserve clean amplitudes or individual feature supports.
New atoms are charged their actual c_j^r cost; assigning them a finite tangent weight at
zero would be an error.

In particular, for the original N slots, a global optimizer must also globally minimize
the weighted linear-amplitude problem

    min sum_i a_i^(r-1) b_i
    subject to ell_alpha(sum_i b_i S_i)>=1 for every alpha,
               b_i>=0, S_i unit real rank-one atoms.             (2.2)

The minimum is C, attained at the original representation. Any other minimizer must have
b_i=a_i for each labelled slot. The slots are still limited to N: freely duplicating the
cheapest slot turns this into a different problem and loses the necessary condition.

Equation (2.1) is a sufficient test for a replacement, not an assertion that such a
replacement can always be found. A cheaper replacement can also exist with E>0 because
the tangent majorant need not be tight at its new amplitudes.

## 3. Strict positive multipliers in the amplitude problem

Fix the atoms T_i and let B have rows indexed by I, with

    B_alpha,i = ell_alpha(T_i).

Then Ba=1. Every nonzero d with Bd>=0 satisfies

    sum_i r a_i^(r-1) d_i > 0.                                 (3.1)

Proof. For small t>0, a+td stays positive and satisfies all active constraints; inactive
constraints stay slack. A negative first derivative decreases cost. A zero first derivative
also decreases cost, since the second derivative is

    r(r-1) sum_i a_i^(r-2) d_i^2 < 0.

Consequently B has full column rank. There are multipliers lambda_alpha>0 for EVERY
alpha in I such that

    B^T lambda = r a^(r-1),   sum_alpha lambda_alpha = r C.      (3.2)

For completeness, this is stronger than just applying a multiplier theorem. On the compact
set {d: Bd>=0, ||d||=1}, (3.1) has a strictly positive minimum. Thus, for sufficiently small
epsilon>0, w-epsilon B^T 1 still belongs to the dual cone of {d:Bd>=0}, where
w=r a^(r-1). The finite-dimensional alternative for linear inequalities represents it as
B^T lambda_0 with lambda_0>=0. Set lambda=lambda_0+epsilon*1. Dot with a to obtain the
sum identity. The compact set is nonempty, since Ba=1.

After normalizing mu=lambda/(rC), this says

    E_mu[ell_alpha(a_i T_i)] = a_i^r/C.                         (3.3)

Every neuron earns exactly its cost share against this strictly positive weighting of all
active constraints. This holds at any local amplitude minimum, so it cannot by itself
distinguish a global optimum.

### Why these positive multipliers cannot be used for arbitrary weight perturbations

They solve only the fixed-atom problem. They need not satisfy stationarity in the atom
directions. This distinction is essential when trying to reuse LOCAL_THEOREM.md.

An analytic example: minimize a^r for a>0 and |t|<1/2, subject to

    a(1-t^2)>=1,    a(1+t)>=1.

The unique global minimizer is (a,t)=(1,0). For t>0 the first constraint forces a>1; for
t<0 the second does. In the fixed-t amplitude problem both active multipliers can be
strictly positive, with lambda_1+lambda_2=r. In the full problem, t-stationarity forces
lambda_2=0. Thus even GLOBAL optimality does not generally upgrade amplitude strict
complementarity to full-parameter strict complementarity.

This is not a counterexample to the modular-addition conjecture. It rules out an unjustified
step in a proposed optimizer proof.

## 4. Full-weight multipliers exist without a hidden qualification assumption

Return to raw weights theta and P_q=sum_i ||theta_i||^q. At a margin-one local minimum
there are lambda_alpha>=0, supported on I, such that

    grad P_q = sum_alpha lambda_alpha grad g_alpha,
    sum_alpha lambda_alpha = (q/nu) P_q.                       (4.1)

Existence can be proved directly. The common scaling direction theta strictly increases
every active gap: grad g_alpha dot theta=nu. If the nonnegative gradient representation
failed, the alternative for linear inequalities would give a direction decreasing P_q
and not decreasing any active gap to first order. Add a sufficiently small positive
multiple of theta to make every active derivative strictly positive while retaining the
cost decrease. A small step is then feasible and improves cost, a contradiction.
Dotting stationarity with theta gives the displayed sum identity.

With mu=lambda/((q/nu)P_q), per-neuron homogeneity gives

    E_mu[g_alpha,i] = ||theta_i||^q/P_q.                        (4.2)

Separate homogeneity in each slot also gives equal slot norms within each nonzero neuron.
For an input slot j and value x_j=t, differentiating by simultaneous scaling of that
coordinate in every neuron gives the further identity

    sum_{alpha:x_j=t} mu_alpha
      = (nu/P_q) sum_i ||theta_i||^(q-2) |E_ij(t)|^2.           (4.3)

Complementarity is used here: g_alpha=1 wherever lambda_alpha>0. Each input-value
marginal is positive, since otherwise all E_ij(t) vanish and those rows cannot have
positive margin.

Equations (4.1)-(4.3) are genuine optimizer restrictions. They do not imply translation
invariance of mu. Averaging mu over translations generally destroys its stationarity at
the ORIGINAL theta; it averages stationarity equations at different translated networks.
Nor is full-weight mu necessarily strictly positive on I, as section 3's example shows.

## 5. An exact structural consequence of a uniform active offset

Call a centered tensor pure here if its Fourier support consists only of tuples

    (a_1,...,a_n,c),  |a_1|=...=|a_n|=|c|=k!=0.

This definition concerns the tensor, and does not require a particular neuron decomposition.

**Lemma.** If such a tensor has a gap at one nonzero offset D that is constant over all input
rows, then every same-frequency junk coefficient vanishes.

Proof. Use Fourier characters e(t)=exp(2*pi*i*t/p), with class index y+D and y=sum_j x_j.
A component (a_1,...,a_n,c) contributes to the gap at row mode

    u=(a_1+c,...,a_n+c)

with multiplier 1-e(cD). In a pure tensor each u_j is either 0 or 2c. For a nonclean
component some u_j is nonzero and determines c=u_j/2 uniquely. It then determines every
a_j=u_j-c. Hence no other component can cancel this coefficient at that row mode.
Because p is prime and c,D!=0, 1-e(cD)!=0. Constancy of the gap kills every such coefficient.
These are exactly the junk coefficients. QED.

Thus for pure tensors, a translation-invariant complementary multiplier would suffice:
some nonzero D has positive multiplier on EVERY row, so every corresponding gap equals
one, and the lemma gives (H). Applying the existing feature-cost theorem would finish the
value comparison for that optimizer. This uses complementarity structurally; it does not
maximize a linear weighted margin over all networks and so is not the false linear-duality
certificate discussed in P3B.md.

Neither purity nor the existence of that invariant multiplier has been proved for an
unknown unrestricted global optimizer.

### The impure extension of the lemma is false, analytically

Here is a feasible tensor showing why the missing qualification matters. Take n=2, any
odd prime p, D_0=1, k=1, and 0<epsilon<1/2. Define f on Z_p by

    f(0)=0, f(1)=-1, f(D)=-2 otherwise,
    b(D)=cos(2*pi*(D-1/2)/p),
    h(x)=cos(4*pi*x_1/p),
    Z(x,c)=f(c-y)+epsilon*h(x)*b(c-y),  y=x_1+x_2.

The 1/2 in b is an ordinary real phase shift; b is p-periodic, and b(0)=b(1).
Subtract the row mean from Z to center the readout. Its gaps are unchanged.

Every gap at offset 1 is exactly one. Every other nonzero-offset gap is at least
2-2epsilon>1. Thus the entire active set is a uniform offset orbit. Nevertheless the
product h(x)*b(c-y) has a nonzero Fourier component (1,-1,1), which is same-frequency
junk. It also has the compensating component (-3,-1,1), plus their conjugates. The latter
is cross-frequency (or has a DC input at p=3). The two readout frequencies contributing
to a given nonzero row mode cancel in the offset-1 gap.

There is no claim that this tensor is a global cost minimizer. It proves that feasibility
and a complete active offset orbit alone do NOT remove junk in the unrestricted problem.
Any optimizer argument using that orbit must also exploit representation optimality.

## 6. New atoms cannot supply a first-order escape for free

Suppose the old amplitudes change by O(t), remaining near their positive baseline, and
at least one new atom has amplitude c*t+o(t), c>0. Changes in the old cost are O(t), but
the new atom costs c^r*t^r+o(t^r). Since r<1, t^r/t tends to infinity. Such a small insertion
cannot lower the total cost. In raw weights this is the familiar comparison between
degree-q penalty and degree-nu output, but the amplitude statement makes the singularity
explicit.

This does not prohibit every higher-order perturbation and is not a proof that an arbitrary
stationary point is locally optimal. It rules out this specific infinitesimal insertion
as the desired improvement. A global replacement can instead remove a finite amount of
existing cost to pay for its new atoms, as section 2 explicitly allows.

## 7. A genuinely global restriction: deleting and rescaling a subnetwork

For a nonempty subset S of neurons let C_S=sum_{i in S} a_i^r, and let m_R be the true
minimum gap of the remaining subnetwork. If m_R>0, global optimality implies

    C-C_S >= C*m_R^r.                                         (7.1)

Indeed, dividing all remaining amplitudes by m_R gives a margin-one network of cost
(C-C_S)/m_R^r. This is a finite comparison, not a necessary condition obtained by
differentiating. In particular no nonempty deletion can leave margin at least one.

There is an explicit consequence that applies to every nonzero atom of a global optimizer:

    a_i >= (sqrt(2)*C)^(-1/(1-r)).                             (7.2)

To prove it, a unit real rank-one atom satisfies |ell_alpha(T_i)|<=sqrt(2): its input
coordinates have magnitude at most one, and |u_i0(y)-u_i0(c)|<=sqrt(2). Put L=sqrt(2).
Feasibility implies sum_i a_i>=1/L, hence C>=L^(-r). If L*a_i<1, deleting that atom leaves
margin at least 1-L*a_i. Apply (7.1) and use (1-t)^r>=1-t for 0<=t<=1 to obtain

    a_i^r <= C*[1-(1-L*a_i)^r] <= C*L*a_i.

This gives (7.2). If L*a_i>=1, the same conclusion follows from C>=L^(-r).

For a hypothetical counterexample C<C_A, (7.2) gives the uniform lower bound
a_i>=(sqrt(2)*C_A)^(-1/(1-r)). Thus a global counterexample cannot contain arbitrarily
small nonzero neurons at fixed p,nu,q. It also gives the width bound

    N <= C^(1/(1-r))*2^(r/(2*(1-r))),

which can be combined with the earlier linear-independence bound; it is not claimed to
improve that bound in the paper's regimes. Its useful feature is the positive lower
amplitude bound. Local minima alone need not obey the finite-deletion comparison.

## 8. What the optimizer attack has and has not established

The route now has a precise finite replacement test, a global deletion bound excluding
vanishing nonzero neurons, automatic full-weight multiplier existence, positive amplitude
multipliers, and an exact link from an active offset orbit to cancellation in pure tensors.
The counterexamples in sections 3 and 5 identify two invalid shortcuts without relying on
numerical experiments.

The unresolved step is addition-specific: use GLOBAL minimality, together with the full
row constraints, to construct a replacement meeting (2.1), or otherwise prove C>=C_A.
It must handle mixed-frequency components that can conceal junk on active rows, and it
must not silently assume invariant multipliers or infinitesimal access to new neurons.

No contradiction to an optimizer with C<C_A has yet been derived. In particular, this
note does not remove (H), does not establish the all-prime global ceiling, and does not
turn the earlier local theorem into a global one.
