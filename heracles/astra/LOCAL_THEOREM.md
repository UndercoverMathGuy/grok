# Unrestricted local optimality, and bad minima that survive weak weight decay

Status: a local theorem, NOT the missing global ceiling (T). The proof allows every raw-weight
perturbation, including DC, mixed frequencies, and additional neurons. It covers n=2,q=2.
Its only computer-assisted ingredient is one exact, prime-independent 18-variable quadratic
form certificate over Q(sqrt(3)); there is no numerical optimizer in that certificate.

## 1. Relation to the earlier work

`ELEGANT.md` already proves the GLOBAL ceiling under (H), at all depths and widths.
`GLOBAL_OPT.md` proves/estimates single-frequency pure-network statements; its
`gopt_local.py` is a numerical escape search restricted to pure single-frequency populations.
`GLOBAL_OPT4.md` computes a different second-order programme, for representation-cost relief
as a function of junk. `P3C.md` establishes local optimality of its population criterion.
`ARCH.md` has the stronger absolute-chain matrix route, with admissibility still open; even
its full-torus conjecture would not by itself close the finite-grid multifeature problem.
The theorem below concerns local maxima of the ACTUAL unrestricted full-table margin.

The previous astra pass's small-prime values are consistent with older numerical Gamma
values, rather than new numerical discoveries. The change there was complete algebraic
certification across supports. This pass uses those certificates but attacks local escape
in the full architecture, without assuming (H).

## 2. The theorem and its hypothesis

Let p be an odd prime, h=(p-1)/2, A_Dk=1-cos(2pi Dk/p). Fix a support S of size K and
positive shares b_k summing to one. Put

    g = min_D sum_{k in S} A_Dk b_k^(3/2).

Suppose there are K distinct offsets T subset {1,...,h}, all attaining this minimum, and
strictly positive numbers omega_D, D in T, such that

    sum_D omega_D = 1,
    sum_{D in T} omega_D A_Dk = g / sqrt(b_k),   k in S.       (2.1)

Construct the usual three-neuron blocks, with weights

    E1_kl(x) = sqrt(b_k) cos(2pi kx/p + pi*l/3 + alpha_k),
    E2_kl(x) = sqrt(b_k) cos(2pi kx/p + pi*l/3 + beta_k),
    W_kl(c)  = sqrt(b_k) cos(2pi kc/p + 2pi*l/3 + alpha_k+beta_k),
    l=0,1,2.

Let M be this 2K-dimensional phase family, with neuron ordering fixed. All its members
have the same logits, squared norm P0=9p/2, and margin m0=3g/4.

**Theorem.** For every fixed width N>=3K, pad these networks with zero neurons. They are
strict local maxima of normalized margin m/P_2^(3/2), modulo common positive scaling
and the phase family M. Equivalently, in a sufficiently small neighborhood of M, a network
with P_2<=P0 and every row/class gap >=m0 must itself belong to M.

No global optimality of b is assumed. Condition (2.1) is a positive stationary multiplier
condition; it can hold at suboptimal ansatz solutions too.

For an ansatz vertex v with Av>=1 and C=sum v^(2/3), the shares are b=v^(2/3)/C and
(2.1) becomes

    (A_TS)^T omega = v_S^(-1/3)/C.                            (2.2)

The six certified optima at p=3,5,7,11,13,17 satisfy this with strict positivity, certified
by exact active-row detection and outward interval arithmetic in `local_duals.py`.
In fact, the following argument shows that EVERY globally optimal ansatz vertex has strict
multipliers. Thus the local theorem covers the global ansatz optimum for every odd prime,
not only the six computationally certified examples.

### 2.1 Prime uncertainty makes the ansatz vertices nondegenerate

For any nonzero coefficients x on a support S of size K, the function

    f(D)=sum_{k in S} x_k(1-cos(2pi kD/p))

has at most 2K+1 Fourier frequencies. If a K-by-K submatrix A_TS were singular, a nonzero
x would make f vanish at 0 and all 2K residues +/-T. The uncertainty theorem stated in
section 4 forbids 2K+1 zeros. Thus EVERY square submatrix A_TS is nonsingular.

For a feasible vertex v with positive support S of size K, consider f(D)-1 instead.
It is not identically zero, since its value at 0 is -1, and still has at most 2K+1
Fourier frequencies. It therefore vanishes at at most 2K residues. Hence v has at most
K active offset pairs. A vertex requires at least K independent active offset constraints
on its positive support, so it has EXACTLY K. Every vertex of the ansatz polyhedron is
therefore nondegenerate, including its zero-coordinate constraints.

Now suppose v globally minimizes F_r(v)=sum v_k^r, for 0<r<1. Let T be its K active
offsets. For each j in T, the direction d_j=A_TS^(-1)e_j on S, zero elsewhere, is
one-sided feasible for sufficiently small positive steps: its positive coordinates remain
positive, inactive constraints stay slack, and only active constraint j is relaxed.
The directional derivative of F_r must be STRICTLY positive. A negative derivative would
improve cost; a zero derivative would also improve it, because

    d_j^T Hess(F_r) d_j = r(r-1) sum_k v_k^(r-2) (d_j)_k^2 <0.

Consequently all coordinates of A_TS^(-T)(r*v_S^(r-1)) are strictly positive.
For r=2/3, division by r*C gives precisely omega in (2.2). Finally

    sum_D omega_D = omega^T A_TS v_S
                  = sum_k v_k^(2/3)/C = 1.

This proves the multiplier hypothesis for every global ansatz optimizer. This argument
also strengthens ARGUMENT.md section 2: at prime p no degenerate ansatz vertices need
special handling, and all of its square active-set systems are nonsingular.

## 3. Positive row multipliers identify the critical directions

Define F(theta) as the weighted average of the gaps: each D in T and -D has weight
omega_D/2, and inputs are uniform on all p^2 rows. Then F<=margin is NOT claimed;
the true inequality is margin<=F. In particular, any network with all gaps >=m0 has
F>=m0. At a member theta0 of M, F(theta0)=m0.

Fourier orthogonality and (2.1) give, for the full raw-weight gradient,

    grad F(theta0) = mu theta0,       mu=g/(2p)>0.

One can also check the constant from homogeneity:
mu P0=3m0. Define L=F-(mu/2)P_2. Then grad L(theta0)=0.

Consider a limit v of unit displacement directions of networks with P_2<=P0 and all gaps
>=m0 tending to theta0. Every active gap derivative is nonnegative, and theta0 dot v<=0.
Taking the strictly positive weighted sum gives

    0 <= dF[v] = mu theta0 dot v <=0.

Consequently every gap derivative at offsets +/-T vanishes, as does the norm derivative.
These are the critical directions. The next step is the important unrestricted reduction:
those selected gap equalities already force the ENTIRE centered logit derivative to vanish.

## 4. Sparse Fourier interpolation of the logit derivative

Write a tensor Fourier frequency as (a,b,c), with c the readout frequency. At first order
about the block network, every neuron term has two unperturbed slots. Thus every nonzero
Fourier coefficient of the logit derivative has at least two of

    |a|_p, |b|_p, |c|_p

equal to the same element of S; |.|_p identifies the pair of nonzero residues +/-k.
The third frequency is completely unrestricted. Additional zero neurons have zero first
variation because their output is cubic. This support statement permits all impurities.

Restrict the readout to the correct label plus D. For a fixed input Fourier mode (u,v),
we have a=u-c and b=v-c. Its gap derivative, as a function of D, is

    f_uv(D) = sum_{c in R_uv} z_c (1-exp(2pi i cD/p)),          (4.1)

where c=0 contributes nothing and is omitted. We claim |R_uv|<=2K.

* If u=v=0, then c belongs to +/-S.
* If u=0,v!=0 (or symmetrically v=0), then c belongs to +/-S. The other possible
  equal-magnitude pairs force c=v/2 and, when allowed, this is already in +/-S.
* If u=v!=0, then c belongs to u+/-S. Equalities involving the readout add only c=u/2,
  already in this set whenever allowed.
* If u,v are nonzero and unequal, the three possible equal-magnitude pairs force
  c=u/2, v/2, or (u+v)/2. There are at most three possibilities, so at most 2K when K>=2.
  When K=1, having both first possibilities would force v=-u, making the third c=0,
  which has been omitted. Thus there are at most two in this case as well.

All arithmetic here is in Z_p; division by two is valid. This proves the claim.

We use the prime-order uncertainty theorem: a nonzero function on Z_p with at most m+1
Fourier frequencies has at most m zeros. This is the support-sum theorem of
[Tao, *An uncertainty principle for cyclic groups of prime order*](https://arxiv.org/abs/math/0308286),
equivalently the nonsingularity of all prime Fourier minors.

The function (4.1) has at most |R_uv|+1<=2K+1 Fourier frequencies. It vanishes at D=0
identically and at all 2K offsets +/-T by section 3. The uncertainty theorem forces
f_uv identically zero. Independence of Fourier characters then forces every z_c=0.
This applies to every (u,v), so the centered logit derivative vanishes completely.
Readout-constant components remain irrelevant to gaps and will be penalized by the norm.

This is a first-order theorem at the ansatz, not a global sparse-support claim about an
arbitrary network. Applying it away from the ansatz would be unjustified.

## 5. One universal exact block calculation controls the Hessian

Use orthonormal real Fourier coordinates in each slot, with the basis (cos,-sin) at each
nonzero frequency. A neuron in block k has, in its own frequency plane, the vectors

    s_k (cos(phi_l), sin(phi_l)),
    s_k (cos(phi_l), sin(phi_l)),
    s_k (cos(2phi_l), sin(2phi_l)),
    s_k=sqrt(p b_k/2),   phi_l=pi*l/3,

up to the phase rotations in section 2. The clean trilinear form in these coordinates is

    f(a,b,c)=(a0 b0-a1 b1)c0+(a0 b1+a1 b0)c1.

For three unit-amplitude neurons, let J0 be the derivative of their full 2x2x2 tensor and
H0 the Hessian of sum_l f(a_l,b_l,c_l) minus the 18x18 identity. Its gradient is zero after
subtracting one half the squared norm. The following is an exact finite algebraic lemma:

    dim ker J0 = 10;
    -H0 is positive semidefinite on ker J0;
    its kernel there is exactly the two constant-tensor phase motions;
    v^T H0 v <= -||v||^2 for v in ker J0 orthogonal to those motions.   (5.1)

`local_block.py` proves these statements over Q(sqrt(3)), without tolerance comparisons.
It differentiates the eight tensor entries, constructs an exact nullspace basis N, and
eliminates the quadratic form -N^T H0 N by congruence. The eight positive pivots are

    4, 16/3, 1, 3, 6, 77/18, 2080/693, 117/160,

followed by two zero directions. It checks that the two phase tangents span this kernel.
For the last bound it separately certifies

    N^T(-H0 + Pi_phase - I)N >=0,

where Pi_phase is the exact orthogonal projection onto the phase tangents. Its positive
pivots are 2,8/3,1/2,3/2,11/3,83/33,392/249,18/49. The full reconstruction is short enough
to replay directly; `local_block_exact.json` records the certificate outputs.

Why the same calculation applies to every p, support, and share vector:

1. F is a sum of clean Fourier trilinear forms. For a neuron based at k, its Hessian in
   any other frequency direction is zero: a second derivative leaves one slot fixed at k,
   whereas a clean term requires all three frequencies to match. Therefore the Hessian of
   L outside the neuron's own frequency plane is exactly -mu I. This includes DC.
2. In its own frequency plane, stationarity says that the trilinear coefficient times s_k
   equals mu. Thus that block's Hessian of L is exactly mu H0, independent of b_k and p.
3. The projection of the full logit derivative onto U_k tensor U_k tensor U_k comes only
   from neurons in block k. A different block has two slots still at its different frequency.
   Section 4 therefore puts the within-frequency displacement of each block in ker J0.
4. Different neurons and frequency components are orthogonal in parameter norm. Summing
   (5.1), and the negative outside-plane quadratic forms, makes the full Hessian strictly
   negative on the critical space normal to the 2K phase motions.
5. An added zero neuron has zero output Hessian and contributes -mu I to the Hessian of L.
   Hence padding with any finite number of zero neurons preserves the conclusion.

This supplies a strict second-order bound in all directions that could pass the first-order
margin constraints. It is stronger than a numerical inability to escape the block.

## 6. From the Hessian statement to an actual local maximum

Here is the short argument avoiding any unproved constraint qualification. Suppose there
were feasible networks theta_j outside M approaching M, with P_2(theta_j)<=P0 and every
gap >=m0. Choose a nearest phi_j in M, set delta_j=theta_j-phi_j, and pass to a subsequence
so delta_j/||delta_j|| converges. Nearest-point orthogonality makes the limit normal to M.
Section 3 makes it critical, and sections 4-5 make its Hessian quadratic form strictly
negative. Phase rotations are orthogonal and preserve F and P_2, so these statements hold
uniformly along the compact phase family M.

But feasibility gives L(theta_j)>=m0-(mu/2)P0=L(phi_j). Since grad L(phi_j)=0, Taylor
expansion and the strictly negative limiting Hessian give the opposite inequality for
large j. Contradiction. The phase family is locally an embedded torus: the 2K displayed
tangent vectors are independent; discrete phase identifications do not affect the argument.
Normalizing any nearby nonzero weight array to P0 proves the normalized-margin statement.

## 7. An unconditional dense local maximum and a certified suboptimal example

Take S={1,...,h}, b_k=1/h, T={1,...,h}, omega_D=1/h. Finite Fourier orthogonality gives

    g_dense = p/(2 h^(3/2)),
    sum_D omega_D A_Dk = p/(2h) = g_dense/sqrt(b_k).

Thus every hypothesis holds. The dense ansatz is a strict unrestricted local maximum,
modulo phases and scale, for every odd prime p at n=2,q=2 and every width N>=3h.
For this dense corollary all offsets are active, so section 4 does not even require the
prime-order uncertainty theorem: all gap derivatives already vanish by section 3.

At p=11, the exact ansatz certificate gives

    Gamma_sparse = 0.5357259257995056002... on {1,3},
    Gamma_dense  = 11/(2*5^(3/2)) = 0.4919349550499537... .

The normalized-margin prefactor is the same. Hence the sparse construction has a margin
about 8.9018% larger at the same penalty, while the dense construction is a strict local
maximum even with all raw weights free. This does not rely on proving that the sparse
construction is globally optimal among unrestricted networks: an explicit better network
is enough to prove the dense local maximum is suboptimal.

## 8. These bad local minima persist as lambda tends to zero

Fix p=11 and any width N>=15. On the P_2=1 sphere, let M_d be the normalized dense phase
family, with margin g_d. Choose a sufficiently small closed tube around M_d. Section 6
and compactness give a strict gap on its boundary:

    max_boundary margin = g_b < g_d,

and M_d is the only maximizing set inside the tube. Shrink the tube so g_b>0 if necessary.
For a direction u and radial logit scale t, set

    J_lambda(t,u)=CE(t*z(u))+lambda*t^(2/3).

The bounds from ARGUMENT.md section 5 hold uniformly: on the boundary,

    CE >= (1/p^2) log(1+exp(-t*g_b)).

Radial minimization of this scalar lower bound has leading value
lambda*(log(1/lambda))^(2/3)*g_b^(-2/3). On M_d, an explicit radial choice has leading
value lambda*(log(1/lambda))^(2/3)*g_d^(-2/3), strictly smaller. Consequently a minimum
restricted to the closed tube lies in its interior for all sufficiently small lambda.
Existence follows from coercivity in the raw weights; the zero network, with CE=log p,
is not selected for small lambda. Such an interior restricted minimum is a local minimum
of the unrestricted regularized objective.

These directions approach M_d as lambda tends to zero: otherwise compactness would give
a strict margin gap on a closed subset a positive distance from M_d and the same comparison
would exclude it. Their leading objective constant is therefore g_d^(-2/3). The sparse
ansatz has margin g_s>g_d and gives a strictly lower objective for sufficiently small
lambda. Thus these local minima are not global. In fact, their objective divided by the
best objective on the sparse ansatz ray tends to

    (g_s/g_d)^(2/3) = 1.0584978673882723... .

This is an existence theorem for suboptimal local minima, not a claim that a particular
training run converges to them. It establishes mathematically that decreasing weight decay
does not remove all bad basins in this architecture. Wei's global-minimizer qualification
therefore has concrete force here.

### 8.1 An explicit stationary dense branch, locally minimizing at large scale

The dense example admits a stronger constructive statement. Let theta_d be any unnormalized
dense block from section 7, so all gaps equal m0 and P_2=P0. For R>0 put

    beta(R) = (p-1) exp(-R^3 m0) / [1+(p-1)exp(-R^3 m0)],
    lambda(R) = (mu/2) R beta(R),       mu=3m0/P0.

Then R*theta_d is an EXACT stationary point of CE+lambda(R)P_2. All wrong classes have the
same softmax probability, so grad CE=-beta(R)*grad F at that point, and

    grad F(R theta_d)=R^2 mu theta_d,
    grad(lambda(R)P_2)(R theta_d)=R^2 mu beta(R) theta_d.

To check that it is a local minimum, let J be the Jacobian of all gaps at theta_d and
H_L=Hess F(theta_d)-mu I. The softmax covariance formula gives the lower bound

    Hess(CE+lambda(R)P_2) / [beta(R) R]
      >= R^3 [1-beta(R)] J^T J / [p^2(p-1)] - H_L.            (8.1)

Indeed, on each row, with rho=beta/(p-1), the covariance is
rho*sum g_j g_j^T-rho^2*(sum g_j)(sum g_j)^T, which is at least
rho*(1-beta)*sum g_j g_j^T by Cauchy-Schwarz. Gap gradients at R theta_d scale as R^2;
the remaining loss-Hessian term is -beta(R) R Hess F(theta_d).

Section 5 says -H_L is positive definite on ker J normal to the phase family. Both J
and H_L annihilate the phase tangent: for H_L, differentiate grad L=0 along that family.
On the orthogonal complement of the phase tangent, a sufficiently large multiple of
J^T J minus H_L is therefore positive definite. This follows directly by splitting into
ker J and its orthogonal complement and taking a Schur complement (or compactness of the
unit sphere). Since R^3(1-beta(R)) tends to infinity, (8.1) is positive definite transverse
to the phase family for all sufficiently large R. The objective is exactly constant on
that family, so this proves a strict local minimum modulo phases.

Finally lambda(R) tends to zero and is strictly decreasing eventually, because

    d log(lambda(R))/dR = 1/R - 3m0 R^2[1-beta(R)] <0

for large R. Thus this explicit branch supplies a suboptimal local minimum for EVERY
sufficiently small lambda at p=11, not merely a sequence of lambda values. Its leading
objective constant is the dense one: log(1/lambda(R))~m0 R^3 and the CE term is negligible
relative to lambda(R)P0 R^2. The sparse ansatz comparison above proves strict suboptimality.

## 9. Verification and the remaining global task

* `local_block.py`: exact universal quadratic-form proof, with a strict normal-space gap.
* `local_exact.py`: independent exact full-weight p=3 calculation (27 variables, 18 gaps,
  critical dimension 13, eleven negative directions and two phase zero directions).
* `local_duals.py`: exact active constraints and certified positive multipliers for the six
  small-prime optimal ansätze. Minimal weights range from 1 at p=3 to >0.11347 at p=17.
* `local_multifeature.py`: independent floating-point full-weight Jacobian/Hessian checks
  at p=3,5,7,11,13,17. The critical dimension is 13K; nullity is exactly 2K; every other
  eigenvalue is negative. These are consistency checks, not the proof.

The sharp GLOBAL ceiling is still not proved. A better unrestricted network, if one exists,
cannot be reached by a sufficiently small perturbation of the certified ansätze. Ruling
out a distant better solution still requires the global inequality in ARGUMENT.md (4.1).
The dense counterexample shows why a local-maximizer classification alone cannot select K_*:
there are already provably suboptimal ansatz local maxima.
