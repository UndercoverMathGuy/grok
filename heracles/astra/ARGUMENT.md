# One argument toward unrestricted ansatz optimality

## Status

The unrestricted sharp ceiling is still OPEN. No numerical optimizer is a proof of it.
This file separates the exact statements established in this session from its missing lemma.
The goal is an attaining ansatz for the unrestricted network problem, followed by a valid
regularization-limit conclusion. All notation is defined below.

**Latest partial resolution of the unrestricted target:** `PHASE_FRAME_EXTENSION.md`
proves it for every odd prime p and every nu>=3 on nu-4/[3(p-1)]<=q<=nu,
and at p=5,nu=3 on the larger interval 5/2<=q<=3. For q<nu there, every optimizer
is junk-free with exactly nu neurons per frequency. Exact checks are in
`phase_frame_verification.json`. The same note gives an exact q=2 obstruction
to a matrix-only extension, explicitly not a counterexample network.
`EXPLICIT_ENDPOINT.md` retains the earlier explicit interval and its separate proof.
`ENDPOINT_NEIGHBORHOOD.md` retains the earlier nonquantitative proof.
`CONTINUATION.md` extends dense local optimality at nu=3 to every 2<=q<3 and
shows why a distant competing optimum can still obstruct global continuation.
The unrestricted sparse-feature regime and the full range 2<=q<nu remain open.

The subsequent analytic optimizer-counterexample attack is in `OPTIMIZER_ATTACK.md`.
It retains global optimality through finite replacement and deletion comparisons, and
does not claim that its necessary conditions imply the missing ceiling.

Follow-up: `P3_ANALYTIC.md` now proves the unrestricted ceiling analytically at p=3,
nu=3, for every 2<=q<=3, including equality rigidity for q<3. It also proves the exact
pure one-frequency theorem at every odd prime and arbitrary width for those exponents.
The general-prime unrestricted multifeature theorem remains open.

`GENERAL_DEPTH.md` extends these analytic results to every nu>=3 for nu-1<=q<=nu
(and a slightly larger explicit interval). Its p=3 theorem is unrestricted; its
all-prime theorem is for pure one-frequency networks. It also proves that the same
largest-junk-coefficient inequality is false at q=2 for every nu>=4, so the missing
low-q range needs joint information about the junk rather than that scalar alone.

## 1. Target and the three logically distinct conclusions

Fix an odd prime p, n operands, nu=n+1, and 2<=q<nu. Put r=q/nu and s=1/r. The real gated-chain
network has logits z_c(x)=sum_i W_ci product_j E_ij(x_j), full-table margin m(theta), and
penalty P_q(theta)=sum_i ||theta_i||_2^q. Write gamma_q=m/P_q^(1/r).
Let B=2(2p nu)^(-nu/2), and let Gamma_s(p) be the free-share programme in the parent notes.
The desired value theorem is

    sup_theta gamma_q(theta) = nu^(-(1-r)/r) B Gamma_(1/r)(p).       (T)

The supremum here ranges over every finite width, allowing DC, mixed frequencies, and junk.
The lower bound is already attained by the share-weighted nu-neuron blocks. The missing
statement is the unrestricted upper bound. Once (T) holds, every fixed width large enough
to contain an attaining block has this same optimum.

Value/attainment: there exists a maximal-margin ansatz with the support of a Gamma maximizer.
Rigidity: every maximal-margin network has a specified feature structure/count.
Limit selection: weakly regularized global minimizers converge in normalized margin to the
maximum; their accumulation points lie in the maximizing set at fixed width.
Only the first of these is needed to exhibit an optimal K-feature network. The second is
needed to force that count on every possible limit. Value convergence alone does not select
a particular member of a non-singleton maximizing set.

## 2. Exact finite formulation of the ansatz problem

Set h=(p-1)/2 and define the h-by-h real matrix

    A_Dk = 1-cos(2pi Dk/p),       D,k=1,...,h.

Every entry is positive. Define the polyhedron

    P = {x in R^h : x>=0 and Ax>=1}.

Here x_k is a feature amplitude normalized so its worst wrong-class offset has margin at
least one. Let

    C_r = min_{x in P} sum_k x_k^r.

Then exactly

    Gamma_(1/r)(p) = C_r^(-1/r).                               (2.1)

Proof: for shares b with sum b=1, let g=min_D sum A_Dk b_k^(1/r)>0. The vector
x_k=b_k^(1/r)/g belongs to P and sum x_k^r=g^(-r), so C_r<=Gamma^(-r).
Conversely, any x in P with c=sum x^r gives shares b_k=x_k^r/c and programme value at
least c^(-1/r). Taking a minimizing x proves the opposite inequality. A minimizer exists:
P is closed and nonempty, and every sublevel of sum x^r is bounded. At a minimizer min Ax=1,
since otherwise scaling x down improves the cost.

### 2.1 Every optimum is a vertex; candidates do not depend on r

For 0<r<1, sum x_k^r is strictly concave on the nonnegative orthant. If a minimizer x were
not extreme in P, write x=t y+(1-t)z for distinct feasible y,z and 0<t<1. Strict concavity
gives cost(x)>t cost(y)+(1-t)cost(z), contradicting minimality. Thus every minimizer is a
vertex. A polyhedron given by finitely many linear inequalities has finitely many vertices.
Call this finite set V. Therefore

    C_r = min_{v in V} F_v(r),       F_v(r)=sum_{k:v_k>0} v_k^r.   (2.2)

The amplitude ratios v_k are chosen from a finite list fixed by p; r changes which list
member wins and changes the shares b_k=v_k^r/F_v(r). On an interval of r with the same
winning vertex, its normalized-margin amplitude ratios stay fixed. This is stronger than
saying a separate nonlinear programme must be solved for each q.

A complete candidate enumeration is explicit. For every support S of size K and every set
T of K offset indices, invert the square matrix A_TS whenever it is nonsingular, set
v_S=A_TS^(-1)1 and v_outside=0, and retain v if it is nonnegative and Av>=1.
Every vertex occurs: the zero-coordinate constraints have rank h-K, so some K of its active
offset constraints are independent on the support. Degenerate candidates may repeat and
zero entries may shrink a nominal support; neither affects completeness.

This is an exact finite characterization, not a claim that enumeration scales cheaply to
all paper primes. It also shows why an equal-share shortlist cannot certify the free-share
optimum unless all discarded candidates are rigorously excluded.

### 2.2 Feature-count uniqueness outside finitely many exponent values

Distinct vertices can have identical functions F_v, for example by frequency symmetry.
If F_v and F_w are identical on an interval, their positive coordinate multisets coincide.
Indeed, after grouping equal coordinates, F_v-F_w is a finite linear combination of
exp(r log a) at distinct real exponents log a. These exponential functions are linearly
independent (differentiate at one point; their coefficient matrix is Vandermonde).
In particular identical functions have the same number of positive coordinates.

A nonzero linear combination of L exponentials with distinct real exponents has at most
L-1 distinct real zeros. Proof by induction: divide by the first exponential, differentiate,
and use Rolle's theorem. The derivative contains at most L-1 exponentials, and the induction
hypothesis bounds its zeros. If there were L zeros of the original function, Rolle would
force L-1 zeros of its derivative, a contradiction. This also rules out infinitely many zeros.

There are finitely many vertex pairs. Therefore, except for a finite set of r in (0,1),
all optimizing vertices have the same positive-coordinate multiset and hence the same K.
At the exceptional exponents one should use the set of optimal support sizes, or its minimum,
unless a further comparison resolves a tie. This does NOT assert that r=2/nu avoids that set.

Frequency dilation k -> fold(a k mod p), for a nonzero residue a, permutes both the
coordinates and the rows of A. Every nonempty support can therefore be mapped to a support
containing 1, with its amplitude multiset unchanged. Enumerating only these supports suffices
for the optimum and all optimal support sizes; it does not enumerate every labeled vertex.

### 2.3 Endpoint structure: a dense interval and a one-feature interval

Let C_Dk=cos(2pi Dk/p). Finite Fourier orthogonality gives

    C 1 = -1/2 1,        C^2 = (p/4)I-(1/2)J,
    A=J-C,              A C = -(p/4)I,
    A^(-1)=-(4/p)C,     A 1=(p/2)1.

Here J is the all-ones matrix. At r=1, summing Ax>=1 gives
sum x>=2h/p. Equality forces every offset inequality to be equality. Since A is invertible,
the unique minimizer is v_dense=(2/p)1. Because V is finite and every other vertex has a
strictly larger cost at r=1, continuity gives an interval r0<r<=1 on which v_dense remains
the unique optimizer. Thus there is generally a dense plateau BELOW q=nu; it is not valid
to assert that every q<nu necessarily yields a sparse optimal support.

As r tends to zero, F_v(r) tends to the number of positive coordinates of v. One-coordinate
vertices exist because A_Dk>0. The finite vertex list therefore implies that all optimizers
have K=1 for sufficiently small positive r. These statements concern r in (0,1); the
allowed network range q>=2 corresponds to r>=2/nu and may not reach the one-feature interval.

### 2.4 An explicit transition and certified small-prime optima

For p=5, put a=(5-sqrt(5))/4 and b=(5+sqrt(5))/4. Then A=[[a,b],[b,a]].
Its feasible polyhedron has exactly three vertices: (1/a,0), (0,1/a), and (2/5,2/5).
For example, on the first axis the stronger inequality is a*x>=1; the intersection of
the two offset equalities is the dense vertex. Thus

    C_r(5) = min{a^(-r), 2(2/5)^r},
    r_c = log(2) / log(5/(2a)).

For 0<r<r_c all optima have K=1, for r_c<r<1 all have K=2, and at r_c both counts
occur. No intermediate point of their edges minimizes, by strict concavity. This is an
actual count ambiguity at an isolated exponent, not just a possible technical pathology.
Here r_c is about 0.539; r=2/3 is in the dense region.

`vertices.py` implements complete enumeration over Q(2*cos(2*pi/p)), with exact rational
field operations for linear solves, singularity, and equality. Chebyshev recurrence builds
the matrix. Nonzero signs are checked by outward interval evaluation; unresolved signs
cause failure rather than a tolerance decision. Algebraically identical positive amplitude
multisets are grouped, and the winning class must be strictly separated from all other
classes by interval evaluation of sum v^(2/3). There is no floating-point shortlist.
The saved `vertices_p*.json` files record the candidate count, algebraic winning amplitudes,
objective intervals, and separation status. These certify the ANSATZ programme only.

| p | optimal K at r=2/3 | one optimal support | Gamma, displayed approximately |
|---|---|---|---|
| 3 | 1 | {1} | 1.5 |
| 5 | 2 | {1,2} | 0.8838834764831844 |
| 7 | 3 | {1,2,3} | 0.6735753140545634 |
| 11 | 2 | {1,3} | 0.5357259257995056 |
| 13 | 3 | {1,2,5} | 0.4977720672100500 |
| 17 | 3 | {1,3,5} | 0.4483405600853964 |

## 3. The whole-network inequality that remains

Use the Fourier convention of the parent Feature Cost proof. Let G_k be the clean Fourier
coefficients, H=sum_k |G_k|^r, and Q=(p/(2nu))^(q/2)P_q. For positive margin, H>0 and
F=min_D sum_k |G_k|A_Dk>0. Define

    rho = m/(2p^(-nu)F),
    tau = F/(Gamma_(1/r)(p) H^(1/r)),
    eta = nu^(1-r)H/Q.

The unrestricted level-set bound gives 0<rho<=1. The definition of Gamma gives 0<tau<=1.
Algebra gives the exact identity

    gamma_q / [nu^(-(1-r)/r) B Gamma_(1/r)] = rho tau eta^(1/r).   (3.1)

Under (H), the Feature Cost and neuron-budget bounds give eta<=1. Without (H), proving
rho tau eta^(1/r)<=1 is the one missing global statement. Same-frequency and cross-frequency
junk are both still present in rho. No purity assumption is hidden in (3.1).

The cheap unrestricted budget H<=Q gives eta<=nu^(1-r). Consequently a spectrum with

    tau <= nu^(-(1-r)/r)

already satisfies (T). More sharply, a support S is harmless if its restricted programme
Gamma_(1/r)(p;S) is at most nu^(-(1-r)/r) times the full programme. This is a rigorous
exclusion criterion, but it is not enough to handle the competitive spectra.

The crucial unresolved issue is whether competitive spectra force a sufficient relationship
between rho and eta. Bounding either one in isolation loses the desired factor. Replacing
this question with an unproved purification or symmetry claim would simply assume the result.

`PROJECTION_OBSTRUCTION.md` strengthens the obstruction: it gives an explicit analytic
positive-margin family for which cleaning the SAME spectrum strictly lowers normalized
margin, even with minimum-cost re-representation. It also gives a different clean spectrum
that beats this family. Thus allowing the spectrum to change is necessary for a universal
replacement argument; positive margin alone does not rescue spectrum-preserving projection.

## 4. A finite-width reduction for the UNRESTRICTED problem

This reduction allows DC and mixed frequencies. It does not assert that optimal width is
nu*K or that a sharp Fourier budget holds. It is distinct from the unsuccessful attempt
to obtain nu atoms from Caratheodory applied to the one-frequency junk equations.

Let d=p^n(p-1), the dimension of the vector space of logit tensors whose sum over the
class index is zero for every input row. Every full-table network can be replaced by a
network with at most d nonzero neurons, identical class differences, and no larger P_q.
For q<nu, a global minimum-cost representation has linearly independent centered rank-one
neuron tensors and therefore at most d nonzero neurons.

Proof, including the cost reduction:

1. Replace every readout w_i by w_i-mean(w_i)*1. This only subtracts a row-dependent common
   logit from all classes. It preserves every margin and CE loss, and weakly decreases cost.
   Delete neurons with a zero slot. Their output is zero and keeping them cannot help cost.
2. Write a remaining neuron as a_i*u_i0 tensor u_i1 tensor ... tensor u_in, where every
   u_ij has Euclidean norm one, the readout u_i0 is centered, and a_i is the positive product
   of the original slot norms. Signs stay in the u_ij. Rescaling the slots to common norm
   a_i^(1/nu) leaves its tensor fixed. AM-GM shows the least neuron cost at this tensor is

       nu^(q/2) a_i^r,       r=q/nu.

3. Fix these unit tensor directions T_i in R^d and the centered total tensor Z. The feasible
   coefficient polyhedron is {a>=0 : sum a_i T_i=Z}. Minimizing sum a_i^r on this polyhedron
   has a solution: it is nonempty and its cost sublevels are compact. For r<1 strict concavity
   forces every solution to be an extreme point, by exactly the argument in section 2.1.
   The positive columns at an extreme point are linearly independent: otherwise a nonzero
   linear dependence gives two sufficiently small opposite perturbations of its positive
   coordinates with the same total tensor. Thus at most d coordinates remain.
4. Balance the remaining tensors as in step 2 to obtain actual real network weights. Their
   centered logits are Z, so all class differences are unchanged. The cost is no larger.

For a global margin-one cost minimizer, the same argument proves the independence claim
directly: dependence would give strict cost improvement. A minimizer exists because width
d suffices, feasibility is closed, and bounded-cost sets of raw weights at this fixed width
are compact. Feasibility follows, for example, from the ansatz or a row-by-row construction.
Consequently the unrestricted maximal normalized margin is attained at width at most d;
it is not merely a supremum escaping to infinite width.

For r=1 the replacement still exists by linear programming (the nonnegative coefficient
polyhedron is pointed, and a finite attained minimum has an optimal vertex), but not every
optimal representation must have independent columns.

### 4.1 What this changes in the proof task

For q=2, the complete unrestricted inequality is equivalent to the following single
finite polynomial feasibility question. At width d, normalize sum of all squared weights
to 1 and let gamma_A be the known ansatz value. Is there a real weight array such that

    sum_i (W_yi-W_ci) product_j E_ji[x_j] > gamma_A
    for EVERY x and c != y=sum x_j mod p ?

If such weights exist, they refute (T); if no such weights exist, width compression proves
(T) for every finite width. No Fourier purity or same-frequency assumption enters this
equivalence. This is a finite real algebraic problem when gamma_A is algebraic (in
particular for rational q and a certified algebraic ansatz vertex). The number of variables
is large; this observation is not itself an efficient certificate or a proof of infeasibility.

For the margin-one cost minimization, one can additionally bound the number of nonzero
neurons by the number of active row/class margin constraints. Indeed, if their active
constraint columns are dependent, a small two-sided coefficient perturbation preserves
active equalities and maintains all strict inequalities. Strict concavity then contradicts
optimality. This is an optimizer-specific necessary condition, not the desired nu*K bound.

### 4.2 The missing inequality in tensor form, with all constants removed from the network

Let A_r(Z) be the minimum of sum_i a_i^r over real decompositions

    Z = sum_i a_i u_i0 tensor ... tensor u_in,
    a_i>=0, ||u_ij||_2=1, sum_c u_i0(c)=0.

Section 4 makes this a minimum over at most d terms. Let C be the polyhedron of centered
logit tensors satisfying Z_y(x)-Z_c(x)>=1 for every input and incorrect class. Balancing gives

    min_{theta: centered logits Z} P_q(theta) = nu^(q/2) A_r(Z).

Thus the precise missing theorem, equivalent to (T), is

    min_{Z in C} A_r(Z) >= nu^(1-r) 2^(q/2-r) p^(q/2) C_r.     (4.1)

The opposite inequality follows from the explicit Fourier tensor

    Z_v(x,c) = sum_k v_k cos(2pi k(sum x_j-c)/p),

where v is a minimizing vertex in section 2. It is centered and its gaps are (Av)_D.
A nu-term common-phase block for its kth summand has each unit-factor coefficient

    a = [2^(nu/2-1) p^(nu/2)/nu] v_k,

so summing nu*a^r over k gives exactly the right-hand side of (4.1).
For n=2,q=2 the target becomes particularly simple:

    min_{Z in C} A_(2/3)(Z) >= 6^(1/3) p C_(2/3).

This formulation keeps every possible junk tensor and every possible neuron mixture. It
shows exactly where the ansatz-only proof stops: the missing comparison is between a
tensor representation cost on a margin polyhedron and the finite Fourier vertex problem.

One tempting extension of section 2 is provably unavailable. The polyhedron C has just one
vertex, Z_0(x,c)=1_{c=y}-1/p: on each row the p-1 excess gaps are independent nonnegative
coordinates. Yet for p=11,r=2/3 the certified Fourier vertex {1,3} has smaller representation
cost than Z_0. Indeed Z_0 corresponds to dense v_k=2/11, and the Feature Cost lower bound
is sharp for this junk-free tensor. The certificate separates its cost from the winner.
Consequently A_r is not concave on C, and minimizing it cannot be reduced to vertices of C
by repeating the ansatz argument. Concavity holds in the coefficients of a FIXED atom
dictionary; passing to the best dictionary destroys that property. This is an explicit
obstruction, not a conjecture about purification.
To justify the nonconcavity inference on this unbounded polyhedron: if a nonnegative concave
function decreased from Z_0 to Z_0+V, then its secant slope along the feasible ray Z_0+tV
would force negative values for sufficiently large t. Representation cost is nonnegative.

## 5. Closing the regularization-limit link, conditional only on the margin value

This section supplies a direct multiclass proof so that no ambiguity about the chosen
penalty, homogeneity, or appendix convention remains. It is the specialization of the
weak-regularization principle in [Wei et al., Theorem 4.1](https://proceedings.neurips.cc/paper/2019/file/8744cf92c88433f8cb04a02e6db69a0d-Paper.pdf).
That theorem concerns global minimizers, not a guarantee that a training algorithm finds them.

Fix a width with positive achievable margin. Let R=P_q^(1/q), a norm for the stated q>=2,
and let g_* = max_{R(u)=1} m(u)>0. Compactness ensures attainment. Define average multiclass
CE on M=p^n rows and the regularized objective

    J_lambda(theta)=CE(theta)+lambda P_q(theta),   lambda>0.

It is continuous and coercive, so global minimizers exist. Writing theta=R*u and t=R^nu
gives logits t*z(u), penalty lambda*t^r, and normalized margin g(u)=m(u).
For g(u)>0 the elementary bounds are

    (1/M) log(1+exp(-t*g(u))) <= CE(R*u) <= (p-1) exp(-t*g(u)).   (5.1)

The lower bound keeps just one incorrect class on a row attaining the minimum margin.
The upper bound uses log(1+a)<=a and all gaps at least g(u).
The lower bound also holds when g(u)<=0; then CE>=log(2)/M.

Put L=log(1/lambda). For a maximizing direction u_*, choose

    t=(L+2 log L)/g_*.

For lambda sufficiently small, (5.1) gives

    J_lambda <= lambda L^r (g_*^(-r)+o(1)).                    (5.2)

In particular the minimum tends to zero. For ANY sequence satisfying (5.2), CE tends to
zero, the attained unnormalized margin t*g tends to infinity, and g>0 eventually.
Since log(1+e^(-v))>=e^(-v)/2 for v>=0, the lower bound in (5.1) implies

    t*g >= L-r log L-O(1).

The nonnegative penalty term and (5.2) imply

    t <= (g_*^(-1)+o(1)) L.

Dividing shows liminf g>=g_*. Since g<=g_* by definition, g -> g_*.
Also t*g<=t*g_* and the preceding lower bound imply t/L>=g_*^(-1)-o(1).
Together these prove the precise objective asymptotic

    min J_lambda / (lambda L^r) -> g_*^(-r).                   (5.3)

All normalized accumulation points of global minimizers are maximal-margin directions,
by compactness and continuity. The same conclusion holds for approximate minimizers with
additive objective error o(lambda L^r), or objective ratio 1+o(1). A fixed multiplicative
approximation factor greater than one does not establish maximal margin by this argument.

Section 4 permits the unrestricted optimization over every finite width to be carried out
at width d: it preserves CE and reduces the penalty. Thus the optimal values for all widths
>=d coincide, and (5.3) also describes the unrestricted regularized optimal value. At a
smaller fixed width that contains an attaining ansatz, equality with the unrestricted
margin follows if (T) is proved.

### 5.1 Precisely what ansatz achievability follows

Let u_A be any normalized optimal ansatz direction, with margin g_A. If (T) holds, g_A=g_*.
The explicit scaled sequence above, using u_A, is then asymptotically globally optimal in
regularized objective with ratio 1+o(1), and has exactly the ansatz's K features throughout.
So existence of a K-feature maximal-margin solution, and existence of K-feature
asymptotically optimal regularized solutions, need no classification of all maximizers.

This does not by itself prove that exact regularized global minimizers approach u_A, or
that every such limit has that K. Even convergence to u_A would not force exactly K nonzero
Fourier coefficients at finite lambda: extra coefficients may be nonzero and tend to zero.
Any assertion about every limiting feature count needs a rigidity statement and a precise
definition of feature count at the limit.

More quantitatively, radial minimization on any fixed positive-margin direction of margin g
has optimal value asymptotic lambda L^r*g^(-r), by the same two-sided argument. Therefore

    (best regularized objective on the ansatz ray)/(unrestricted optimum)
        -> (g_*/g_A)^r.                                     (5.4)

Thus asymptotic relative optimality of the ansatz ray is EQUIVALENT to its unrestricted
margin optimality. The limit theorem cannot remove the missing margin inequality in section 3.
The known unsharp ceiling gives only (g_*/g_A)^r<=nu^(1-r), not equality to one.

## 6. Verification and current endpoint

The algebraic/interval enumeration completed for p=3,5,7,11,13,17 at r=2/3. It considered
respectively 1,3,10,126,462,6435 square active-constraint systems, with no numerical rejection
of candidates. Each run found one strictly winning positive-amplitude multiset class.

`verify.py` independently builds the winning real n=2,q=2 networks for p=3,5,7,11,13 and
evaluates their complete row/class tables. Their margins agree with the certified programme.
It then performs 20 constrained raw-weight searches permitting DC and mixed frequencies;
none exceeds the ansatz. Four trials report solver failure and are NOT optimization
certificates; their actual feasible ratios are recorded rather than inferred from solver
status. No part of the proof rests on the absence of a numerical counterexample.

`check_reductions.py` checks ten random full-tensor compressions from 23 to 18 neurons at
p=3,n=2. The tensors are preserved to at most 5.4e-15 and costs decrease in every case.
It also checks radial CE minimization on the p=11 ansatz: the objective divided by the
prediction in (5.3) is 1.000158 at log(1/lambda)=200. These check calculations and constants;
the arguments in sections 4 and 5 are the proofs.

The logical endpoint is therefore:

* The ansatz programme is an exact finite vertex problem, with generic uniqueness of its
  feature count, explicit endpoint structure, and certified small-prime optima.
* The unrestricted network optimum exists at finite width; arbitrary width and failure of
  compactness are not additional missing hypotheses.
* The regularization limit and the precise conditional achievability statement are proved.
* Inequality (4.1), equivalently the sharp joint inequality (3.1), is still unproved.
  Neither the vertex certificates nor the width reduction establish it.

The next central mathematical step is a lower bound for A_r on the entire margin polyhedron
C that matches (4.1), or a feasible tensor and decomposition that violates it. It must
account for the cost of cross-frequency components jointly with the worst row/class gap.
A fixed row average only recovers the known unsharp bound; concavity in unrestricted logits
is false; and a cost-decreasing purification cannot be assumed. This isolates the actual
comparison without promoting any of those unsupported reductions to lemmas.

## 7. Second attempt: an unrestricted local theorem and persistent bad minima

The detailed proof is in `LOCAL_THEOREM.md`. It builds on the existing Fourier assembly,
rather than treating that assembly as new. The new conclusion is about the actual full
raw-weight problem, not a pure-neuron proxy:

**At n=2,q=2, every globally optimal ansatz is a strict unrestricted local maximum of
normalized margin, modulo scale and the constant-logit phase family, at every odd prime
and every fixed width containing the ansatz (additional neurons are initialized at zero).**

The proof has three substantive steps. Prime Fourier uncertainty forces every ansatz
vertex of support K to have exactly K active offset pairs; global optimality and strict
concavity force strictly positive row multipliers on them. Next, the first-order logit
derivative has at most 2K possible readout frequencies at each input Fourier mode. The
same uncertainty theorem turns the active gap equalities into vanishing of the entire
centered logit derivative, even with mixed-frequency perturbations. Finally a universal
18-variable exact quadratic-form certificate proves strictly negative Hessian in every
remaining direction except the known phase motions. No global purification is assumed.

The theorem also applies to the dense equal-share ansatz at every odd prime, even when it
is suboptimal. At p=11 its Gamma is 0.4919349550..., while the certified sparse ansatz has
Gamma 0.5357259258..., about 8.9018% larger. A compact-neighborhood argument using the CE
asymptotics of section 5 then proves that suboptimal local minima of regularized CE persist
for every sufficiently small lambda at any fixed width >=15. Their directions approach
the dense phase family. This establishes a genuine optimization obstruction, without
claiming that any particular training run must converge to these minima.

Exact scripts: `local_block.py` proves the universal Hessian lemma;
`local_exact.py` independently checks all p=3 raw weights; `local_duals.py` certifies the
positive multipliers at all six previously certified small primes. The supplementary
`local_multifeature.py` and `local_support_check.py` check the full-weight Hessians and
the Fourier support-counting argument numerically/combinatorially.

This closes LOCAL escape, not global optimality. A distant better unrestricted network
has not been excluded; inequality (4.1) remains the missing global statement. The dense
local maxima also show why proving local stability, or even classifying local maxima,
cannot on its own identify the globally optimal K.
