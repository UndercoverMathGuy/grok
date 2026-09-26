# H1b: a finite, checkable certificate for aiming depth

**PROVEN:** H1b lower bounds above 0.35 on all five previously labelled optimal supports,
and on all three later LADDER supports. The depth-0.47 atom theorem is also now certified,
closing the pure-network margin theorem on the recorded p=71 support without hypothesis (H).
**OPEN:** the full unrestricted theorem and the general lower-depth mixture inequality.

This continuation first derived the logical certificate, then ran it. All reported certificates
have passed exact residual checks and exhaustive replay. The inherited
`astar_papers.json` supports and the later LADDER supports are different and are kept separate.

## 1. Symbols and the polar reformulation

Let p be an odd prime and S=(k_1,...,k_K) distinct elements of {1,...,(p-1)/2}. Set
E_ak=exp(-4 pi i a k/p), for rows a=0,...,p-1. The aiming-depth moment body is the convex hull
of the p vectors (E_ak)_k in C^K. Its polydisc inradius A* is

    A* = inf_{z != 0} max_a Re sum_k E_ak z_k / sum_k |z_k|.

Writing f_z(a)=Re sum_k E_ak z_k, the mean of f_z is zero, and f_z is nonzero when z is nonzero
by Fourier orthogonality. Thus max f_z>0 for z!=0. Normalize max f_z<=1 and define

    P = {z : f_z(a)<=1 for every a},    M = max_{z in P} sum_k |z_k|.

Then A*=1/M by positive rescaling. The maximum exists: P is closed, and each |z_k|<=2.
Indeed g=1-f_z is nonnegative with mean 1, so every Fourier coefficient of g has modulus <=1;
the coefficient at mode 2k is -z_k/2 or its conjugate. In particular each real coordinate and
each imaginary coordinate of z has absolute value <=2. This bound also makes the numerical
LP duals rigorously repairable, rather than treating the solver's feasibility tolerance as proof.

## 2. Cyclic symmetry removes one phase dimension

Let L be an integer with 3<=L<=p, and let c_L=cos(pi/L)>0. Define L^(K-1) direction vectors

    d_1=1,    d_j=exp(2 pi i h_j/L),  h_j in {0,...,L-1}, j=2,...,K.

For each d let U_d be any certified upper bound on the linear-programme value

    max_{z in P} Re sum_k conjugate(d_k) z_k,

and put U=max_d U_d. Then

    A* >= c_L/U.                                             (1)

Proof: P is invariant under z_k -> z_k exp(-4 pi i b k/p), because that permutes its row
constraints. Since 2k_1 is invertible modulo p, choose b so the rotated first coordinate has
phase in [-pi/p,pi/p] (if it is zero any b works). For each other coordinate choose its nearest
Lth-root direction. The first coordinate's projection is at least cos(pi/p)|z_1|, and every
other projection is at least c_L|z_j|. Since p>=L, cos(pi/p)>=c_L. Hence one enumerated d has
Re sum conjugate(d_k)z_k >= c_L sum |z_k|. Maximizing and using the LP bounds gives M<=U/c_L,
which is (1). No continuity mesh in magnitudes, no global phase invariance, and no unproved
optimiser lower bound are being used. The cyclic action, not arbitrary output rotation, is
what permits fixing the first direction.

## 3. Exact repair of each LP dual

Use real coordinates x=(Re z,Im z), with H=[Re E,-Im E], so P={x:Hx<=1}.
The direction is the real vector d=(Re d,Im d). For ANY nonnegative vector y in R^p,

    d dot x = y dot (Hx) + (d-H^T y) dot x
            <= sum_a y_a + 2 sum_j |d_j-(H^T y)_j|.           (2)

Thus (2) is a rigorous upper bound even when y is an approximate dual solution. The script
rounds nonnegative dual multipliers to rational numbers with denominator 10^12. It encloses
each sine and cosine and each direction coordinate in rational intervals, with endpoints on
a grid of denominator 10^30, using directed interval arithmetic. Integer dot products then
give an exact upper bound for the residual in (2). The LP solver is only a source of candidate
y: every accepted bound is checked independently of its primal/dual feasibility flags.

All L^(K-1) direction indices must occur exactly once. A separate replay verifies that complete
enumeration and every residual bound. The lower endpoint of the interval for cos(pi/L),
divided by the largest certified U_d, proves (1).

## 4. Upper witnesses are also checked

Any explicit nonzero z gives A*<=max_a f_z(a)/sum_k |z_k|. A candidate selected during the LP
search is rounded to rational real and imaginary coordinates. The numerator is bounded above
using the same trigonometric intervals; each modulus in the denominator is bounded below by
an integer-checked square-root enclosure. This yields a rigorous upper bound. An upper bound
below 0.47 refutes availability of that depth on that particular support; it does NOT refute
the network target (T).

## 5. Why the naive perturbed Riesz route is weak

For lambda in the radius-t polydisc, t<=1/2, the nonnegative Riesz product has normalization
1+E0(lambda) and kth moment numerator lambda_k+Ek(lambda). Let N0,j and Nk,j count the
nontrivial signed-sum representations of 0 and k of degree j. Then |Ei|<=sum_j Ni,j t^j.
Brouwer's fixed-point theorem applied to lambda_k=a_k(1+E0)-Ek shows that radius A is
attainable whenever

    A (1+sum_j N0,j t^j) + max_k sum_j Nk,j t^j <= t.

The normalization cannot vanish: each factor is strictly positive for t<1/2, while at t=1/2
and odd p each factor can vanish on at most one row; K<p leaves a positive row.
This logical bound is valid but weak. A preliminary scan gives only about 0.19,0.32,0.25,0.28,
0.21 on the five previously labelled K_star supports at p=59,71,97,113,127. These are scan
values for this sufficient bound, not claims about A* itself. The finite LP certificate above
avoids the absolute-value sum of all collision errors.

## 6. Results

The following are certified intervals, with displayed lower endpoints rounded down and upper
endpoints rounded up. "Old" means the K=K_star row of `FINAL_RUNS/P3C/astar_papers.json`;
"LADDER" means the later table at LADDER.md:252–256. Supports are explicitly listed to avoid
silently treating different supports as the same theorem.

| Source | p | Support S | Certified lower <= A* <= certified upper |
|---|---:|---|---|
| Old | 59 | 1,4,12,18,25,27 | 0.358249 <= A* <= 0.382550 |
| Old | 71 | 1,7,12,16,34 | 0.473741 <= A* <= 0.479643 |
| Old | 97 | 1,4,10,28,35,47 | 0.407377 <= A* <= 0.461218 |
| Old | 113 | 1,5,12,15,38,46 | 0.413104 <= A* <= 0.473769 |
| Old | 127 | 1,11,14,20,36,38,43 | 0.381914 <= A* <= 0.434801 |
| LADDER | 59 | 1,6,14,16,19 | 0.400911 <= A* <= 0.449420 |
| LADDER | 71 | 1,4,11,17,31 | 0.473741 <= A* <= 0.479643 |
| LADDER | 113 | 1,6,16,31,35,44,55 | 0.373393 <= A* <= 0.422217 |

The old and LADDER p=71 supports are equivalent: multiply the old frequencies by 31 modulo
71 and identify k with -k. The resulting set is exactly the LADDER set. Multiplication
permutes rows, and identifying signs conjugates individual moment coordinates; both preserve
the polydisc and A*. Thus the stronger p=71 certificate transfers to both supports.

Certificates use L=8 for old p=59 (32,768 directions), L=6 for old p=97,113 (7,776 each),
L=6 for old p=127 and LADDER p=113 (46,656 each), L=6 for LADDER p=59 (1,296), and L=16
for LADDER p=71 (65,536). In total these selected certificates check 208,464 LP directions.
All dual residual checks after trigonometric enclosure are exact integer operations. The
trigonometric constants use 70-digit directed interval arithmetic. Gzipped dual certificates,
exact rational endpoints, rational upper witnesses, and SHA-256 hashes are in
`FINAL_RUNS/HOLES_aiming/`. L=4 preliminary certificates are retained there as well.

This **closes H1b at the requested useful depth level**: A*>0.35 on every listed support.
It does not prove H1c, and it does not assert those supports globally optimize Gamma.
The old optimizer estimates remain valid *upper* bounds; the new witnesses show they were
too loose to infer depth 0.47. In particular, depth 0.47 is **REFUTED** for old p=97,127 and
LADDER p=59,113. The endpoints certify genuine separating directions in the moment body,
not merely failed attempts to construct densities.

## 7. Extending the exact atom certificate to A=47/100

The analytic inequality (3) in HOLES_logic.md does not depend on aiming depth. Keep its r=2/3
and the same two explicit bounds L2,L3. Only the coefficient and peel inputs change. For a
normalized positive aimed output mu=1-d and S=sqrt(1+w^2), put

    D=d/A, T=d/A^2,
    u=min(S,D+2w), v=min(S,T+2w),
    a2=max(0,3-2Du-Tv)/9,
    a3=max(0,1-2Du-Tv-D^2 v-u^2 T)/27,
    b1=max(0,(mu-(1-A)^2 S)/(4A)),
    b2=max(0,(mu-(1-A^2)S)/(2A(1+A))).

The same coefficient and tail arguments prove these inputs. The sorted-majorization detail
continues to hold at A=47/100: b1<=(2-A)S/4<S/2. When b2>0,

    b1-2b2 >= S(2-3A-A^2)/(4(1+A)) >0,

because the coefficient of mu in b1-2b2 is negative and mu<=S. When b2=0 the claim is
immediate. These facts also hold for the cell tail bounds using mu=1-d1 and S1, with
b1<=(2-A)S0/4 because mu<=1<=S0<=S1. Therefore the lower-bound cell proof is unchanged.

For the analytic corner use d<=dc=1/100 and w<=wc=3/50. Let

    a1=2/A+1/A^2, a2c=2/A^2+1/A^4.

The second-minor bracket is 6-d(3+a2c)-2w a1. The third-minor bracket is

    3-3d+d^2-d a2c-2w a1-(d/A^2)(d/A^2+2w)-(d/A+2w)^2/A^2.

Drop its positive d^2 term to get a lower bound decreasing in d,w. At the far corner the
two resulting brackets are, exactly,

    2254734357/487968100 >0,    743089657/487968100 >0.

This proves the corner by coefficientwise comparison with (1+(1-d)t/3)^3. The remaining
rectangle is covered by the same rational cell rules as before. `holes_aiming_depth47.py`
passed and replayed 916 terminal cells (1 corner, 171 cubic, 739 peel, 5 trivial), with no
floating-point arithmetic in the certificate. Thus Theorem DA is now **PROVEN at depth 0.47**,
and by monotonicity at every larger depth, for r=2/3 and every finite number of atoms.

## 8. Attempt at the remaining mixture inequality

The depth certificates do not by themselves prove the mixture condition. I tested a logical
three-variable extension of the previous atom bound. Write t=|C_12| and retain d,w as above.
The singleton aimed output is mu=1-d; the one-block aimed output is at most (1-A t)_+.
For a singleton mixture weight lambda, the sufficient cost target is

    3^(1/3)[lambda (1-d)^(2/3)+(1-lambda)(1-A t)_+^(2/3)].     (3)

The individual coefficient estimates can be strengthened. Aim t1 against C1 and t2 against
C2 simultaneously. The remaining quadratic term has real part at most A^2|C12|, giving

    |C1|+|C2| <= d/A+A t,      |C1|,|C2|<=d/A.

Also the triangle inequality for C12 gives x1<= (S+t)/2, so the first peel tail can be
increased to max(b1,(S-t)/2). These are actual inequalities, not numerical conjectures.
Let D=d/A, L=min(2D,D+A t), v=min(S,t+2w). The damage in the minors is bounded by

    H=min(S L, D^2+(L-D)^2+2w L)+t v,
    E2_lower=3-H,
    E3_lower=1-H-(L/2)^2 v-min(S^2,(L/2+2w)^2)t.

For the first bound, sum delta_i u_i <= min(S sum delta_i,sum delta_i^2+2w sum delta_i),
and at fixed sum <=L and individual cap D, sum delta_i^2<=D^2+(L-D)^2. For the product
terms use delta1 delta2<=(L/2)^2 and u1 u2<=min(S^2,(L/2+2w)^2). These bound the same
chain matrix as before, with a2=(E2_lower)_+/9 and a3=(E3_lower)_+/27.

This strengthened relaxation STILL DOES NOT PROVE the mixture inequality. At A=0.35,
lambda=0.6, a preliminary scan finds cost-bound/target about 0.9598 near
(d,w,t)=(0.06875,0.35355,0.33273). This is a failure of these scalar sufficient bounds,
not an atom population or a counterexample to the conjecture. The precise rational point
and interval-verified failure are recorded by the verification script below. The missing
information is a stronger coupling of the three junk coefficients with phase misalignment;
simply extending the old certificate by one variable is not enough here.

At the exact rational point

    A=7/20, d=11/160, w=7/20, t=1/3,

interval arithmetic puts the largest of the four cost lower-bound tools below 1.307010905, whereas the target for
lambda=3/5 is at least 1.356327739. In fact the bound is below BOTH individual partition
cost targets, so it fails for EVERY mixture weight lambda in [0,1] at this point. This is
verified by comparing all four upper endpoints with
3^(1/3) min((1-d)^(2/3),(1-A t)^(2/3)). Therefore changing the mixture weight alone cannot
repair this particular scalar relaxation. H1c remains open and needs additional information.

## 9. A paper-support theorem now follows

Combine the audited P3C reduction from HOLES_logic.md, the exact depth-0.47 theorem in
section 7, and A*>0.473741 on the p=71 LADDER support. This gives the following paper wording:

> For full-table two-operand modular addition modulo 71, among gated-chain networks whose
> neurons are pure single-frequency cosines with frequencies in S={1,4,11,17,31}, the maximum
> normalized margin under q=2 is 3^(-1/2) B Gamma_(3/2)(71;S), with B=2(6*71)^(-3/2).
> The upper bound holds at every finite width and without any assumption that same-frequency
> junk vanishes. Share-weighted blocks attain the maximum using three neurons per active frequency.

Here Gamma_(3/2)(71;S) is the share programme restricted to S, including zero shares. The
maximum exists by compactness of its share simplex. The upper bound is the audited reduction;
the lower bound is the existing share-weighted block construction evaluated at any maximizing
share vector. Thus this is an exact optimization theorem on the stated class, not just a
comparison with a constructed network. The same theorem holds on every nonempty subset of S:
the moment polydisc projects onto the smaller support. The old p=71 support is equivalent as
shown in section 6.

The qualifiers are essential: this does NOT optimize over all frequency supports, does NOT
allow impure neurons, and does NOT certify the numerical value or uniqueness of the optimizer
in the Gamma programme. Those are the remaining global-theorem and H3 issues.

## 10. The singleton route is provably insufficient at the actual p=113 support

Consider the exact two-atom population c1=c2=1/2, v1=(1,1), v2=(-1,-1). It has
h(t1,t2)=1+t1*t2 and cost 2^(1/3). Its singleton partition score is

    R_singleton = sqrt(3/2)(1-A^2).

This exceeds one whenever A<sqrt(1-sqrt(2/3)), approximately 0.428373. The certified upper
witness for the LADDER p=113 support gives A*<0.422217<0.423. At the rational value
A=423/1000 the exact squared score is (3/2)(1-(423/1000)^2)^2>1, as checked with integers.
Consequently the universal singleton-partition criterion is **REFUTED on that support**.
The population itself is explicit; the comparison involves no optimizer. This does not
refute (T): the partition mixture can still succeed. It proves that merely sharpening the
single-partition depth certificate cannot cover this support using the available A*.

## 11. Reproduction, verification and status

Run from the repository root, with only local CPU resources:

    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_certificate.py --names old_59 --L 8
    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_certificate.py --names old_97 old_113 old_127 ladder_59 ladder_113 --L 6
    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_certificate.py --names old_71 --L 4
    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_certificate.py --names ladder_71 --L 16
    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_depth47.py
    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_aiming_verify.py

Add `--verify-only` to an aiming-certificate command to replay stored duals without running
any optimization. Every generation already does a complete replay. The p=71 L=16 certificate
was additionally replayed in a fresh process.

Independent fp64 checks reconstruct each rational upper witness using direct complex
exponentials, agreeing with its interval-certified bound to <1e-10. One thousand random
polar directions on each of the eight supports exceed the certified lower bound (sanity
checks only). At depth 0.47, 250 independent atom populations were checked, 141 with positive
aimed output; the minimum observed theorem ratio was 1.010309 and every analytic tool stayed
below the actual atom cost. Thirty arbitrary-phase moment LPs at p=71, depth 0.47, were
feasible, with maximum residual 6.87e-15. Exact and interval certificates, not these random
checks, establish the claims.

| Hole | Status after this continuation |
|---|---|
| H1a | **PROVEN extension:** depth 0.47 at r=2/3, every finite width; other exponents remain unverified. |
| H1b | **PROVEN:** A*>0.35 on all five old K_star supports and all three later LADDER supports; p=71 has A*>0.473741. |
| H1c | **PARTIAL:** the p=71 pure fixed-support theorem closes via the singleton partition. The mixture bound at A=0.35 remains **OPEN**. The strengthened scalar relaxation is insufficient for every mixture weight at an explicit rational point. |
| H1d, H1e | **OPEN:** higher degree and impure networks are not covered. |
| H2 | No further general C1/C2 or E+ result; the inherited Re-form special case remains available. |
| H3 | **OPEN:** support optimality and count uniqueness are not certified. Support labels were reconciled explicitly. |
| H4 | Scope stated precisely; no new assertion about trained endpoints or optimization dynamics. |
| H5, H6 | **OPEN / not pursued in this continuation.** |

The requested depth-0.47 availability is **REFUTED** on several specific supports, and the
singleton criterion is **REFUTED** on the actual p=113 support. Neither refutes target (T).
Only new files were written; no inherited files were changed and no commits were made.
