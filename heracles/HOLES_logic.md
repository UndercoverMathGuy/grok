# H1a: a logical reduction to exact rational inequalities

**PROVEN:** Theorem DA at depth 1/2 and r=2/3, every finite width, with an exact rational
certificate; its pure-network margin corollary follows from the audited reduction.
**OPEN:** the unrestricted global-optimality target. This is a closure of H1a's q=2
proposition, not a closure of every hole in the brief.

This pass follows PROMPT_HOLES.md's priority order and the request to solve logically before
verifying. No inherited files are changed. The new ingredient below is an analytic replacement
for DEPTHA's numerical cubic roots. It gives an exact-arithmetic proof certificate, not a
computer-free proof. Only A = 1/2 and r = 2/3 are claimed in this pass.

## 1. Statement and notation

Let N be any finite number of atoms, c_l complex numbers, and v_l1,v_l2 complex numbers of
modulus one. Define

    h(t1,t2) = sum_l c_l (1+v_l1 t1)(1+v_l2 t2),
    m = min_{|t1|=|t2|=1/2} Re h(t1,t2).

Here m is the output left after aiming a half-depth probe against the population. Theorem DA is

    sum_l |c_l|^(2/3) >= 3^(1/3) (max(m,0))^(2/3).

The proof below reduces this statement to explicitly specified rational inequalities. The exact
certificate and independent numerical verification have both passed; see section 7.

If m <= 0 there is nothing to prove. Write A0 = sum_l c_l. Averaging over the probe torus gives
m <= Re A0 <= |A0|, so A0 is nonzero. Divide the atoms by the positive number |A0|; now |A0|=1.
Set x_l=|c_l|, S=sum_l x_l, mu=m, d=1-mu and w=sqrt(S^2-1). Thus 0<=d<1 and S>=1.
Rotation of the atoms is used only in the matrix estimates below: it does NOT assert that m is
invariant under rotating the output. This resolves the normalization ambiguity in DEPTHA D2.

## 2. Existing lemmas, with their needed logical details

For T a nonempty subset of {1,2}, let C_T=sum_l c_l product_{j in T}v_lj. Averaging the torus
phases with the phase of product_{j in T}t_j fixed kills the other nonconstant characters.
Consequently m <= Re A0 - 2^(-|T|)|C_T|, and

    |C_1|,|C_2| <= 2d,       |C_12| <= 4d.

Multiply all atoms by conj(A0) for the following matrix estimates. The moduli of every C_T and
mirror moment remain unchanged. Writing the rotated atoms as a_l+i b_l gives sum a_l=1 and
sum b_l=0. The triangle inequality in R^2 gives

    S >= sqrt((sum |a_l|)^2+(sum |b_l|)^2) >= sqrt(1+(sum |b_l|)^2).

For any interval I, the mirror moment U_I=sum_l c_l conjugate(product_{j in I}v_lj) differs from
conjugate(C_I) by 2i sum_l b_l conjugate(product_{j in I}v_lj). Hence

    |U_I| <= min(S, |C_I|+2w).

The chain vector is (1,v_l1,v_l1 v_l2); let M be the sum of its outer products weighted by the
rotated c_l. M has diagonal 1, lower entries C_1,C_2,C_12 and upper entries U_1,U_2,U_12.
The sum E2 of its principal two-by-two minors and its determinant E3 satisfy

    E2 = 3-C_1 U_1-C_2 U_2-C_12 U_12,
    E3 = 1-C_1 U_1-C_2 U_2-C_12 U_12+C_1 C_2 U_12+U_1 U_2 C_12.

Cauchy–Binet and the Gram determinant bound give e_j(x)>=|Ej|/3^j. Here e_j is the jth
elementary symmetric sum of the nonnegative numbers x_l. Thus, with

    u=min(S,2d+2w), v=min(S,4d+2w),
    a2=max(0,3-4du-4dv)/9,
    a3=max(0,1-4du-4dv-4d^2 v-4du^2)/27,

we have, for every t>=0,

    product_l(1+x_l t) >= 1+S t+a2 t^2+a3 t^3.                 (1)

This uses the already established Feature Cost machinery, without repeating its proof.

The second tool is DEPTHA D5, now with the ordering made explicit. Sort x in decreasing order
and pad by zeros if necessary. Aiming both slots at atom 1 gives mu <= x1/4+9(S-x1)/4.
Aiming one slot each at atoms 1 and 2 gives mu <= 3(x1+x2)/4+9(S-x1-x2)/4. Therefore

    sum_{l>1}x_l >= b1=max(0,(mu-S/4)/2),
    sum_{l>2}x_l >= b2=max(0,2(mu-3S/4)/3).

Since mu<=1<=S, b1<=3S/8 and b1>=2b2. The latter follows, when b2>0, from
b1-2b2=7S/8-5mu/6>=S/24; when b2=0 it is automatic.
Thus (S-b1,b1-b2,b2,0,...) is sorted and majorizes x: its first and second partial sums are
upper bounds on those of x, and its third is the total S. Concavity of z^(2/3) gives

    sum_l x_l^(2/3) >= (S-b1)^(2/3)+(b1-b2)^(2/3)+b2^(2/3).   (2)

Also sum_l x_l^(2/3)>=S^(2/3). These are the only tools needed besides the next analytic bound.

## 3. New analytic lower bounds; no complex roots

For a polynomial P(t)=1+sum_{j=1}^m a_j t^j with positive coefficients and m=2 or 3, define

    I(P) = c_r integral_0^infinity log P(t) t^(-1-r) dt,
    r=2/3, c_r=r sin(pi r)/pi.

The integral converges. By (1) and the Bernstein identity, the atom cost is at least I(P),
including when P is formed by dropping a nonnegative highest-degree term.

Fix alpha>0 and set p_j(t)=binom(m,j)(alpha t)^j/(1+alpha t)^m, j=0,...,m.
These are nonnegative and sum to one. Weighted AM–GM, with a_0=1, yields

    log P(t) >= m log(1+alpha t)
                + sum_{j=1}^m p_j(t) log(a_j/(binom(m,j)alpha^j)).

Integrating gives

    I(P) >= m alpha^r + alpha^r sum_j L_j log(a_j/(binom(m,j)alpha^j)),
    L_j = c_r binom(m,j) B(j-r,m-j+r).

The beta integral here follows directly by substituting z=alpha t in the integral of p_j.
All its arguments are positive. Summing j p_j=m alpha t/(1+alpha t) and integrating shows
sum_j j L_j=m r. Put B0=sum_j (L_j/m) log(a_j/binom(m,j)) and choose alpha=exp(B0/r).
The logarithmic terms then cancel, proving

    I(P) >= m product_j (a_j/binom(m,j))^(L_j/m).              (3)

For r=2/3, the beta recurrence and B(r,1-r)=pi/sin(pi r) evaluate these exponents exactly:

    m=2: (L1/2,L2/2)=(4/9,1/9),
    m=3: (L1/3,L2/3,L3/3)=(10/27,2/27,4/81).

Thus the two lower bounds on atom cost are

    L2 = 2 (S/2)^(4/9) a2^(1/9),
    L3 = 3 (S/3)^(10/27) (a2/3)^(2/27) a3^(4/81).            (4)

If a coefficient is zero, the corresponding bound is simply zero and valid by nonnegativity
(or by taking decreasing positive coefficients in the integral). Formula (3) is the logical
replacement for finding roots of a cubic. It also avoids any analytic-continuation branch issue.

## 4. Compact domain and analytic corner

The target is T=3^(1/3)(1-d)^(2/3), so T^3=3(1-d)^2. The trivial tool settles
S^2>=3(1-d)^2. In particular it settles d>=1/2 and w>=3/2. It is enough to prove the bound on
the rational rectangle 0<=d<=1/2, 0<=w<=3/2.

On 0<=d<=1/50, 0<=w<=3/25, use u<=2d+2w and v<=4d+2w. The differences between the untruncated
minor lower bounds and the coefficients required for (1+(1-d)t/3)^3 are

    E2_lower - 3(1-d)^2 = d(6-27d-16w),
    E3_lower - (1-d)^3 = d(3-27d-31d^2-16w-40dw-16w^2).

Both brackets decrease on the corner; at its far vertex they are 177/50 and 503/2500,
respectively. They are positive. S>=1-d, so coefficientwise comparison in (1), followed by
Bernstein, proves the target throughout this corner, including the equality point d=w=0.

## 5. Exact verification obligations outside the corner

For a rational cell [d0,d1] times [w0,w1], take rational bounds S0<=sqrt(1+w0^2) and
S1>=sqrt(1+w1^2), with S0>=1. Every actual point has S>=S0 and S<=S1. Let tau=3(1-d0)^2.
Compute u,v,a2,a3 at d1,w1,S1. These give lower bounds for the true coefficients in (1), since
the damage terms increase with d,w,S. In (4) replace S by S0. Each of the following suffices:

    S0^2 >= tau,
    2^9 (S0/2)^4 a2 >= tau^3,
    3^81 (S0/3)^30 (a2/3)^6 a3^4 >= tau^27.                 (5)

These are exact rational comparisons.

For the peel, use b1=max(0,((1-d1)-S1/4)/2) and
b2=max(0,2((1-d1)-3S1/4)/3). They are lower bounds on the actual tails.
They satisfy b1<=3S0/8 and b1>=2b2, since 1-d1<=1<=S0<=S1. Thus for every actual S,
(S-b1,b1-b2,b2) is a sorted majorizing vector, and its cost increases with S. Consequently
(S0-b1)^(2/3)+(b1-b2)^(2/3)+b2^(2/3) is a valid cell lower bound.
For each nonnegative rational z the script constructs a rational l with l^3<=z^2,
and checks that inequality using integers. If the cube of the sum of these three lower bounds
is >=tau, the whole cell passes. No rounding tolerance is accepted as proof.

The script bisects cells at rational midpoints, first respecting the two corner boundaries.
It writes each terminal cell with its tree path and proof method. A replay starts from the
whole rectangle, checks every split, rechecks every terminal inequality, and verifies that
all saved cells were reached. This proves exhaustive coverage, not just a collection of
sampled points. Integer square-root bounds and cube-root bounds are checked by powering back.

## 6. Audit of the network reduction

Theorem P3C-1's six steps are sound with the zero/sign cases and coefficient convention below.

**Step 1.** Use real cosine amplitudes: at frequency k a neuron has embeddings
r_j cos(2pi k x_j/p+alpha_j) and readout rho cos(2pi k c/p+beta).
Set c_l=rho product_j r_j exp(i(sum alpha_j-beta)) and v_lj=exp(-2i alpha_j).
Expanding the n operand cosines and pairing conjugate terms gives

    2^n(z_sum(x)-z_(sum+D)(x))
      = sum_k Re[(1-omega^(-kD)) h_k(omega^(-2kx_1),...,omega^(-2kx_n))],

where omega=exp(2pi i/p). The minus sign in 1-omega^(-kD) is required. These c_l are
2^(n+1) times the Fourier-normalized c_l of ELEGANT when Fourier coefficients use 1/p;
this uniform rescaling must accompany the budget. We use cosine amplitudes throughout this section.

**Step 2.** A block B of a slot partition receives a probability mass function nu_B on Z_p,
spread uniformly over the p^(|B|-1) rows with each prescribed block sum. Its Fourier transform
vanishes unless the mode is constant on B. At mode 2k 1_T it is nuhat_B(2k) if B is contained
in T, one if disjoint, and zero otherwise. Independent blocks give (3.1) of P3C exactly.
The moment body is the linear image of the finite probability simplex, hence compact.
If its polydisc inradius is A*, every collection of moments of modulus <=A* is attained,
including the boundary: this follows from compactness. Therefore each block can prescribe
its moments at all frequencies simultaneously, and different blocks can be chosen independently.
There is no additional independence assumption about individual rows or frequencies.

**Step 3.** Write Lambda for 2^n times the margin. For the SAME density, Lambda is at most each
of the expected gaps for D and -D, hence at most their average. Conjugate W factors cancel
all imaginary parts, leaving sum_k (1-cos(2pi kD/p)) Re g_P^(k). No phase rotation of the
network is used in this step.

**Step 4.** Each g_P is continuous on a compact torus, so its real minimum exists. The simultaneous
moment freedom in Step 2 realizes all minimizers in a single density. Replacing a negative
minimum by zero increases the sum since 1-cos is nonnegative. Set the resulting nonnegative
values to y_k. The density can be chosen independently of D, so the minimum over D remains legal.

**Step 5.** If sum y_k^r=0, the margin is nonpositive and (T) follows immediately. Otherwise put
b'_k=y_k^r/sum_j y_j^r, padding unsupported frequencies with zeros. These shares belong to
the simplex defining Gamma_(1/r), giving exactly the programme inequality in P3C Step 5.

**Step 6.** Ignore zero-cost frequencies. For the remaining ones put cost_k=sum_{l at k}|c_l|^r,
C=sum_k cost_k, b_k=cost_k/C, and kappa0=(n+1)^((1-r)/r). The PFP condition gives
sum y_k^r<=kappa0^(-r) C. Orthogonality of a nonzero real cosine at odd p gives squared norm
p times its amplitude squared divided by two. Arithmetic–geometric mean on the n+1 amplitudes
of each neuron yields P_q >= ((n+1)p/2)^(q/2) C. First bound Lambda by the nonnegative
quantity Gamma_(1/r) kappa0^(-1) C^(1/r), then divide by P_q^(1/r). This order handles
negative margins correctly. Since n+1=nu,

    2^(-n)((n+1)p/2)^(-nu/2)=2(2p(n+1))^(-nu/2)=B,

so gamma_q <= B kappa0^(-1) Gamma_(1/r), as claimed. All-zero networks are excluded from the
normalized ratio or assigned zero by convention. Zero neurons can be deleted.

**Proposition P3C-2 needs a quantifier correction.** Its sufficient implication (b)=>(a) is
valid: average (b) with weights b_k, then at least one partition has average score <=1.
The converse at P3C.md:280–282 silently replaces mixtures of at most |S| populations by
arbitrary finite mixtures. The fixed-support hypothesis does not justify that replacement.
Nor is the union over all finite widths automatically a compact set of populations.
This is a gap in that proof, not a counterexample to its particular atom claim.

The exact corrected equivalence is: (b) holds iff EVERY FINITE probability mixture of
one-frequency populations admits a partition of average score <=1, at the fixed depth A.
For completeness, let f(pop) be the vector with coordinates R_P(pop,A)^r. If (b) fails,
the closed sets {lambda in the partition simplex: lambda dot f(pop)<=1} have empty
intersection. Compactness of the finite-dimensional simplex gives a finite subfamily with
empty intersection. The corresponding finite matrix game has min_lambda max_pop payoff >1
(the minimum is attained). Finite linear-programming duality gives a finite probability
mixture with min_P average payoff >1. This proves the contrapositive without asserting
population compactness or continuity across changing widths. The forward direction is the
averaging argument already given. Only the sufficient direction is used by Corollary DA-2.

There is also a repair of the fixed-support converse when K=|S| is at least the number M of
partitions. In the finite game just obtained, let lambda* be an optimal partition mixture,
b* an optimal population mixture, and v>1 its value. Every population used by b* has
lambda* dot f=v: all have payoff <=v against lambda*, and their average payoff equals v.
Thus their score vectors lie in an affine hyperplane of dimension at most M-1. Their average
can be expressed using at most M of them. To see this without an extra compactness assumption,
if more than M vectors have positive weights, their affine dependence gives a nonzero change
of weights preserving both their sum and their weighted average; increase this change until
one weight reaches zero, and repeat. This yields a violating mixture on at most M populations.
Assign these populations to distinct frequencies in S and scale each population's amplitudes
to realize its prescribed cost share. Every complex population is realizable by pure cosines:
choose alpha_j=-arg(v_j)/2, choose beta to realize arg(c), and balance the positive amplitudes
to realize |c|. If unused frequencies must be nonzero, give them a sufficiently small total
cost share of clean blocks; the strict violation survives. This contradicts (a).
For n=2 there are exactly two partitions, so this repairs the converse on every support with
K>=2, precisely the multi-frequency case in H1a. For smaller supports the original converse
is not established here. This finite-game argument requires no claim that all populations
form a compact space.

**One harmless inherited error.** DEPTHA.md:61 claims simultaneous attainability of all D1
bounds with h=1-2d t1-2d t2+4d t1t2. At t1=t2=exp(i pi/3)/2 its real value is 1-3d/2,
not >=1-d. So this example does not have the stated aiming loss. None of the proof uses
simultaneous attainability; bounding the three coefficients separately remains valid.

## 7. Verification record

`holes_logic_certificate.py` passed using only Python integer and Fraction arithmetic.
The complete rectangle has 313 terminal cells: 1 analytic corner, 96 cubic-bound cells,
2 quadratic-bound cells, 211 peel cells and 3 trivial cells. All 313 were replayed successfully.
There were 417 nodes in the subdivision tree. Square- and cube-root rational enclosures use
denominator 10^24; their directions are checked by exact integer powering, not inferred from
floating-point rounding. The files are `FINAL_RUNS/HOLES_logic/certificate.json` and
`FINAL_RUNS/HOLES_logic/summary.json`. This is a computer-assisted exact proof.

`holes_logic_verify.py` independently checked the analytic bounds against numerical quadrature
on 200 positive polynomials and checked their two repeated-root equality cases. It tested
500 atom populations (278 with positive aimed output): no atom bound failed; the smallest
observed theorem ratio was 1.01836. These are sanity checks, not the proof certificate.
On 24 full-table raw-weight networks with n=2,3 and p=5,7,11, the largest profile-identity
error was 9.95e-14 and the largest density-identity error was 9.77e-15. Twenty simultaneous
half-depth moment LPs on p=11, S={1,4} had residual at most 5.83e-16. Six single-frequency
clean blocks at p=3,5,7,11,17,31 attained their restricted-support ceilings to relative error
7.22e-14. Results are in `FINAL_RUNS/HOLES_logic/verification.json`.

Reproduce from the repository root:

    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_logic_certificate.py
    UV_CACHE_DIR=/private/tmp/heracles-uv-cache uv run --offline python heracles/holes_logic_verify.py

## 8. Consequences and limits

Theorem DA at A=1/2,r=2/3 and the audited sufficient reduction prove the following paper wording:

> For full-table two-operand modular addition modulo an odd prime, every gated-chain network
> of pure single-frequency neurons whose support has aiming depth at least one half satisfies
> gamma_2 <= 3^(-1/2) B Gamma_(3/2)(p), regardless of width or same-frequency junk.

Here B=2(6p)^(-3/2). For a single frequency, the moment body is the regular p-gon, so its
inradius is cos(pi/p)>=1/2 for all odd primes p, including p=3. This removes both the width
restriction and the finite-p relaxation in the earlier single-frequency result.
The result also covers supports whose 3^K signed sums of modes 2k are all distinct modulo p:
P3A's explicit nonnegative Riesz density then realizes all moments of modulus at most 1/2.
This is the precise distinct-sums assumption used there, rather than a weaker interpretation
of the word "dissociate."
For a restricted support, the same proof uses Gamma maximized on that support. Its
share-weighted block attains that restricted ceiling. Attainment of the unrestricted Gamma
ceiling additionally requires an optimal support satisfying the depth assumption; it cannot
be asserted for every restricted support. No uniqueness of optimal feature count is asserted.

H1b, H1c, H3, H2, H1e, H5, H1d and H6 are not resolved by this argument. In particular, it does
not establish depth 0.47, other q, impure networks with junk, or the unrestricted target (T).
The brief asks to stop and write up upon closing the first hole; this pass follows that scope.

| Hole | Status of this pass |
|---|---|
| H1a(i) | **PROVEN** at A=1/2, r=2/3, every finite width; exact certificate replaces fp64 roots. Other sampled exponents and A=0.47 are not hardened here. |
| H1a(ii) | **PROVEN** sufficient reduction with zero/sign cases explicit. Fixed-support minimax converse repaired when K is at least the number of partitions, including n=2,K>=2. General smaller-K converse remains **OPEN**. |
| H1b | **OPEN**, no paper-support inradius certificates added. |
| H1c | **PARTIAL**: n=2,q=2,A=1/2 follows by putting all partition weight on singletons; lower depths and n=3 remain **OPEN**. |
| H1d, H1e | **OPEN**, no new higher-degree or impure-network result. |
| H2 | **PARTIAL**: the Re-form at n=2,r=2/3 follows since m_1<=m_(1/2); no general C1/C2 or E+ proof. |
| H3 | **OPEN**, no optimal-support or count-uniqueness certificate. |
| H4 | Scope retained: full table, pure class for the new corollary, q=2; training-dynamics citations not independently audited here. |
| H5, H6 | **OPEN / not pursued**, no new claim. |

The simultaneous-attainability example at DEPTHA.md:61 is **REFUTED** as specified in section 6;
this does not refute Theorem DA or target (T). No counterexample to (T) was found or claimed.
