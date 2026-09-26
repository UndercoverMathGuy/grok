# Arbitrary depth: an analytic upper-exponent range, and an exact obstruction below it

Status: the full range 2<=q<=nu at arbitrary depth is NOT proved here. The following
range is proved analytically, with no numerical optimization or interval certificate:

    nu>=3,    nu-1<=q<=nu.

In this range:

* At p=3, the ansatz is globally margin optimal among all real gated-chain networks,
  of arbitrary finite width, with DC and mixed-frequency components allowed.
* At every odd prime p, it is globally optimal within the pure one-frequency class,
  with arbitrary width and arbitrary junk.
* For q<nu, equality in these sharp bounds forces zero junk and exactly nu nonzero
  neurons; in the unrestricted p=3 result, input/readout DC must also vanish.

Section 5 gives a slightly wider explicit exponent range. Section 6 proves that the
largest-junk-coefficient inequality used here is FALSE at q=2 for every nu>=4.
That is an obstruction to this particular extension, not a counterexample to the
network margin conjecture. The general-prime unrestricted multifeature problem remains
open even within the exponent range above.

## 1. Population statement

Put D=nu and n=D-1. For arbitrary complex c_i and unit complex v_ij, define

    G = sum_i c_i,
    J_T = sum_i c_i prod_{j in T} v_ij,    empty != T subset {1,...,n},
    delta = max_{T!=empty} |J_T|,
    r = q/D.

For D>=3 and (D-1)/D<=r<=1,

    sum_i |c_i|^r >= D^(1-r) (|G|-delta/2)_+^r.                (1.1)

D=3 is already proved in P3_ANALYTIC.md, including the endpoint r=2/3. Below we prove
the statement for every D>=4, so n>=3. There is no restriction on the number of atoms.

Normalize G=1 by dividing c_i by G. Cases G=0 or delta>=2|G| are trivial and can be
removed first. In this normalization write

    c_i=x_i exp(i alpha_i),
    e=sum_i x_i>=1,
    u=sqrt(2e(e-1)),    z=1-delta/2 in (0,1].

## 2. A balanced phase cut improves the matrix construction

Let b=floor(D/2), and define the prefix products

    P_i0=1,    P_ij=prod_{ell=1}^j v_iell,    j=1,...,n.

Form the D-vector m_i whose jth entry, indexed j=0,...,n, is

    (m_i)_j = P_ij                    for j<b,
              exp(i alpha_i) P_ij     for j>=b,

and let M=sum_i x_i m_i m_i^*. The matrix is positive semidefinite with diagonal e.
Every lower-triangular entry crossing the cut is an interval junk coefficient J_T.
Every lower-triangular entry within either half differs from the corresponding J_T
by at most

    sum_i x_i |1-exp(i alpha_i)| <= sqrt(2e(e-1))=u.

There are b(D-b) crossing pairs and

    R = b(b-1)/2 + (D-b)(D-b-1)/2

within-half pairs. Put

    a=2R/D,
    F(e,u,delta)=e^2+a*u^2+2a*delta*u+n*delta^2.

Then

    tr M=D e,    tr M^2 <= D F.                               (2.1)

The balanced cut gives

    a=(D-2)/2                 if D is even,
    a=(D-1)^2/(2D)            if D is odd,

so 0<a<=n/2. It absorbs the amplitude phase in roughly half of all interval entries;
placing that phase in all but the first coordinate would leave more uncontrolled entries.

Exactly as in P3_ANALYTIC.md, Gram-matrix concavity and Holder imply, for 0<r<1,

    sum_i x_i^r
       >= D^(-r) (D e)^(2-r)/(D F)^(1-r)
       = D^(1-r) e^(2-r)/F^(1-r).                             (2.2)

The elementary proofs in that file apply in dimension D unchanged. No spectral
subadditivity theorem or eigenvalue case enumeration is needed.

## 3. The scalar inequality at every depth

For n>=3, 0<a<=n/2, e>=1, u=sqrt(2e(e-1)), and 0<=delta<2,

    (1-delta/2)^n F(e,u,delta) <= e^(n+2).                     (3.1)

### 3.1 First establish the inequality at delta=0

Set t=1-1/e in [0,1). Then

    F(e,u,0)=e^2[1+2a*t] <= e^2[1+n*t] <= e^2(1-t)^(-n)
             =e^(n+2).

The last inequality follows from the tangent-line lower bound for (1-t)^(-n).
For e>1 it is strict.

### 3.2 Show that adding delta decreases the left side

The derivative of (1-delta/2)^n F is nonpositive if

    Q := F - ((2-delta)/n) partial_delta F >=0.

Expansion gives

    Q=e^2+a*u^2-(4a/n)u
         +[2a(n+1)/n]u*delta-4delta+(n+2)delta^2.

Since e^2>=1, it suffices to bound the unconstrained quadratic

    1 + x^T H x - 2 l^T x,
    x=(u,delta)^T,
    H=[[a, a(n+1)/n], [a(n+1)/n, n+2]],
    l=(2a/n,2)^T.

H is positive definite: its determinant divided by a is

    L=n+2-a(n+1)^2/n^2
      >= n+2-(n+1)^2/(2n)>0.

Completing the square gives a lower bound 1-l^T H^(-1)l, and direct inversion gives

    l^T H^(-1)l = 4(1-a/n)/L.

This is strictly less than one, because

    L-4(1-a/n)
      =n-2-a(n-1)^2/n^2
      >=n-2-(n-1)^2/(2n)
      =(n^2-2n-1)/(2n)>0,           n>=3.

Thus Q>0. The left side of (3.1) strictly decreases with delta, proving (3.1),
strictly whenever delta>0. This proof bounds the quadratic on all real u,delta,
so it loses no cases from their actual nonnegative domain.

### 3.3 Finish the population proof

Combine (2.2) with (3.1):

    sum_i x_i^r
      >= D^(1-r) e^((n+1)r-n) z^(n(1-r))
      >= D^(1-r) z^r,

because r>=n/(n+1), e>=1, and 0<z<=1. At r=1 use sum_i x_i=e>=1>=z directly.
Rescaling proves (1.1).

For r<1, equality forces delta=0 and e=1. All positive c_i have the same phase, M=I_D,
and its Gram matrix is a rank-D orthogonal projection with diagonal D x_i. Equality
sum_i (D x_i)^r=D forces every positive D x_i to equal one. There are exactly D
nonzero atoms of equal amplitude. This repeats the elementary equality argument in
P3_ANALYTIC.md in dimension D.

## 4. Carry the population result through to actual network margins

### 4.1 Unrestricted networks at p=3

Use normalized Fourier coefficients and center the readout. Every readout then has
frequencies +1,-1 only. Write the clean atom as

    c_i = W_i(1) prod_j conj(E_ij(1)),
    v_ij = E_ij(1)/conj(E_ij(1)),

with arbitrary phase when a coefficient vanishes. Parseval and AM-GM give

    ||theta_i||^2 >= 6D |c_i|^(2/D),
    P_q >= (6D)^(q/2) sum_i |c_i|^r.                          (4.1)

For every nonempty subset T, choose the wrong offset d uniformly from {1,2}, and choose
the input uniformly subject to

    sum_{j in T} x_j + d/2 = s in Z_3.

Here d/2 uses the inverse of 2 in Z_3. The expected gap is exactly

    E gap = 3 Re(G-J_T omega^(2s)),    omega=exp(2*pi*i/3).      (4.2)

Proof for an arbitrary Fourier component with readout frequency +1: averaging outside
T forces each corresponding input frequency to be -1. Averaging on the affine
hyperplane forces all input frequencies within T to equal the same value b in Z_3.
The offset multiplier is

    (1/2) sum_{d=1,2} (1-omega^d) omega^(-(b+1)d/2).

For b=-1,+1,0 this is respectively 3/2,-3/2,0. The first two terms give G and J_T;
the third eliminates DC inputs. Readout frequency -1 supplies the conjugates. This
proves (4.2) at every depth, including T containing all inputs.

Choosing T with |J_T|=delta and a suitable cube root gives

    m <= 3(Re G-delta/2) <= 3(|G|-delta/2).                    (4.3)

Combining (1.1), (4.1), and (4.3) yields the sharp unrestricted ceiling

    gamma_q <= 3 D^(-(D-q)/q) (6D)^(-D/2),
    p=3, D>=3, D-1<=q<=D.                                   (4.4)

This is exactly the ansatz value, since Gamma_(D/q)(3)=3/2. For attainment take D
neurons, l=0,...,D-1, with every input phase pi*l/D and readout phase n*pi*l/D.
The resulting logits are

    (D/2^(D-1)) cos(2*pi*(sum_j x_j-c)/3),

the margin is 3D/2^D, and P_q=D*(3D/2)^(q/2), giving equality in (4.4).

For q<D equality forces zero junk and exactly D nonzero neurons. Equality in the cost
bound also forces absence of DC, and balanced slot norms. Thus this is a statement
about actual unrestricted global optimizers, not merely an attainable lower bound.

### 4.2 Pure one-frequency networks at every odd prime

Suppose every neuron has all slots in the same fixed real Fourier plane {+k,-k},
k!=0. No restriction is imposed on width, atom phases, or junk. Write

    h(x)=sum_{T subset {1,...,n}} J_T exp(4*pi*i*k*sum_{j in T} x_j/p),
    J_empty=G.

The average of gaps at offsets d and -d at a fixed input is

    2(1-cos(2*pi*k*d/p)) Re h(x).

Averaging h on the hyperplane sum_{j in T} x_j=s, with all outside variables free,
retains only G and J_T exp(4*pi*i*k*s/p). Hence

    min_x Re h(x) <= Re G-cos(pi/p)*delta <= |G|-delta/2.

Choose d with kd=1 mod p. With A_min=1-cos(2*pi/p),

    m <= 2 A_min (|G|-delta/2),
    P_q >= (2pD)^(q/2) sum_i |c_i|^r.

The resulting sharp class ceiling is

    gamma_q <= D^(-(D-q)/q) * 2(2pD)^(-D/2) * A_min,
    D>=3, D-1<=q<=D.                                         (4.5)

The D-neuron block attains it; for q<D equality forces exactly D nonzero neurons
and zero junk. This does not compare that one-feature value with the optimal
multifeature ansatz when the latter has a larger Gamma value.

## 5. A slightly wider explicit range from the same proof

The preceding range is easy to state but conservative. Keep the exact balanced-cut a,
and define

    rho_* = max(2a, sqrt(1+4n+a/(n-a))-1),
    q_* = D*rho_* /(1+rho_*).                                 (5.1)

For D>=4, all the population and network conclusions hold for q_*<=q<=D. For D=3
use the stronger earlier result q>=2.

Proof. Put rho=q/(D-q) for q<D. The delta=0 estimate generalizes to
F(e,u,0)<=e^(rho+2) whenever rho>=2a, using
1+2a*t<=1+rho*t<=(1-t)^(-rho).
For monotonicity of (1-delta/2)^rho F, the relevant quadratic matrix and vector are

    H_rho=[[a, a(1+1/rho)],
           [a(1+1/rho), n(1+2/rho)]],
    l_rho=(2a/rho, 2n/rho)^T.

Direct inversion shows

    l_rho^T H_rho^(-1) l_rho
      = 4n(n-a) / [(n-a)rho(rho+2)-a].

The condition rho>=sqrt(1+4n+a/(n-a))-1 makes the denominator at least the
positive numerator. It also implies H_rho is positive definite. Therefore the same
completed-square argument proves

    (1-delta/2)^rho F <= e^(rho+2).

Substitute in (2.2). Since rho=r/(1-r), the powers of e cancel and the power of z
is exactly r. This proves (1.1) and all subsequent margin bounds.

Equality still requires e=1 and delta=0: when e>1 the delta=0 tangent estimate is
strict. When e=1, u=0, and the derivative quadratic is

    1-(4n/rho)delta+n(1+2/rho)delta^2.

Its minimum is 1-4n/[rho(rho+2)]>0 under (5.1), since a/(n-a)>0. Thus the
left side strictly decreases away from delta=0 even at the refined endpoint.

Examples, stated exactly:

    D=4: q_* = 4*(1-sqrt(2/27));
    D=5: q_* = 5*(1-sqrt(3/53));
    D=6: q_* = 24/5.

These are sufficient thresholds from this proof, not asserted to be the true thresholds
of validity of (1.1). The simpler q>=D-1 range follows already from sections 1-3.

## 6. Why the same largest-coefficient lemma cannot prove the full q range

For EVERY D>=4, take N=D-1 atoms indexed ell=1,...,D-1. Let

    zeta=exp(2*pi*sqrt(-1)/D),
    c_ell=1/(D-1),
    v_ell,j=zeta^ell for every input slot j=1,...,D-1.

Then G=1. Every nonempty subset has size t in {1,...,D-1}, so

    J_T=(1/(D-1)) sum_{ell=1}^{D-1} zeta^(ell*t) = -1/(D-1).

Thus delta=1/(D-1). At q=2, r=2/D, (1.1) would require

    (D-1)^(1-2/D)
      >= D^(1-2/D) (1-1/[2(D-1)])^(2/D).

Raising to the power D/2 and dividing by D^((D-2)/2) would give

    (1-1/D)^((D-2)/2) >= 1-1/[2(D-1)].

This is false: (D-2)/2>=1 makes the left side at most 1-1/D, which is strictly
smaller than the right side. At D=4 this is simply sqrt(3)<sqrt(10/3).

This is an analytic counterexample to (1.1) at the low-q endpoint. It does NOT
refute the margin conjecture. The junk coefficients are coherent: at input x=0,

    h(0)=1-(2^(D-1)-1)/(D-1)=(D-2^(D-1))/(D-1)<0.

The corresponding pure network has negative margin, detected by averaging its two
opposite-offset gaps at that row. Bounding the margin using only delta ignores how
all these junk terms reinforce one another.

Thus the remaining range, including q=2 at higher depth, needs joint information
about the junk pattern. Extending the existing one-scalar bound through more algebra
cannot settle it: that bound is false there. The full margin theorem remains a
separate, open question.
