# An explicit unrestricted margin theorem

Follow-up: `PHASE_FRAME_EXTENSION.md` widens this interval, with the simpler
sufficient strip nu-4/[3(p-1)]<=q<=nu, and proves the p=5,nu=3 case down to
q=5/2. It also identifies an exact obstruction to the matrix-only q=2 extension.
The earlier proof and constants below are retained for reference.

This strengthens `ENDPOINT_NEIGHBORHOOD.md`: the interval is now explicit,
and the proof no longer needs entropy selection, compactness, a width bound,
or a local-neighborhood argument. Every inequality below applies directly
to the original arbitrary-width competitor.

## 1. The theorem

Fix an odd prime p, degree nu>=3, n=nu-1, and h=(p-1)/2. Define

    A_nu = 2*nu,
    c_nu = nu + sqrt(nu^2+8*nu),
    r_0 = c_nu*h/(1+c_nu*h),    q_0=nu*r_0.                (1.1)

For EVERY q in [q_0,nu], the dense Fourier ansatz is globally normalized-margin
optimal among all real gated-chain networks of arbitrary finite width, without
any assumption of purity or junk cancellation. The sharp value is

    gamma_q = p*nu / [(2p*nu)^(nu/2) (nu*h)^(nu/q)].        (1.2)

For q<nu in this interval, every optimizer, after scaling to margin one and
discarding zero neurons, has exactly nu*h neurons, with nu at each frequency.
Each is balanced, centered, pure, and clean-phase aligned. Its full centered
output is Z0(x,c)=1[c=sum x]-1/p, and all its junk cancels.

The interval has width nu/(1+c_nu*h), an explicit order-1/p lower bound
for fixed nu. These constants are sufficient, not claimed optimal.
They do not settle ordinary q=2 at general p or the sparse-feature regime.

A simpler consequence is UNIFORM IN DEPTH:

    nu - 6/(9p-7) <= q <= nu,    every nu>=3.              (1.3)

Indeed c_nu<=3*nu for nu>=3, and

    nu/(1+c_nu*h) >= nu/(1+3*nu*h)
                   >= 3/(1+9*h) = 6/(9p-7).

Thus (1.3) is an explicit sufficient range independent of depth except
for the location of its endpoint. Formula (1.1) gives a sharper range.

For nu=3, c_3=3+sqrt(33)<9. Thus the simpler rational sufficient range is

    q >= 27*h/(1+9*h),    q<=3.

Examples of this rational bound:

| p | proved sufficient q |
|---|---|
| 5 | q>=54/19 = 2.84210526... |
| 7 | q>=81/28 = 2.89285714... |
| 11 | q>=135/46 = 2.93478261... |
| 59 | q>=783/262 = 2.98854962... |

At p=3, the stronger previously proved ranges in `P3_ANALYTIC.md` and
`GENERAL_DEPTH.md` remain preferable.

## 2. Normalize the competitor and retain its endpoint excess

Use centered real unit rank-one atoms T_i as in `ENDPOINT_NEIGHBORHOOD.md`:

    Z=a0 sum_i b_i T_i,    b_i>=0,
    a0=(2p)^(nu/2)/(p*nu),    N0=nu*h,
    F_r=sum_i b_i^r,    B=sum_i b_i,    E=B-N0.

The dense construction has N0 coefficients equal to one, so F_r=N0.
Centering and balancing can only reduce the network penalty, and the
margin-one problem is equivalent to minimizing F_r.

Let beta=(2p)^(-nu/2), and let G_k be the normalized Fourier coefficient
at (-k,...,-k,k). The endpoint functional is

    L(X)=beta^(-1) Re sum_{k=1}^h G_k(X).

For every unit atom L(T)<=1. Averaging all row/class gaps gives
L(Z)=a0*N0*gbar(Z). Hence a feasible margin-one network satisfies

    E>=0,
    sum_i b_i [1-L(T_i)]<=E.                              (2.1)

These facts, including all normalizations and the equality cases, are
proved in section 2 of `ENDPOINT_NEIGHBORHOOD.md`.

## 3. A linear map from tensors to chain matrices

For each positive frequency k, define a Hermitian nu-by-nu matrix Q_k(X):

* Every diagonal entry is Re G_k(X)/(nu*beta).
* For row a>column b, indexed 0,...,n, the entry is J_{k,{b+1,...,a}}(X)/(nu*beta).
* Upper entries are the conjugates of the lower entries.

Here J is the normalized same-frequency Fourier coefficient obtained by
flipping the listed input signs from -k to +k, keeping readout frequency k.
This map is linear over the reals. Set Q(X)=blockdiag_k Q_k(X), with N0 rows.
For X0=Z0/a0, Q(X0)=I_N0.

For an exposed atom V at frequency k, with chain vector m, Q(V) has just
one nonzero block, m*m^*/nu. It is a rank-one orthogonal projection, so

    tr Q(V)=1,    ||Q(V)||_F=1.

Project every original atom to such an atom V_i as specified in section 4,
preserving its b_i, and put U=sum_i b_i V_i. Consequently

    X := Q(U) = blockdiag_k [(1/nu) sum_{i at k} b_i m_i m_i^*] >=0,
    tr X=B.                                               (3.1)

where m_i are the chain vectors with ||m_i||^2=nu. This is the only point
where positive semidefiniteness is used; Q of a general competitor need NOT
be positive semidefinite.

## 4. Direct matrix projection costs at most twice the exposure deficit

For every unit atom T there is an exposed atom V such that

    ||Q(T)-Q(V)||_F^2 <=2[1-L(T)].                         (4.1)

The constant is independent of both p and nu. It avoids the exponential
loss from estimating a full tensor distance and then applying a matrix map.

Write z_jk=sqrt(2p)*uhat_j(k), rho_k=product_j |z_jk|, and let alpha_k
be the clean phase, so L(T)=sum_k rho_k*cos(alpha_k). Holder and Parseval give

    sum_k rho_k^(2/nu)<=1.

Select a frequency k_* maximizing rho_k. At this frequency retain the input
phases and align the readout phase with their sum to define V. If all rho_k
vanish, choose any exposed V; then Q(T)=0 and (4.1) is immediate.

For each k, the diagonal entries of Q_k(T) equal rho_k*cos(alpha_k)/nu,
and all its off-diagonal entries have modulus rho_k/nu. Therefore

    ||Q(T)||_F^2=sum_k rho_k^2 [1-sin(alpha_k)^2/nu]
                 <=sum_k rho_k^2,
    <Q(T),Q(V)>_F=rho_{k_*} cos(alpha_{k_*}).

The last identity follows by cancelling the common chain phases in every
entry of the Frobenius inner product. Its value is real. Hence

    ||Q(T)-Q(V)||_F^2 <=1+sum_k rho_k^2-2*rho_{k_*}cos(alpha_{k_*}).

The scalar fact needed to finish is

    sum_k rho_k^2 +2 sum_{k!=k_*} rho_k <=1.               (4.2)

To prove it, let y_k=rho_k^(2/nu) and m=max y_k. Then sum y_k<=1. Since nu>=3,

    sum y_k^nu +2 sum_{k!=k_*} y_k^(nu/2)
      <=sum y_k^3 +2 sum_{k!=k_*} y_k^(3/2)
      <=m^2 +2*sqrt(m)*(1-m)
      =1-(1-sqrt(m))^3*(1+sqrt(m)) <=1.

Using (4.2) and cos(alpha_k)<=1 proves (4.1) for all phases and deficits.
No feasibility assertion is made about replacing T by V.

Project all neurons using this rule. Weighted Cauchy-Schwarz and (2.1) give

    ||Q(U)-Q(Z/a0)||_F^2
      <=B sum_i b_i ||Q(V_i)-Q(T_i)||_F^2 <=2*B*E.         (4.3)

The original network still has cost F_r; its original margins are used below.

### The margin constraints give a second, sharper residual estimate

For each wrong class define its nonnegative gap excess

    e_xc=Z_y(x)-Z_c(x)-1,

and set e_xy=0. The centered difference Z-Z0 is -e plus the mean of e in
each row. The row mean has no nonzero-readout Fourier coefficients. Therefore
every coefficient used by Q has modulus at most

    p^(-nu) sum_xc e_xc.

The gap average and (2.1) give sum e<=p^(nu-1)(p-1)E/N0. Using
a0*beta=1/(p*nu), every entry of Q(Z/a0-X0) has modulus at most 2E/nu.
There are h*nu^2 entries, so

    ||Q(Z/a0-X0)||_F <=2*sqrt(h)*E.                        (4.4)

Combining the last two estimates, with no locality restriction, gives

    D := ||X-I||_F <= sqrt(2*B*E)+2*sqrt(h)*E.             (4.5)

## 5. The original cost dominates a spectral moment

For each frequency form the matrix with columns sqrt(b_i/nu)*m_i. Its
Gram matrix has diagonal entries b_i and has the same nonzero eigenvalues
as X_k. Diagonalize the Gram matrix. Since t^r is concave for 0<r<1,
Jensen applied to each diagonal entry, and then summed, gives

    F_r=sum_i b_i^r >= tr X^r.                            (5.1)

This also holds at r=1. It does not require that a frequency have exactly
nu atoms; zero columns, fewer atoms, and arbitrarily many atoms are allowed.

If lambda_j are the nonnegative eigenvalues of X, Holder gives

    sum lambda_j
      <= (sum lambda_j^r)^(1/(2-r))
         (sum lambda_j^2)^((1-r)/(2-r)).

Thus, for 0<r<1,

    F_r >= B^(2-r)/(tr X^2)^(1-r),
    tr X^2=N0+2E+D^2.                                    (5.2)

All quantities in (5.2) concern the original coefficients or an explicitly
constructed matrix. No optimization over projected networks is used.

## 6. Optimize one scalar constant, then close the theorem

Let eta>0 be the positive solution to

    A_nu^2 * eta*(1+eta)=8*nu,

and put c=A_nu*(1+eta). Equivalently c=c_nu in (1.1), and

    c^2/2=4*nu*(1+1/eta).

Weighted Young applied to (4.5) yields

    D^2 <= 2*(1+eta)*B*E +4*(1+1/eta)*h*E^2.

Write x=E/N0>=0. Since N0=nu*h, the last inequality is

    D^2/N0 <= c*h*x + [c*h+(c*h)^2/2]*x^2.

Substituting into (5.2) proves the following GLOBAL comparison, valid for
every feasible network and every 0<r<1:

    F_r/N0 >= (1+x)^(2-r) /
       [1+(2+c*h)*x + (c*h+(c*h)^2/2)*x^2]^(1-r).          (6.1)

Now assume r>=c*h/(1+c*h). Then

    a=(2-r)/(1-r) >=2+c*h >=2.

For x>=0, Taylor's theorem with nonnegative third derivative gives

    (1+x)^a >= 1+a*x+a*(a-1)*x^2/2.

The linear coefficient is at least 2+c*h. The quadratic coefficient exceeds
the one in the denominator of (6.1), because

    (2+c*h)(1+c*h)/2 - [c*h+(c*h)^2/2] =1+c*h/2 >0.

Therefore (6.1) implies F_r>=N0, strictly if E>0. This proves the unrestricted
value bound throughout the explicit interval, including its lower endpoint.

If equality holds, E=0. Then every positive atom has L(T_i)=1, and all
nonnegative gap excesses vanish. Equality in the atom bound forces purity
and phase alignment: sum rho_k^(2/nu)<=1 and Re sum rho_k*exp(i*alpha_k)=1
require exactly one rho_k=1, all others zero, and alpha_k=0 modulo 2*pi.
Thus Z=Z0 and Q(Z/a0)=I is a sum of the positive rank-one matrices b_i Q(T_i).
Each Q(T_i) has eigenvalue one, so b_i<=1. Since r<1,
b_i^r>=b_i with equality only at b_i=0 or 1. Thus exactly N0 coefficients
equal one. Each frequency has total coefficient nu and therefore nu atoms;
their chain vectors are orthogonal and all junk cancels. At r=1 the larger
endpoint equality class is retained.

## 7. What this adds, and what it does not

This is a direct global inequality and an explicit interval for every p,nu.
It removes all compactness and unspecified-neighborhood constants from the
previous theorem. Its rigidity conclusion concerns unrestricted optimizers.

The proof's losses are visible: the direct matrix projection bound, the gap
residual estimate, the use of only the first two spectral moments, and weighted Young.
Improving any of these could widen the certified interval. None currently
justifies extrapolation to q=2 or to the first sparse-feature transition.

`explicit_endpoint_verify.py` verifies the algebra, Fourier residual map,
and representative full finite tensors exactly. It performs no optimizer
search, and the all-parameter result is the analytic proof above.
