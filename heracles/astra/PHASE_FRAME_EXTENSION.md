# A larger unrestricted range, a p=5 theorem down to q=5/2, and the q=2 obstruction

## 1. Results and remaining scope

The full unrestricted theorem is NOT proved here. This pass establishes:

1. For every odd prime p and degree nu>=3, the dense ansatz is unrestrictedly
   globally margin optimal throughout

       nu - 4/[3(p-1)] <= q <= nu.                         (1.1)

   A sharper explicit bound is in section 5. Width, mixed frequencies, and
   DC components are unrestricted. For q<nu in the proved interval, every
   optimizer is junk-free, with nu neurons at each frequency.

2. At p=5, nu=3, the unrestricted theorem and this rigidity hold throughout

       5/2 <= q <= 3.                                     (1.2)

3. At p=5, nu=3, q=2, there is an explicit feasible margin-one logit tensor
   whose retained Fourier matrix is positive semidefinite and has Schatten
   2/3 power sum strictly less than the target 6. All its clean Fourier
   coefficients are real. This REFUTES the proposed matrix-only extension,
   not the original network conjecture. The missing tensor components are
   not free to represent. A cheap network with the same retained matrix
   fails a margin constraint by an explicit amount (section 7).

The proof is analytic. `phase_frame_verify.py` checks the algebraic identities
and the explicit obstruction exactly. No optimizer search proves these results.

## 2. Keep every frequency and separate phase loss from excess margin

Use the notation of `EXPLICIT_ENDPOINT.md`:

    n=nu-1, h=(p-1)/2, N=nu*h,
    a0=(2p)^(nu/2)/(p*nu), beta=(2p)^(-nu/2), r=q/nu,
    Z=a0 sum_i b_i T_i, F_r=sum_i b_i^r.

Each T_i is a real centered unit rank-one atom, and Z has every gap at least
one. The dense solution has F_r=N. Centering and balancing preserve all gaps
and can only reduce the original penalty, as in the previous proofs.

For each atom and positive frequency, write its clean Fourier coefficient as

    beta*rho_ik*exp(i*alpha_ik),
    rho_ik=product_{j=0}^n |sqrt(2p)*uhat_ij(k)|.

Holder and Parseval give sum_k rho_ik^(2/nu)<=1. Since q>=2, r>=2/nu, and
therefore

    sum_k rho_ik^r <=1,
    F_r >= sum_{i,k} w_ik^r,    w_ik=b_i*rho_ik.            (2.1)

This is a budget comparison in the proof, not an assertion that replacing
the network by its same-frequency components preserves its margin.

Let L be the uniform-gap endpoint functional, and define

    C=sum_ik w_ik,
    Delta=C-L(Z/a0)=sum_ik w_ik(1-cos alpha_ik),
    e=L(Z/a0)-N,
    E=C-N=Delta+e.

Both Delta and e are nonnegative. Delta measures phase loss; e measures the
average gap excess. The previous bound allowed both to be as large as E
simultaneously. Retaining the exact identity Delta+e=E improves the comparison.

## 3. Rotate the entire phase frame

Use the Hermitian Fourier chain map Q from `EXPLICIT_ENDPOINT.md`: in each
frequency block it has diagonal Re G/(nu*beta), lower entries J_interval/(nu*beta),
and upper entries the conjugates. Q(Z0/a0)=I_N.

After conjugation by the original chain phases, a unit-amplitude frequency
contribution to Q has the form H_alpha/nu, where

    (H_alpha)_aa=cos alpha,
    (H_alpha)_ab=exp(i alpha) for a>b,
    (H_alpha)_ab=exp(-i alpha) for a<b.

Take alpha in [-pi,pi] and the unit vector

    v_j=exp(2*i*j*alpha/nu)/sqrt(nu),    j=0,...,nu-1.

A finite geometric sum gives

    H_alpha*v = [sin(alpha)*cot(alpha/nu)] v.              (3.1)

For example, at row zero the sum is cos alpha plus
exp(-i alpha) sum_{j=1}^{nu-1} exp(2*i*j*alpha/nu), which simplifies to
sin(alpha)*cot(alpha/nu). Successive normalized rows differ by zero because
exp(2*i*alpha/nu)^nu=exp(2*i*alpha). Values at alpha=0 are taken by continuity.

Put P_alpha=v*v^*. Its entries are a valid chain rank-one projection: the
phase ramp amounts to adding alpha/nu to every input phase and aligning the
readout to their sum. Undoing the original diagonal conjugation therefore
gives a pure, aligned phase frame for each frequency contribution.

The Frobenius error is exactly

    d_nu(alpha)^2
      =||H_alpha/nu-P_alpha||_F^2
      =2-sin(alpha)^2/nu -[2*sin(alpha)/nu]*cot(alpha/nu).

For 0<=alpha<=pi, x*cot x decreases on (0,pi/2), since its derivative has
the sign of sin x*cos x-x<=0. Because nu>=2,

    [sin(alpha)/nu]*cot(alpha/nu) >= (1+cos alpha)/2.

The same conclusion follows for negative alpha by symmetry. Hence

    d_nu(alpha)^2 <=1-cos alpha.                          (3.2)

For every (i,k), choose this rank-one projection P_ik, and define

    X=blockdiag_k sum_i w_ik P_ik >=0,
    tr X=C.

Weighted Cauchy-Schwarz and (3.2) give

    ||X-Q(Z/a0)||_F <=sqrt(C*Delta).                      (3.3)

The matrix construction charges every frequency contribution through (2.1).
Mixed-frequency components remain in the original feasible tensor Z; they
are not declared harmless or removed from its margin constraints.

## 4. An exact norm bound for the gap-excess contribution

Let e_xc=Z_y(x)-Z_c(x)-1>=0 for wrong classes, and e_xy=0. Then Z-Z0 is -e
plus its row mean. The row mean disappears from every coefficient used by Q.

Consider one gap excess at offset D, with total contribution e' to e.
Its k-th Q block is a diagonal-unitary conjugate of

    -(2e'/nu) H_{2*pi*k*D/p}.

This follows directly from the phase D+2 sum_{j in T} x_j at each interval
coefficient. Since

    ||H_alpha||_F^2=nu*cos(alpha)^2+nu*(nu-1),
    sum_{k=1}^h cos(2*pi*k*D/p)^2=(p-2)/4,

the norm of this generator is exactly g*e', where

    g^2=4h-p/nu.                                         (4.1)

Every gap-excess tensor is a nonnegative sum of such generators. The
triangle inequality therefore gives

    ||Q(Z/a0)-I||_F <= g*e,
    D:=||X-I||_F <=sqrt(C*Delta)+g*e.                     (4.2)

Because e<=E and Delta+e=E, a second Cauchy-Schwarz bound yields

    D^2 <= (C+g^2 E)E = N E+(1+g^2)E^2.                 (4.3)

## 5. General-prime and general-depth theorem

The Gram-diagonal concavity argument from `EXPLICIT_ENDPOINT.md`, applied
to the weights w_ik, gives sum w_ik^r>=tr X^r. Holder gives

    F_r >= C^(2-r)/(tr X^2)^(1-r),
    tr X^2=N+2E+D^2.

Writing x=E/N and using (4.3),

    F_r/N >= (1+x)^(2-r) /
      [1+(2+N)x+(1+g^2)N*x^2]^(1-r).                    (5.1)

Define

    a_* = max{N+2, [1+sqrt(1+8(1+g^2)N)]/2}.

It suffices that

    r >= 1-1/(a_*-1),    q >= nu*[1-1/(a_*-1)].          (5.2)

Indeed a=(2-r)/(1-r)>=a_*, and the constant, linear, and quadratic terms
of (1+x)^a dominate the denominator polynomial in (5.1). The Taylor
remainder is strictly positive for x>0 because a>=N+2>=5.

For the simpler bound in (1.1), note a_*-1<=(3/2)N. The first branch follows
from N>=3. For the second it suffices that

    (1+3N)^2 >= 1+8(1+g^2)N.

Their difference is N[-2+9N-8g^2]. At nu=3 the bracket is (h+2)/3>0.
At nu>=4, substitute N=nu*h and g^2=4h-(2h+1)/nu to see that the bracket
is at least -2+h*(9nu-32)>0. Thus

    nu/(a_*-1) >=2/(3h)=4/[3(p-1)].

### Equality and network structure

The comparison is strict when E>0. At E=0, Delta=e=0, so all nonzero
frequency contributions are aligned, the original centered logits equal Z0,
and X=I. Each weight w_ik is at most one, since w_ik P_ik<=I. For r<1,
sum w_ik^r>=sum w_ik=N, with equality only at weights zero or one.

The proved intervals have q>2. Equality in (2.1) then requires every original
nonzero neuron to have exactly one rho_ik=1 and all other rho_ik=0: for
r>2/nu, any rho strictly between zero and one makes the budget inequality
strict. Thus no hidden mixed-frequency or DC neurons survive at equality.
Every frequency has nu unit weights and an orthonormal chain frame.

The endpoint q=nu has the larger equality class described in the earlier
endpoint proof; the value assertion still holds there.

## 6. At p=5, nu=3, retain higher-order phase information

Here N=6 and g^2=19/3. We prove F_r>=6 for every 5/6<=r<1, giving (1.2).

### 6.1 The cubic phase formula

Put d=1-cos alpha and u=1-cos(alpha/3), so 0<=u<=1/2. The exact formula
in section 3 simplifies to

    d_3(alpha)^2 = d^2/3 +4u/3,
    d=u*(3-2u)^2.

The following bound retains the small-phase coefficient:

    u <= d/9 +5*d^2/72,
    d_3(alpha)^2 <=(4/27)d+(23/54)d^2
                   <=(4/27)d+(4/9)d^2.                  (6.1)

For the first inequality, cancel u^2 when u>0; the required inequality is
5(3-2u)^4>=32(3-u). Its difference factors as

    (1-2u)(309-430u+220u^2-40u^3)>=0,

since the second factor is at least 309-215-5=89 on [0,1/2]. The u=0 case
is equality. Consequently, after summing contributions,

    ||X-Q(Z/a0)||_F <=sqrt((4/27)C*Delta)+(2/3)Delta.

Use g<=17/6 (a deliberately loose rational bound). With e=E-Delta,

    D <=(17/6)E+sqrt((4/27)C*Delta)-(13/6)Delta.           (6.2)

Maximize this elementary expression over 0<=Delta<=E. Its unconstrained
maximum occurs at Delta/C=4/507. With x=E/6 and C=6(1+x), there are two cases.

### 6.2 Small excess: x<=4/503

The right side of (6.2) is maximized at Delta=E. Thus

    D <=sqrt((4/27)C E)+(2/3)E,
    D^2/6 <=(16/9)x+(64/9)x^2,
    tr X^2/6 <=1+(34/9)x+(64/9)x^2 <=(1+x)^7.

The middle bound uses (a+b)^2<=2a^2+2b^2. The last inequality follows
coefficient by coefficient from the binomial expansion.

### 6.3 Larger excess: x>=4/503

Evaluating the unconstrained maximum in (6.2) gives

    D <=(667x+4)/39,
    tr X^2/6 <=1+2x+(667x+4)^2/9126.

We now prove this is strictly less than (1+x)^7 on the stated interval.
Multiplying the difference by 9126 gives

    P(x)=9126x^7+63882x^6+191646x^5+319410x^4
           +319410x^3-253243x^2+40294x-16.

All terms above degree four are nonnegative. Write its degree-four truncation
as x*R(x)-16, where

    R(x)=40294-253243x+319410x^2+319410x^3.

An exact positive decomposition is

    R(x)=319410*x*(x-7/25)^2
          +(2491398/5)*(x-7/25)^2
          +(93979/125)*x+3840248/3125.

Thus R(x)>1200 for x>=0. For x>=1/50, P(x)>1200x-16>=8.
For 4/503<=x<=1/50, discard the positive cubic and higher terms and use

    P(x)>=-16+x*(40294-253243/50)
          >-16+35000x>=-16+140000/503>0.

This proves tr X^2/6<=(1+x)^7 for all x>=0, strictly when x>0.
Since (2-r)/(1-r)>=7 for r>=5/6, the Holder comparison proves F_r>=6.
The equality argument in section 5 applies unchanged. This is a complete
unrestricted analytic theorem at p=5 for 5/2<=q<=3.

## 7. Exact obstruction to the matrix-only q=2 extension

This section is NOT a counterexample to the network theorem.

At p=5, nu=3, let Z0(x,c)=1[c=x1+x2]-1/5. Define nonnegative gap excesses
of size 5*sqrt(5) at exactly two row/class pairs:

    (x1,x2,c)=(3,3,2),    (2,2,3).

Set every other excess to zero, including the correct classes, and define

    Z(x,c)=Z0(x,c)-e_xc+(1/5)sum_d e_xd.

This is centered and feasible: each gap is exactly 1+e_xc. Reflection sends
the two modified pairs to each other, so Z(-x,-c)=Z(x,c); ALL its Fourier
coefficients, including the clean ones, are real.

Put

    c=(sqrt(5)-1)/4,    d=-(sqrt(5)+1)/4,    s=2/sqrt(5).

The two Q blocks are

    Q_1(Z/a0)=I-s*H_1,    H_1=(c-d)I+d*J,
    Q_2(Z/a0)=I-s*H_2,    H_2=(d-c)I+c*J,

where J is the 3-by-3 all-ones matrix. To check the phases: the first excess
has offset 1 and input coordinates (3,3). Its singleton interval phases
are 2*3+1=2 mod 5, and its full interval phase is 2*6+1=3 mod 5. Their
cosines agree. The reflected excess conjugates every phase. Clean phases
are +/-1, giving precisely the displayed diagonal/off-diagonal values.

Define u=(1+3/sqrt(5))/2. The six eigenvalues of Q(Z/a0) are

    0, 0, 1+u, 2, 2, u.

They are nonnegative, yet

    tr Q(Z/a0)^(2/3)=(1+u)^(2/3)+u^(2/3)+2*2^(2/3)<299/50<6.  (7.1)

This last inequality needs no numerical eigenvalue calculation. First
u<1171/1000, by squaring 3/sqrt(5)<671/500. Then the rational comparisons

    (2171/1000)^2 < (42/25)^3,
    (1171/1000)^2 < (28/25)^3,
    4 < (159/100)^3

bound the three types of terms in (7.1); their upper bounds sum to 299/50.

### Why this is not a counterexample network

The retained matrix admits a pure aligned four-neuron realization: at
frequency 1 take weight 1+u and zero input phases; at frequency 2 take
the standard three phase atoms with weights u,2,2. Its matrix is Q(Z/a0)
and its coefficient cost is the left side of (7.1).

But its actual logit tensor Y is DIFFERENT from Z. Writing theta=2*pi/5,

    Y(x,c)=(8/15)[(1+u)cos(theta*x1)cos(theta*x2)cos(theta*c)
                  +(u-2)cos(2theta*x1)cos(2theta*x2)cos(2theta*c)
                  +(3/2)cos(2theta*(x1+x2-c))].

At row (1,4), correct class 0 and wrong class 2, its gap is

    Y((1,4),0)-Y((1,4),2)=1-1/sqrt(5)<1.                 (7.2)

Thus the inexpensive matrix realization is not feasible. The feasible Z has
additional Fourier components that Q discards; representing those components
must also be paid for. Neither (7.1) nor (7.2) asserts that Z has a cheap full
network representation, or that Y beats the ansatz after margin normalization.

The specific statement refuted is: every feasible Z with Q(Z/a0)>=0 has
tr Q(Z/a0)^(2/3)>=6. Therefore even perfect eigenvalue analysis of this
retained matrix, with positive semidefiniteness and real clean coefficients,
cannot alone prove the full q=2 theorem.

## 8. What a full proof still has to add

The improved comparison pays for all frequency contributions and all phase
errors. It proves (1.1) and (1.2), but it does not yet control the cost of
repairing the omitted tensor components sharply enough at lower q.

Section 7 identifies that missing comparison concretely: a matrix can look
cheaper than the dense optimum while its cheap realization has a bad row.
A successful extension must relate the required correction of those rows
to the representation cost of the additional components, or use a different
global replacement argument. Simply tightening the spectral moment bound
cannot remove this obstruction. The full range 2<=q<=nu remains open.
