# An analytic global theorem at p=3, and an all-prime one-frequency theorem

Status: PROVED by the inequalities below, without numerical optimization, interval
subdivision, or a computer-assisted inequality certificate. Exact symbolic arithmetic was
used only to check the displayed polynomial identities and the finite Fourier averaging
identities; their proofs are fully written out here.

These are restricted instances of the desired global theorem, not a proof at every prime
with multiple interacting frequencies. The architecture has n=2, nu=3 throughout, and
the margin is over the full modular-addition table and all wrong classes.

1. At p=3, the ansatz is globally margin optimal among ALL real networks of arbitrary
   finite width, including DC and mixed-frequency components, for every 2<=q<=3.
2. At every odd prime p, the three-neuron ansatz is globally optimal within the pure
   single-frequency class, at arbitrary width and with arbitrary junk, for every 2<=q<=3.

The first result replaces the interval-certified last step of
`../opus55/P3_THEOREM.md` with an analytic proof and extends its stated q=2 range to
2<=q<=3. The positive-semidefinite matrix construction is inherited from that file;
the trace-moment estimate and scalar proof below replace its case enumeration and cells.
The individual averaging identities used for p=3 are derived again here, so no claim
about completeness of a family of test measures is needed.

## 1. The population inequality

Let c_i be arbitrary complex numbers and v_i,w_i complex numbers of modulus one. Define

    G = sum_i c_i,
    J_1 = sum_i c_i v_i,
    J_2 = sum_i c_i w_i,
    J_12 = sum_i c_i v_i w_i,
    delta = max(|J_1|,|J_2|,|J_12|).

For every 2/3<=r<=1,

    sum_i |c_i|^r >= 3^(1-r) (|G|-delta/2)_+^r.                (1.1)

This holds for every finite number of atoms, with no phase alignment hypothesis.

### 1.1 A positive-semidefinite matrix

The assertion is immediate if |G|=0 or delta>=2|G|. Otherwise replace each c_i by c_i/G,
so that G=1. Write c_i=x_i exp(i alpha_i), x_i>=0, and

    e = sum_i x_i >=1,
    u = sqrt(2e(e-1)),
    m_i = (1, exp(i alpha_i)v_i, exp(i alpha_i)v_i w_i)^T,
    M = sum_i x_i m_i m_i^*.

M is positive semidefinite. Its diagonal entries are e. Its three lower-triangular
entries are J_1, J_12, and K=sum_i x_i w_i. Since sum_i c_i=1,

    |K-J_2| <= sum_i x_i |1-exp(i alpha_i)|
              <= sqrt((sum_i x_i)(sum_i x_i |1-exp(i alpha_i)|^2))
              = sqrt(2e(e-1)) = u.

Consequently

    tr M = 3e,
    tr M^2 <= B := 3e^2 + 6delta^2 + 4delta*u + 2u^2.          (1.2)

Let lambda_j be the three eigenvalues of M. For 0<r<1,

    3^r sum_i x_i^r >= sum_j lambda_j^r
                    >= (3e)^(2-r) / B^(1-r).                 (1.3)

Both inequalities have elementary proofs. For the first, form the matrix whose columns
are sqrt(x_i)m_i. Its Gram matrix has diagonal entries 3x_i and nonzero eigenvalues
lambda_j. Diagonalize the Gram matrix and apply concavity of t^r to each diagonal entry;
sum the resulting inequalities. For the second, Holder gives

    sum_j lambda_j
      = sum_j (lambda_j^r)^(1/(2-r))
              (lambda_j^2)^((1-r)/(2-r))
      <= (sum_j lambda_j^r)^(1/(2-r))
         (sum_j lambda_j^2)^((1-r)/(2-r)).

Rearrange and use (1.2).

### 1.2 The scalar inequality, proved by a positive polynomial

For every e>=1 and 0<=delta<=2, put z=1-delta/2. Then

    z^2 B <= 3e^4.                                           (1.4)

Here is a direct proof. Write f=z^2 and A=3-2f>=1. The identity

    3e^4 - 3e^2 - 3u^2 = 3e(e-1)^2(e+2) >=0

and e^2>=1 imply

    3e^4-fB
      >= 3(1-f) + A*u^2 - 4f*delta*u - 6f*delta^2
      = A*(u-2f*delta/A)^2 + P/A,

where

    P = 3(1-f)(3-2f) - delta^2*(18f-8f^2).

Set t=delta/2 in [0,1]. Direct expansion gives P=t Q(t), with

    Q(t)=6-19t-8t^2+126t^3-128t^4+32t^5
        =6(1-t)^5 + 54t^4(1-t) + 9t^5
         +t(1-t)^2*[11(1-t)^2-24t(1-t)+48t^2].

The bracket is positive, since

    11s^2-24st+48t^2
      = 11(s-12t/11)^2 + (384/11)t^2,   s=1-t.

Thus Q(t)>0 throughout [0,1]. This proves (1.4), strictly if delta>0.
If delta=0 and e>1, it is also strict: then

    3e^4-B = e(e-1)(3e^2+3e-4)>0.

### 1.3 Complete the proof for every exponent in the claimed range

Assume delta<2, so z>0. Combining (1.3) with B<=3e^4/z^2 gives

    sum_i x_i^r
      >= 3^(1-r) e^(3r-2) z^(2-2r)
      >= 3^(1-r) z^r,                    2/3<=r<1.             (1.5)

The last step uses e>=1, 0<z<=1, 3r-2>=0, and 2-2r<=r.
For r=1, sum_i x_i=e>=1>=z directly. Rescale by |G|^r to prove (1.1).

### 1.4 Equality

When |G|-delta/2>0 and 2/3<=r<1, equality in (1.1) forces delta=0 and e=1 after
normalizing G=1. The x_i-positive atoms then all have alpha_i=0, so M=I_3.
The associated Gram matrix is a rank-three orthogonal projection with diagonal 3x_i.
Thus 0<=3x_i<=1 and sum_i 3x_i=3. Equality

    sum_i (3x_i)^r = 3

forces every 3x_i to be zero or one, since t^r>t for 0<t<1. There are exactly three
nonzero atoms, each of amplitude 1/3 in this normalization. Their m_i/sqrt(3) are
orthonormal. At r=1, equality still forces delta=0 and e=1, but this argument does
not force three atoms.

## 2. Unrestricted global optimality at p=3

Let omega=exp(2*pi*i/3), and use the normalized Fourier convention

    U_i(k) = (1/3) sum_x E_i1(x) omega^(-kx),

and similarly let V_i,W_i be the coefficients of the second input slot and the readout.
Centering each readout preserves all gaps and does not
increase penalty. With real weights W_i(-1)=conj(W_i(1)), and the centered readout
has no other frequencies.

For each neuron define

    c_i = conj(U_i(1)) conj(V_i(1)) W_i(1),
    v_i = U_i(1)/conj(U_i(1)),
    w_i = V_i(1)/conj(V_i(1)).

If a denominator vanishes, c_i=0 and its phase can be chosen arbitrarily. In the full
logit tensor's Fourier expansion, G is the coefficient at (-1,-1,1); J_1,J_2,J_12
are the coefficients at (1,-1,1), (-1,1,1), and (1,1,1), respectively.

### 2.1 The cost bound includes arbitrary DC inputs

Parseval and AM-GM give, neuron by neuron,

    ||theta_i||^2
      >= 6(|U_i(1)|^2+|V_i(1)|^2+|W_i(1)|^2)
      >= 18 |c_i|^(2/3).

Therefore, for q in [2,3] and r=q/3,

    P_q >= 18^(q/2) sum_i |c_i|^r.                            (2.1)

Dropping the input DC terms in this COST bound does not claim they can be dropped
from the logits without affecting the margin.

### 2.2 Exact averaging identities eliminate DC from the margin tests

Let gap(x,D)=Z(x_1,x_2,y)-Z(x_1,x_2,y+D), y=x_1+x_2.
Every average of these gaps is at least the true margin m.

For J_1, choose D uniformly in {1,2}, fix x_1+D/2=s in Z_3, and average x_2 uniformly.
Here D/2 means multiplication by the inverse of 2 in Z_3. The result is

    E gap = 3 Re(G-J_1 omega^(2s)).                           (2.2)

To check this for an arbitrary tensor, consider its component (a,b,1). Averaging x_2
forces b=-1. Its remaining multiplier is

    omega^((a+1)s) * (1/2) sum_{D=1,2}
        (1-omega^D) omega^(-(a+1)D/2).

For a=-1,1,0, the last average is respectively 3/2,-3/2,0. The components with
readout frequency -1 are the conjugates. This proves (2.2) and eliminates the DC terms.

Interchanging the inputs proves the same identity for J_2. For J_12, impose
x_1+x_2+D/2=s and average uniformly along that line. This forces a=b, and the same
three-value calculation proves

    E gap = 3 Re(G-J_12 omega^(2s)).                           (2.3)

Some cube root zeta has Re(J_T*zeta)>=|J_T|/2. Choose T with |J_T|=delta and the
corresponding s in (2.2) or (2.3). Then

    m <= 3(Re G-delta/2) <= 3(|G|-delta/2).                    (2.4)

Thus positive margin implies |G|-delta/2>0. These identities concern the complete
logit tensor, with arbitrary input DC and arbitrary width.

### 2.3 The sharp value

Combine (1.1), (2.1), and (2.4). Every positive-margin network satisfies

    gamma_q = m/P_q^(3/q)
      <= 3^(2-3/q) / 18^(3/2).                               (2.5)

Networks with nonpositive margin satisfy the upper bound trivially. At q=2 the value is
sqrt(3)/18^(3/2). This is exactly the proposed ansatz value at p=3.

Attainment is explicit. For l=0,1,2 take

    E_l1(x)=cos(2*pi*x/3 + pi*l/3),
    E_l2(x)=cos(2*pi*x/3 + pi*l/3),
    W_l(c)=cos(2*pi*c/3 + 2*pi*l/3).

Their sum has logits (3/4)cos(2*pi*(x_1+x_2-c)/3), margin 9/8, and
P_q=3*(9/2)^(q/2), giving equality in (2.5).

For 2<=q<3, equality in the whole chain also forces no DC in any nonzero neuron,
balanced slot norms, zero junk, and exactly three nonzero neurons, by section 1.4.
At q=3 the value is still sharp, but the three-neuron uniqueness statement is not made.

In particular, a hypothetical unrestricted global optimizer beating the ansatz at p=3
contradicts (2.5). This closes that optimizer-counterexample problem analytically for
the entire allowed q interval at nu=3.

## 3. Every odd prime, pure one-frequency networks, arbitrary width

Fix any nonzero frequency k modulo an odd prime p. Suppose every nonzero neuron has all
three slots entirely in the real Fourier plane {+k,-k}; DC is absent. No restriction is
put on width, amplitudes, phases, or same-frequency junk. Use the same definitions of
c_i,v_i,w_i with normalized p-point Fourier coefficients.

Let

    h(x_1,x_2)=G+J_1 exp(4*pi*i*k*x_1/p)
                 +J_2 exp(4*pi*i*k*x_2/p)
                 +J_12 exp(4*pi*i*k*(x_1+x_2)/p).

Averaging the gaps at D and -D at the SAME input row gives exactly

    (gap(x,D)+gap(x,-D))/2
      = 2(1-cos(2*pi*k*D/p)) Re h(x).                         (3.1)

Averaging Re h over x_2 at fixed x_1 isolates J_1; the analogous operation isolates
J_2; averaging along x_1+x_2=s isolates J_12. The p roots of unity contain a point
within angle pi/p of any target direction. Consequently

    min_x Re h(x) <= Re G-cos(pi/p)*delta <= |G|-delta/2.       (3.2)

Choose D with kD=1 mod p and put A_min=1-cos(2*pi/p). Since the margin is bounded by
every average in (3.1),

    m <= 2 A_min (|G|-delta/2).

Parseval and AM-GM now give P_q>=(6p)^(q/2) sum_i |c_i|^r. Applying (1.1) proves

    gamma_q <= 3^(-(3-q)/q) * 2(6p)^(-3/2) * A_min.           (3.3)

The usual three-neuron block at frequency k attains this value. Thus arbitrary width
and arbitrary junk do not improve the pure one-frequency optimum, for any odd prime
and every 2<=q<=3. For q<3, equality again requires exactly three nonzero neurons and
zero junk.

This is an exact finite-grid theorem, with no large-p approximation or error factor.
It is a class optimum: at primes where Gamma_(3/q) has a better multifeature solution,
that multifeature ansatz exceeds the value in (3.3).

## 4. Remaining boundary

At p>=5, unrestricted tensors have cross-frequency components with nonzero frequencies
in every slot. The special p=3 averages no longer reduce the entire problem to one
population. Nor can the worst row for several pure populations be chosen independently.
The analytic population lemma does not resolve either interaction.

This proof alone does not settle unrestricted multifeature networks at general p or
depth nu>3.

Subsequent extension: `GENERAL_DEPTH.md` now proves the unrestricted p=3 theorem and
the all-prime pure one-frequency theorem at every nu>=3 for nu-1<=q<=nu, with a
slightly wider explicit range. Lower exponents at higher depth and unrestricted
multifeature networks at general p remain open.
