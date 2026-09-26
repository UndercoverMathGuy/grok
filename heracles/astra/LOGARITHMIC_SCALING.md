# Logarithmic feature counts and unrestricted margin scaling

2026-09-25. New analytic deductions from the existing two-sided product-network
bound. All logarithms are natural. The modulus p is an odd prime; the proofs
for the allocation programme also work for odd composite moduli.

## 0. Results and scope

Fix s>1, independently of p. For the free-amplitude Fourier programme,

\[
 \boxed{\Gamma_s(p)=\Theta_s((\log p)^{1-s}).}                 \tag{V}
\]

Every maximizing allocation, including all maximizers at tied exponents, has

\[
 \boxed{K^*(s,p)=\Theta_s(\log p).}                          \tag{K}
\]

For fixed product-model degree nu and penalty exponent 2<=b<nu, put s=nu/b.
Let gamma* be the unrestricted normalized-margin supremum over all finite
widths on the FULL modular-addition table. The existing upper bound and
interference-free attaining construction then imply

\[
 \boxed{\gamma_b^*(p)
 =\Theta_{\nu,b}\!\left(p^{-\nu/2}(\log p)^{1-\nu/b}\right).} \tag{G}
\]

**(G) is unrestricted.** The unknown interference advantage only changes
the constants at fixed nu,b. It does not change the powers of p and log p.
The same order holds at any prescribed width at least
nu*ceil(8 log p); a probabilistic construction below suffices at that width.

For a target full-table margin h_0>0, the minimum penalty consequently obeys

\[
 \boxed{\inf_{h(\theta)\ge h_0}P_b(\theta)
 =\Theta_{\nu,b}\!\left(h_0^{b/\nu}p^{b/2}
                         (\log p)^{1-b/\nu}\right).}        \tag{P}
\]

These are order bounds with constants depending on the fixed exponents,
not asymptotic equivalences with a known leading constant. They are not
uniform as b approaches nu. At b=nu the allocation optimizer is dense:
K=(p-1)/2 and Gamma_1(p)=p/(p-1). Nor do these results prove that training
finds a maximum-margin network, or that an unrestricted optimizer has only
O(log p) frequencies. Its clean active frequency count must be Omega(log p),
as shown in section 7; the matching upper count is proved for the programme.

## 1. Programme and explicit constants

Let F={1,...,(p-1)/2}, r=1/s, and

\[
 F_y(\delta)=\sum_{j\in F}y_j(1-\cos(2\pi j\delta/p)),\quad
 m(y)=\min_{\delta\ne0}F_y(\delta),\quad
 \Gamma_s(p)=\max_{y\ge0,\;\sum_j y_j^r\le1}m(y).
\]

Define

\[
 a_s=3+s/2,\quad c_s=\tfrac12 9^{1-s},\quad
 C_s=\tfrac32(2a_s)^s.
\]

We prove the explicit bounds

\[
 \tfrac12\lceil8\log p\rceil^{1-s}
 \le\Gamma_s(p)\le C_s(\log p)^{1-s},                       \tag{1}
\]

and therefore c_s(log p)^(1-s)<=Gamma_s(p). Constants have been chosen
for transparent proofs rather than sharp finite-p predictions.

## 2. Lower bound: logarithmically many random frequencies

Take L=ceil(8 log p) independent frequencies J_1,...,J_L, uniformly from F,
allowing repetitions. For each nonzero delta,

\[
 \mathbb E\cos(2\pi J_\ell\delta/p)=-1/(p-1).
\]

Hoeffding's inequality for variables in [-1,1] gives

\[
 \Pr\left[\frac1L\sum_\ell\cos(2\pi J_\ell\delta/p)>\frac12\right]
 \le e^{-L/8}.
\]

A union bound over p-1 nonzero offsets is at most (p-1)/p<1. Hence some
choice has sum_l(1-cos(2 pi J_l delta/p))>=L/2 for EVERY nonzero delta.

If frequency j occurs n_j times, set y_j=n_j L^(-s). Since r=1/s<1,

\[
 \sum_j y_j^r=L^{-1}\sum_j n_j^r\le L^{-1}\sum_j n_j=1.
\]

Thus this allocation is feasible, supported on at most L frequencies, and
has margin at least L^(1-s)/2. Repetitions do not cause a budget problem;
merging them only reduces the concave amplitude cost. Finally L<=9 log p
for p>=3, proving the simpler lower bound with c_s.

## 3. The positive-Fourier exponential bound

Define the function

\[
 I_0(z)=\frac1\pi\int_0^\pi e^{z\cos\theta}\,d\theta,\qquad
 \phi(z)=z-\log I_0(z),\quad z\ge0.
\]

The Fourier coefficients of exp(z cos theta) are nonnegative: expand
exp((z/2)e^(i theta)) exp((z/2)e^(-i theta)) into its absolutely convergent
power series. Its constant coefficient is I_0(z). Therefore averaging a
product of these functions over delta in Z_p retains all terms whose
combined frequency is zero modulo p, all nonnegative; in particular it
retains the product of the individual constant coefficients. Consequently,

\[
 \frac1p\sum_{\delta\in\mathbb Z_p}e^{-tF_y(\delta)}
 \ge \prod_j e^{-ty_j}I_0(ty_j)
 =\exp\!\left[-\sum_j\phi(ty_j)\right].                    \tag{2}
\]

**No independence of the phases j*delta is assumed.** Additional resonances
only add nonnegative terms, which is the direction needed in (2).

We next prove, for all z>=0,

\[
 0\le\phi(z)\le a_s\log(1+z^r).                            \tag{3}
\]

For 0<=z<=1, I_0(z)>=1, so phi(z)<=z. Since
log(1+z^r)>=z^r/2>=z/2 and a_s>=2, (3) follows.
For z>=1, use cos(theta)>=1-theta^2/2 and integrate only over
0<=theta<=1/sqrt(z):

\[
 I_0(z)\ge \frac{e^{z-1/2}}{\pi\sqrt z},\qquad
 \phi(z)\le\log\pi+\tfrac12+\tfrac12\log z.
\]

But 3 log(1+z^r)>=3 log 2>log pi+1/2, and
(1/(2r)) log(1+z^r)>= (1/2)log z. Adding proves (3), since
a_s=3+1/(2r). Nonnegativity follows from I_0(z)<=e^z.

If y has K nonzero entries and sum y_j^r<=1, concavity of log gives

\[
 \sum_j\phi(ty_j)
 \le a_s\sum_j\log(1+t^r y_j^r)
 \le a_s K\log(1+t^r/K)\le a_s t^r.                        \tag{4}
\]

## 4. Upper bounds, with and without a support limit

Write m=m(y). Because F_y(0)=0 and F_y(delta)>=m for delta!=0,

\[
 \frac1p\sum_\delta e^{-tF_y(\delta)}
 \le p^{-1}+e^{-tm}.                                       \tag{5}
\]

First choose t=(log p/(2a_s))^s. Equations (2), (4), and (5) imply
e^(-tm)>=p^(-1/2)-p^(-1)>=p^(-1/2)/3 for p>=3. Hence

\[
 m\le\frac{\frac12\log p+\log3}{t}
 \le\tfrac32(2a_s)^s(\log p)^{1-s},
\]

proving the upper half of (1), and therefore (V).

Retaining the K-dependent bound in (4), instead choose

\[
 t=\left[K\left(p^{1/(2a_sK)}-1\right)\right]^s.
\]

The same calculation proves, for EVERY feasible K-frequency allocation,

\[
 \boxed{m(y)\le
 \frac{\frac32\log p}
      {[K(p^{1/(2a_sK)}-1)]^s}.}                            \tag{6}
\]

For x=log p/K this becomes

\[
 \frac{m(y)}{(\log p)^{1-s}}
 \le\frac32\left[\frac{x}{e^{x/(2a_s)}-1}\right]^s.         \tag{7}
\]

In particular, K=o(log p) forces m(y)/Gamma_s(p)->0. This is stronger
than merely ruling out a bounded number of features at the optimum.

## 5. Lower support bound for every programme optimizer

For an optimizer, m>=c_s(log p)^(1-s). Combining with (7) and
e^z-1>=z^2/2 yields

\[
 c_s\le\tfrac32(8a_s^2/x)^s,
 \qquad K\ge
 \frac{\log p}{8a_s^2(3/(2c_s))^{1/s}}.                    \tag{8}
\]

Thus every optimizer has Omega_s(log p) active frequencies. Equation (7)
gives a substantially better implicit constant than the loose explicit (8).

## 6. Upper support bound: delete and rescale

At an optimizer the budget is saturated, sum y_j^r=1. Let m=Gamma_s(p)>0.
If K>=2, delete any active coordinate y_j. Every offset coverage decreases
by at most 2y_j. Rescale the remaining amplitudes by (1-y_j^r)^(-s), which
restores budget one. Optimality implies

\[
 m\ge\frac{m-2y_j}{(1-y_j^r)^s},\qquad
 2y_j\ge m[1-(1-y_j^r)^s]\ge m y_j^r.
\]

Therefore each active frequency has

\[
 y_j^r\ge(m/2)^{1/(s-1)},\qquad
 K\le(2/m)^{1/(s-1)}
 \le9\,4^{1/(s-1)}\log p.                                \tag{9}
\]

For K=1 the same final bound holds (m<=2). The argument does not require
uniqueness, equal amplitudes, or a differentiable optimum. Equations (8)
and (9) prove (K). It also proves a lower bound of constant/log p on each
active feature's share y_j^r at an exact programme optimum.

## 7. Consequences for unrestricted product networks

The established network bounds, with B_p=2(2p nu)^(-nu/2), are

\[
 B_p\nu^{1-s}\Gamma_s(p)
 \le\gamma_b^*(p)\le B_p\Gamma_s(p).                       \tag{10}
\]

At fixed nu,b their ratio nu^(s-1) is independent of p. Inserting (V)
proves (G). The lower construction in section 2 uses at most
nu*ceil(8 log p) neurons. Inserting its explicit lower bound instead of
the exact programme optimum proves the same scaling at every width at
least that large. The unit-margin penalty is gamma*^(-1/s), by positive
homogeneity, which proves (P).

There is also a support consequence outside the interference-free class.
For an arbitrary product network with clean amplitudes A_j and P_b>0,
define y_j=A_j/(B_p P_b^s). The universal single-neuron budget implies
sum y_j^r<=1, and its actual normalized margin satisfies

\[
 \gamma_b(\theta)/B_p\le m(y).
\]

Thus (6) applies as an upper bound to every unrestricted network with K
active clean frequencies. Any sequence of networks attaining a fixed
positive fraction of the unrestricted optimum must have
K=Omega_{nu,b,epsilon}(log p). Networks with K=o(log p) attain a vanishing
fraction of the optimal normalized margin, even if they contain arbitrary
other Fourier components. This counts nonzero CLEAN amplitudes, not neuron
purity or a thresholded empirical frequency count.

Conversely, an interference-free programme optimizer uses O_s(log p)
frequencies and is at least nu^(1-s) times the unrestricted optimal margin.
It is an explicit constant-factor approximation with logarithmic support.
This does not prove an O(log p) support upper bound for every unrestricted
optimizer: tiny additional frequencies remain a distinct issue.

## 8. Concrete predictions and limits

At ordinary weight decay b=2, the unrestricted optimal normalized margin
and the minimum unit-margin squared norm have these orders:

| Degree nu | gamma*_2(p) | Minimum P_2 at margin one |
|---|---|---|
| 3 | p^(-3/2) (log p)^(-1/2) | p (log p)^(1/3) |
| 4 | p^(-2) (log p)^(-1) | p (log p)^(1/2) |
| 5 | p^(-5/2) (log p)^(-3/2) | p (log p)^(3/5) |

Each column is a Theta statement, with degree-dependent constants.
The testable order predictions are that gamma*p^(nu/2)*(log p)^(s-1),
minimum P_b/[p^(b/2)*(log p)^(1-b/nu)], and programme K*/log p stay bounded
above and away from zero as p increases at fixed exponents. The theorem
does not say these quantities converge to a constant, predict their finite-p
slopes, or establish that a trained network realizes the optimal orders.

There is a qualitative endpoint distinction: any fixed b<nu has logarithmic
programme support as p grows, whereas b=nu has linear support. There is no
claim of a discontinuity at fixed p: the constants deteriorate near the
endpoint, consistently with the known finite-p dense plateau.

## 9. Relation to existing work and verification

The random-frequency lower construction is the cyclic Fourier version of
the established random-Cayley-expander mechanism; see Alon and Roichman,
*Random Cayley Graphs and Expanders* (1994),
https://web.math.princeton.edu/~nalon/PDFS/exp1.pdf.
No novelty claim is made for that ingredient. The weighted concave-budget
upper bound, the deletion argument, and their consequences for this network
programme are derived explicitly above; a full literature-priority review
remains separate from checking these proofs.

`logarithmic_scaling_verify.py` checks the scalar bound, the finite-group
positive-Fourier inequality, both margin upper bounds, the random lower
construction, and the deletion condition on previously certified exact
small-prime optimizers. Numerical checks are supplementary consistency
checks; the all-p statements are proved by the analytic inequalities above.
