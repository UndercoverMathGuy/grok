# Review of the pasted paper appendix

Reviewed the attachment supplied on 2026-09-25, the corresponding paper
statements in `paper/iclr2027/sections/cyclic.tex`, and the local Lean statement
map, status, selected theorem declarations, and saved axiom-audit outputs.
I did not rebuild Lean or modify the manuscript/formalization.

## Assessment

The main arguments are mathematically clean. The feature-cost proof and the
positive-Fourier scaling bound have clear central ideas; neither relies on
an unexplained numerical certificate, case explosion, or an unnecessarily
powerful external theorem. The vertex, bottleneck, and ablation arguments
are short and natural. The main presentation issue is fragmentation into
many small named results and bullet lists.

The auxiliary remarks need corrections. In particular, the statement that
the whole pasted text is verified is broader than the local Lean project's
own status: it explicitly refutes the ReLU bound and excludes several remarks
and the global-conjecture discussion from its verification scope.

## A. Corrections and clarifications

### A1. ReLU normalization: an actual incorrect bound

The `app:models` ReLU paragraph claims

    sum_j |hhat(j,...,j) what(j)| <= (p/4) ||omega||^2,
    C_2 <= p^(-n)/2.

With the stated unnormalized Fourier transform, the proposed argument gives

    sum_j |hhat(j,...,j)|^2 <= (p^(2n-1)/2) sum_t ||u_t||^2,
    sum_j |what(j)|^2 <= (p/2) ||w||^2.

The first line follows by pairing positive and negative diagonal modes,
using Parseval on p^n inputs, and then

    Var(ReLU(sum_t u_t)) <= Var(sum_t u_t)
                          <= p^(-1) sum_t ||u_t||^2.

Consequently Cauchy--Schwarz and AM--GM give p^n/4 rather than p/4, and
the resulting valid budget is C_2 <= 1/(2p).

This discrepancy is already recorded in `../lean/STATUS.md` and formally
refuted in `../lean/Heracles/ReluCounter.lean`, theorem
`Heracles.relu_budget_claim_false`. The saved axiom report includes that
counterexample with the standard Lean axioms.

A separate simple exact witness at p=5,n=2 is

    u_1=u_2=(1,1,-1,-1,-1),   w(c)=2 cos(2 pi c/5).

Then h(a,b)=2 1_{0,1}(a)1_{0,1}(b), ||omega||^2=20, and only frequency
j=1 contributes to beta. Fourier orthogonality gives

    sum_j |beta_j| / ||omega||^2 = (3+sqrt(5))/250 > 1/50.

The corrected upper bound is not claimed sharp or formally verified here.
If ReLU is peripheral to the paper, removing this paragraph is also reasonable.
The main product-network theorems do not depend on it.

### A2. Three tied power neurons are interference-free under the stated definition

The concluding claims "the smallest known interference-free population ...
uses 6 neurons" and "with tied embeddings ... no interference-free population
exists" conflict with the actual definition, which permits zero-slot modes.

For n=2, let theta_x=2 pi x/p, alpha_l=pi l/3, l=0,1,2, and take

    u_l(x)=cos(theta_x+alpha_l),
    w_l(c)=cos(theta_c+2 alpha_l),
    z(a,b;c)=sum_l (u_l(a)+u_l(b))^2 w_l(c).

Elementary cosine expansion gives exactly

    z = (3/2) cos(theta_a+theta_b-theta_c)
      + (3/4) cos(2 theta_a-theta_c)
      + (3/4) cos(2 theta_b-theta_c).

The last two terms each have a zero operand Fourier slot. Therefore they are
allowed by Definition `def:if`, and every prohibited all-nonzero sign-flipped
mode vanishes. This is a three-neuron tied-embedding interference-free network.
It is also nontrivial and has positive full-table margin: the same formula is

    (3/2)[1+cos(theta_a-theta_b)] cos(theta_a+theta_b-theta_c),

whose prefactor is positive for odd p. Its margin is

    (3/2)(1-cos(pi/p))(1-cos(2 pi/p)) > 0.

It does not attain the proposed class ceiling: allowed zero-slot interference
reduces its minimum margin. Distinguish three notions consistently:

1. same-frequency sign-pattern cancellation (the paper's definition);
2. complete offset dependence, with no other candidate-dependent modes;
3. attaining the class upper bound.

If the six-neuron/minimality remarks use a stronger condition, state that
condition explicitly instead of calling it `interference-free` as defined here.
The two formulas above were also checked by direct Fourier evaluation at
p=5 and p=7 (maximum discrepancies below 4e-15); the cosine identities provide
the exact argument.

### A3. Zero-amplitude exception in the feature-cost equality statement

The main-text equality characterization needs beta != 0. When beta=0,
equality holds if and only if every beta_l=0; it does not require exactly nu
nonzero coefficients. The Lean equality theorem already has B != 0 explicitly.

Suggested statement: "If beta != 0 and 0<r<1, equality holds if and only if ...".
The proof's opening sentence should record the zero equality case as well.

### A4. Normalize the random allocation before applying the block formula

In the scaling proof the random allocation has q=sum_j y_j^r <=1, not
necessarily equality (repeated draws can make it strict). The network paragraph
invokes the block formula for budget exactly one. Insert:

    Replace y by y/q^s. Its budget is one, its support is unchanged,
    and its coverage is at least that of y.

Here q>0 because the construction has positive coverage. This is a one-line
repair to the written proof, not a change in the theorem. The Lean construction
handles normalization separately.

### A5. Keep width quantifiers explicit in the scaling proof

The exact lower bound gamma* >= nu^(1-s) B_p Gamma_s(p) is immediate at
sufficient width to realize a programme optimizer. At the theorem's smaller
stated sufficient width nu*ceil(8 log p), use the random allocation directly;
do not assume it realizes Gamma_s(p). Write gamma*_{b,m} for a fixed width and
use the random lower bound plus the universal upper bound there. The formal
network theorem has the correct fixed-width form.

### A6. Scope and boundary wording

* `cor:few`: add K>=1 and nonzero parameter norm when dividing by the norm.
  If all A_j=0, the phase bound gives h<=0; handle that case before dividing
  by ||A||_{1/s}.
* The conjecture-status paragraph needs theorem pointers and exact parameter
  ranges. In particular, specify the degree in the p=5,b>=5/2 statement.
  The local Lean status says this paragraph was not formalized.
* "No ascent from any start" should be "None of the tested starts" and be
  separated from the proved claims.
* "Strict local maximum" needs its precise quotient/restriction: normalized
  margin is invariant under global rescaling, and the constructions have
  continuous phase symmetries even after norm normalization.
* Update the group-counterexample list if retained: S5 at nu=4 is now included
  in this directory's exact constructions. Repair `sec:groups` if that section
  has been removed.
* The main-text sentence saying logarithmic growth "holds ... over all
  networks" conflates two results. State explicitly: programme support is
  Theta(log p); unrestricted optimal margin has the displayed order; a
  constant-factor optimal unrestricted network needs Omega(log p) clean
  frequencies. No unrestricted support upper bound has been proved.
* At switches, optimal budget shares can jump. Their formula is continuous
  along a fixed vertex branch, not necessarily across transitions. Likewise
  arbitrary selections among permanently tied vertices need not be piecewise
  constant. State the finite-branch/set-valued conclusion precisely.
* The dense-window displayed denominator must be positive: impose
  (p-1)/2 > ceil(8 log p), or say "for sufficiently large p". The asymptotic
  conclusion remains valid.

## B. Recommended simplifications

### B1. Make feature cost a three-step proof

Keep the chain lemma. Combine the minor expansion, Hadamard bound, symmetric
polynomial bounds, and generating inequality into one determinant-comparison
argument (retain the coefficientwise strengthening as an optional remark).
After beta=1, u_l=m_l/sqrt(nu), d_l=nu beta_l, x_l=|d_l|:

    (1+t)^nu = |det(I+t H)|
      = |sum_L t^|L| (prod_{l in L} d_l) det Gram(u_l:l in L)|
      <= sum_L t^|L| prod_{l in L} x_l
      = prod_l (1+t x_l),                 t>=0.

Sylvester's identity plus principal-minor expansion justifies the equality;
Gram positivity and Hadamard justify the inequality. The note that H is
triangular, not necessarily Hermitian, is important and should remain.

For the integration step, the exact Beta-integral constant is unnecessary.
For 0<r<1 define

    J_r(x)=integral_0^infinity log(1+xt) t^(-1-r) dt.

Convergence is established exactly as in the present proof. Substitution gives
J_r(x)=x^r J_r(1), with 0<J_r(1)<infinity. Integrating the logarithmic comparison
and cancelling J_r(1) yields sum x_l^r>=nu. This avoids integration by parts
and evaluating the Beta integral without changing the result or equality proof.

The existing equality argument remains useful: equality of the positive
integrals forces equality of the generating polynomials; their roots give
exactly nu unit x_l, and the top-degree coefficient forces Gram determinant
one. Do not shorten this part to an unsupported "equality is immediate".

### B2. Replace Minkowski--Weyl in the vertex proof by coercivity

The recession decomposition is correct for this polyhedron, but is unnecessary.
The feasible set is nonempty and closed, and sum v_j^r has compact sublevel
sets there, so a minimum exists. Strict concavity then rules out a non-extreme
minimizer by the midpoint argument already in the text.

This is shorter and avoids introducing a convex-geometric theorem that does
not otherwise do work in the proof.

### B3. Explain the scaling proof's mechanism before its estimates

Add two sentences before the positive-Fourier inequality:

    "If every nonzero offset had large coverage, the exponential average
    would be small. Positivity of Fourier coefficients forces that same
    average to remain large under a bounded amplitude budget. Comparing
    the two bounds limits the achievable margin."

One line makes the nonnegative coefficients self-contained:

    exp(z cos theta)=exp((z/2)e^(i theta)) exp((z/2)e^(-i theta)).

The absolutely convergent exponential series has nonnegative coefficients.
There is no need to introduce all modified Bessel functions I_k: only the
constant coefficient I_0 and its elementary integral bound are used.

Keep the "no independence" sentence: it addresses a natural concern about
correlated phases. State a_s=3+s/2 again at the proof's start and make clear
that its numerical value is chosen for convenience, not sharpness.

### B4. Reduce formatting fragmentation

Use numbered steps for the long cost and scaling proofs, and short paragraphs
for the bottleneck and ablation proofs. The current repeated itemize blocks
make a continuous argument look like a collection of disconnected checks.
Standard facts such as Hadamard need not each receive a separately numbered lemma.

### B5. Small algebraic and editorial improvements

* In the block proof, package the already-computed penalty as
  P_b=nu^(1-r) B_p^(-r) sum_j y_j^r. Raising to 1/r immediately gives the
  normalized-margin constant and avoids the long string of powers of 2.
* Remove the numerical-check parenthetical from the middle of that exact
  algebra; put reproducibility checks in a separate paragraph.
* In the power-neuron paragraph the single-neuron constant actually satisfies
  C_b^(nu/b)=n! B_p: beta_power=n! beta_product and the same balanced cosine
  neuron attains the product budget. Network-level attainability is a different
  question. This removes the need to use the external quadratic-network result
  merely to identify C_3.
* In the dense proof, call phase alignment equality in the triangle inequality,
  rather than equality in subadditivity at r=1 (the latter step is linear).
* Replace `\frac129^{1-s}` by `\tfrac12\,9^{1-s}` for readability.

## Recommended priority

First repair the explicit incorrect auxiliary claims and the omitted equality
condition. Then simplify the feature-cost proof, normalize the random allocation
explicitly, and tighten the main-text scope around unrestricted support counts.
The principal product-network proof strategy does not need to be replaced.
