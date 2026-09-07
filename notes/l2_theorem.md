# Plain-l2 max-margin sparsity: what is provable with Morwani's machinery at b=2 (2026-09-04)

Setting (Morwani et al. Sec 2/4): one-hidden-layer network, one-hot a,b in Z_p,
neuron phi({u,v,w},a,b) = (u_a + v_b)^2 w in R^p, f = sum_i phi_i, homogeneity
nu = 3. Their norm: ||theta||_{2,3} = (sum_i ||omega_i||_2^3)^{1/3}. Weight decay:
||theta||_2^2 = sum_i ||omega_i||_2^2, i.e. their (a,b) = (2,2).

## 1. Where their pipeline survives at b=2 and where it breaks

Step 1 (single-neuron optimisation, Lemma 5 first bullet) uses only a = 2 and
survives verbatim: for any class weighting tau the tau-weighted expected margin
of a unit neuron is sum_k c_k(tau) |u_k||v_k||w_k| (phase-aligned), a sum of
products over frequencies, maximised at a SINGLE frequency (sum_k c_k x_k y_k z_k
<= max_k c_k ||x|| ||y|| ||z|| <= max_k c_k 3^{-3/2}, equality iff one k and
|u|=|v|=|w| = 3^{-1/2}).

Step 2 (combining neurons, Lemma 5 second bullet) needs b = nu (their remark,
p.23). At b = 2 the network's weighted margin is sum_i lambda_i^3 psi_i under
sum_i lambda_i^2 <= 1, a convex function on the sphere: it is maximised at a
VERTEX, one neuron with all the mass. A single quadratic neuron cannot cancel
its own junk (cos(w(a+b+c)), cos(w(a-b)) and squared terms), so its true
margin is strictly below its weighted margin and condition C.1 fails. Hence no
certificate pair (theta*, q*) of their type exists at b = 2: the max-min
inequality is strict. Their technique is not just inconvenient at b=2, it is
blocked by a duality gap.

Quantified: the weakest tau-bound is min_tau max_k <tau, g_k> with g_k(D) =
1 - cos(2 pi k D/p). By symmetry this LP's value is the uniform-mixture
coverage 1 + 1/(2 nf) = 1.0089 at p=113 (sum_{k=1}^{nf} cos(2 pi k D/p) =
-1/2 for every D != 0, verified). That is the ONLY upper bound their route
gives, in units where one unit-mass frequency delivers coverage 1.

## 2. Junk lemma (new, elementary): non-clean logit content never helps the margin

Decompose any logit table L(a,b,c) = C + T with C the projection onto
functions of (a+b-c) and T orthogonal to all such functions. For fixed offset
D the margin is m_D(a,b) = [C(0) - C(D)] + [T(a,b,a+b) - T(a,b,a+b+D)]. The
indicator of {c = a+b+D} is itself a function of (a+b-c), so both T-terms
have zero mean over (a,b), hence the bracket has min <= 0 over (a,b), hence
    min-margin(L) <= min-margin(C).
So the max-margin problem over ANY network is bounded by the max-margin of its
clean part, and the whole question is the representation cost of clean
amplitudes.

## 3. Theorem (restricted class = Morwani's neuron class)

Over networks whose neurons are each supported on ONE frequency (the class
their Theorem 7 shows the L_{2,3} optimum lives in, and the class trained
neurons occupy empirically — their Fig 4b, our nudial ladder):
  (i)  a single-frequency neuron with factor norms alpha, beta, gamma delivers
       clean amplitude A = c0 alpha beta gamma (c0 = (2/p)^{3/2}/2) and costs
       alpha^2+beta^2+gamma^2 >= 3 (A/c0)^{2/3} (AM-GM; equality iff balanced);
  (ii) same-frequency neurons add amplitude at most additively, and x^{2/3} is
       subadditive, so ||theta||^2 >= 3 sum_k (A_k/c0)^{2/3};
  (iii) by the junk lemma, margin <= min_D sum_k A_k g_k(D).
Therefore max-margin <= c0 3^{-3/2} Gamma*(p), with
    Gamma*(p) = max { min_D sum_k x_k g_k(D) : sum_k x_k^{2/3} <= 1, x >= 0 },
and the bound is attained by the phase-complete construction (4 balanced
phase-shifted neurons per frequency cancel all junk at zero extra cost —
Morwani's own construction, mass now split per the program). Every maximiser's
frequency support is a maximiser of the finite program. At p=113 the program
is solved exactly by enumeration: support size K* = 6, Gamma* = 0.266
(equal-share; free amplitudes do not improve it), ladder
K=4: .886, 5: .974, 6: 1, 7: .966, 8: .938, 12: .870 (the wide nu=3 plateau).
Asymptotics K*(p) ~ 4 log p /(nu-2) already proven (nudial/kstar_pscale.py).
For general nu the same three steps go through for product circuits of nu
balanced factors (exponent 2/nu), which is the transformer/MLP ladder.

## 4. The honest open gap: superposition

The restricted-class optimum is 0.266; the unrestricted upper bound from
Section 1 is 1.009 — a 3.8x gap. Closing it means bounding the clean
amplitude a MULTI-frequency neuron can deliver per unit norm net of the junk
it must then emit. The naive per-neuron bound fails in the wrong direction
(x^{2/3} is concave, so one neuron carrying two frequencies is cheaper than
two neurons), so the proof must use that superposed neurons emit cross-
frequency junk of the same order as their clean content, which by the junk
lemma costs margin. Nobody has that bound. Evidence that the gap is not real:
trained networks under weight decay have single-frequency neurons (Morwani
Fig 4b, our ladder), all our constructions and every trained run sit at or
below the restricted optimum, and He et al. 2602.16849's "dense under wd"
result is an assumed dynamics condition, not a margin theorem (adversarial
review 08-24).

## 5. Status for the paper

Provable now: Theorem (Sec 3) + junk lemma (Sec 2) + the impossibility of a
Morwani-type certificate at b=2 with the 3.8x quantified gap (Sec 1). That
converts "ansatz optimum" into "theorem in the neuron class the prior theorem
itself singles out", states the exact remaining open problem, and explains
mechanically why their norm gives density (mass-splitting is free at b=nu).
Not provable now: sparsity over all networks.

## 6. Re-check of Section 3 (added after review) and the exact shape of the open step

Two things the Section 3 proof needs that were implicit:
- Exactness of the clean projection for a single-frequency neuron: the
  expansion (u_a+v_b)^2 w_c has terms u_a^2 w_c, v_b^2 w_c (functions of (a,c)
  or (b,c) only) and 2 u_a v_b w_c = (alpha beta gamma/2)[cos(w(a+b-c)+phi)
  + cos(w(a+b+c)+.) + cos(w(a-b+c)+.) + cos(w(a-b-c)+.)]. Each non-clean term
  is orthogonal to every function of (a+b-c) (its inner product with
  e^{i w_j (a+b-c)} contains a vanishing single-variable sum unless
  2w = 0). So the clean amplitude is exactly alpha beta gamma / 2 and
  A = c0 ||u|| ||v|| ||w|| with c0 = (2/p)^{3/2}/2, using ||u||^2 = alpha^2 p/2
  (+ any DC part, which only raises cost and adds no clean content).
- Phase lemma: with clean part sum_k A_k cos(w_k n + phi_k), n = a+b-c, the
  margin is min_{n != 0} sum_k A_k [cos phi_k - cos(w_k n + phi_k)]. Averaging
  the phi and -phi versions (n -> -n symmetry) and using min(f+g) >= min f +
  min g gives margin(A, phi) <= margin(A cos phi, 0) <= margin(A, 0). So
  phi = 0 is optimal and the finite program with g_k = 1 - cos is the right one.
With these, Section 3 is complete: cost lower bound (AM-GM + subadditivity),
margin upper bound (junk lemma + phase lemma), attainment (phase-complete
construction, mass split per the program).

The general case, precisely: a neuron whose u, v, w carry frequencies
{k_1..k_K} delivers clean amplitude a_k = |u_k v_k w_k|/2 at each k at cost
(p/2) sum_k (u_k^2 + v_k^2 + w_k^2) >= 3 (sum_k a_k / c0)^{2/3}. Because
t^{2/3} is concave, carrying K frequencies in ONE neuron is K^{1/3} cheaper
than K single-frequency neurons at the same amplitudes, i.e. up to K^{1/2}
more margin (margin ~ cost^{3/2}). That is the whole 3.8x gap. What the
superposed neuron also emits is cross junk 2 u_{k} v_{j} w_{l} cos cos cos for
k != j, of amplitude comparable to its clean content. The missing theorem is
a QUANTITATIVE junk lemma: min-margin(C + T) <= min-margin(C) - kappa ||T||
for some kappa > 0 depending only on p (the mean-zero argument gives only
kappa >= 0), strong enough that the margin lost to junk exceeds the K^{1/2}
gained from cheaper cost. Symmetrising the network over the task symmetry
(a,b,c) -> (a+s, b+t, c+s+t) does NOT work at b=2: the averaged network has
p^2 m neurons at scale p^{-2/3} each and costs p^{2/3} times more — the same
concavity that produces sparsity makes symmetrisation expensive.

## 7. Correction (2026-09-07): the superposition gap is not real, and the attainment claim was wrong

Two errors in Sections 3-4 and 6, found by direct computation (session
2026-09-07; numerics reproduced inline below).

(a) THE UPPER BOUND IS UNRESTRICTED. Section 6 bounded a multi-frequency
neuron's cost by 3 (sum_k a_k / c0)^{2/3}, i.e. AM-GM applied to the
TOTAL clean amplitude. That is loose. Parseval splits every factor norm
over frequencies, ||u||^2 = (p/2) sum_k rho^u_k^2, and the clean amplitude
at k involves only the k-components, a_k = rho^u_k rho^v_k rho^w_k / 2
(verified: a multi-frequency neuron with unit cosines at K frequencies has
a_k = 0.5 at each of them and leak < 1e-14 elsewhere). So AM-GM applies
PER FREQUENCY:
    (p/2)(rho^u_k^2 + rho^v_k^2 + rho^w_k^2) >= (3p/2) (2 a_k)^{2/3},
and summing over k,   cost_n >= (3p/2) 2^{2/3} sum_k a_{k,n}^{2/3}
for EVERY neuron, single- or multi-frequency. Subadditivity of x^{2/3}
across neurons then gives sum_k A_k^{2/3} <= ||theta||^2 / ((3p/2) 2^{2/3})
for the whole network, which is exactly the constraint of the finite
program. With the junk lemma and the phase lemma:
    max-margin(any network) <= c0 3^{-3/2} Gamma*(p) ||theta||^3.
Checked on random band-limited multi-neuron networks: sum A_k^{2/3} /
budget = 0.19-0.54, never above 1. A neuron carrying K frequencies costs
EXACTLY the same per unit of sum_k a_k^{2/3} as K single-frequency neurons
(269.06 at p=113 for K = 1, 2, 3, 4): superposition saves nothing. The
1.009 "unrestricted bound" of Section 1 is simply a weaker bound, not
evidence of a gap. The restriction to single-frequency neurons in Section
3 is unnecessary; the 3.8x of Section 4 is void.

(b) THE BOUND IS NOT ATTAINED BY THE PHASE-COMPLETE CONSTRUCTION. Section
3 claimed that 4 phase-shifted neurons per frequency cancel all junk "at
zero extra cost". That is the L_{2,3} accounting (mass-splitting is free
at b = nu), not the l2 accounting. Under l2, n_k neurons per frequency at
factor amplitude rho deliver clean n_k rho^3/2 at cost 3 n_k rho^2 (p/2),
whereas the bound allows one neuron of amplitude n_k^{1/3} rho at cost
3 n_k^{2/3} rho^2 (p/2): the split costs n_k^{1/3} more, so the
construction's normalised margin is n_k^{-1/2} of the bound. Verified on
the K* = 6 support with (phi, psi, phi+psi) product-grid phases: n_k = 16
(4x4 grid) is the smallest exactly junk-free block found (junk energy 0,
gamma = 0.250 x bound, exactly 16^{-1/2}); n_k = 64 gives 0.125; n_k = 1
and 4 leave 89% / 75% junk energy and negative margin. No exactly
junk-free block with n_k <= 5 exists on the pi/4 phase grid in this
family (exhaustive search). The same n_k^{1/3} splitting price is what
makes l2 sparse across frequencies; here it is paid within a frequency
to cancel junk.

STATUS after the correction (quadratic 1-hidden-layer MLP, plain l2):
- Upper bound, all networks: c0 3^{-3/2} Gamma*(p) ||theta||^3, with
  Gamma* from the sparse finite program (K* = 6 at p = 113; ladder as in
  Section 3). THEOREM, unrestricted.
- Lower bound, constructive: n_k^{-1/2} of that with n_k = 16 (0.25), same
  program shape for every support, so the construction's argmax is also K*.
- The true supremum lies in [0.25, 1] x bound (junk-tolerant networks with
  fewer neurons per frequency may beat the junk-free construction). The
  constant does NOT affect K* on the upper-bound side; on the lower-bound
  side supports with Gamma_S < Gamma*/4 are excluded as optima, which
  removes K <= 2 and K >= ~25 but not the plateau. The exact K* of the
  supremum remains an empirical statement (P-N12 fixed-support) plus the
  upper-bound argmax, not a two-sided theorem.
- Consequence for the paper: state "sparse program = unrestricted upper
  bound on the l2 max-margin; junk-free construction within 4x with the
  same shape"; drop "theorem in the single-frequency class" and drop the
  3.8x superposition gap; the open constant is the junk-cancellation
  price, min over networks of (margin lost to junk) vs (cost of splitting).
