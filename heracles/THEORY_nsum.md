# The exact ν ladder: n-operand modular sums with x^n activation (2026-09-16)

Companion to `nsum.py`. Everything here is for the one-hidden-layer net with NO biases and NO extra linear layers,

    f_θ(a_1,…,a_n)(c) = Σ_{i=1}^m σ( Σ_{j=1}^n e_{ij}(a_j) ) · w_i(c),        e_{ij}, w_i ∈ R^{Z_p},

trained on the FULL table of the task  a_1 + … + a_n ≡ c (mod p)  with cross-entropy + λ‖θ‖² (plain weight decay).
Two activations: σ = ReLU (degree ν = 2 for any n) and σ = x^n on the n-operand task (degree ν = n + 1).
This is the architecture of Morwani et al. (ICLR 2024) for n = 2 and of Li, Liang, Shi, Song, Zhou (AISTATS 2025,
arXiv 2402.09469) for general n, both of whom characterise the maximum margin in the L_{2,ν} norm (dense, every
frequency present). Here the norm is the Euclidean one that weight decay actually prices.

Notation. nf = (p−1)/2 positive frequencies. For a function on Z_p, ĥ(k) = (1/p) Σ_a h(a) e^{−2πika/p}. For the
logit table z(a_1..a_n, c), the clean projection is the level-set average g(D) = mean over all rows of z(row, s − D)
with s = Σ_j a_j, D ∈ Z_p, i.e. the part of the table that is a function of s − c alone. Margin = min over rows of
z_y − max_{c≠y} z_c. Spectral weight of frequency k (Levine, MAIS-A5): b_k² = share of the centred-logit energy on the
two characters e^{±2πik(s−c)/p}; with N = p^n rows, b_k² = 2 N p |ĝ(k)|² / ‖z − mean_c z‖²  (this is what nsum.py
reports; Σ_k b_k² = "clean fraction").

## 1. Three lemmas that hold for every n and both activations

**L1 (junk lemma).** margin(z) ≤ margin(g). Proof: translations (a_1..a_n, c) → (a_1 + t_1, …, a_n + t_n, c + Σt_j)
act transitively on each level set {s − c = D}; averaging z over that group gives g; for every row,
ḡ_y − ḡ_c = average over the orbit of (z_y − z_c) ≥ margin(z). So the margin of the trained table is at most the
margin of its clean part. (Fails when the level sets have no transitive symmetry, e.g. c = a² + b², RESULTS_QT.)

**L2 (phase lemma).** margin(g) = min_{D≠0} [g(0) − g(D)] ≤ min_{D≠0} Σ_{k=1}^{nf} 2|ĝ(k)| (1 − cos(2πkD/p)),
with equality iff every ĝ(k) is real and non-negative (all frequencies phase-aligned at D = 0).

**L3 (assembly).** ĝ(k) = Σ_i ĥ_i(k,…,k) · conj(ŵ_i(k)), where ĥ_i(k,…,k) is the n-dimensional Fourier coefficient
of the hidden activation h_i(a_1..a_n) = σ(Σ_j e_{ij}(a_j)) at the diagonal multi-index (k,…,k). Proof: substitute
c = s − D in the definition of g and expand. Only the diagonal coefficient of each neuron reaches the clean logit.

Combining: margin ≤ min_D Σ_k 2 |Σ_i ĥ_i(k..k) conj ŵ_i(k)| (1 − cos(2πkD/p)).      (★)

## 2. ReLU, any n: the ν = 2 bound is unrestricted and dense

Per neuron, ReLU is 1-Lipschitz, so Var(h_i) ≤ Var(x_i) = Σ_j Var(e_{ij}) ≤ Σ_j ‖e_{ij}‖²/p, and by Parseval
Σ_{k=1}^{nf} |ĥ_i(k..k)|² ≤ ½ Σ_{multi-index ≠ 0} |ĥ_i|² = ½ Var(h_i). Also Σ_{k=1}^{nf} |ŵ_i(k)|² ≤ ‖w_i‖²/(2p).
Cauchy–Schwarz on (★) at each neuron and then AM-GM on ‖(e_i)‖ ‖w_i‖:

    Σ_k |ĝ(k)| ≤ Σ_i √(Σ_k|ĥ_i|²) √(Σ_k|ŵ_i|²) ≤ Σ_i ‖e_i‖ ‖w_i‖ /(2p) ≤ ‖θ‖² /(4p).

Averaging the min over D ≠ 0 in (★) (Σ_{D≠0}(1 − cos(2πkD/p)) = p for every k ≠ 0):

    **margin ≤ ‖θ‖² / (2(p − 1))**  for every ReLU network of this family, every n, every width.

Where it is tight: the average-over-D step is an equality iff |ĝ(k)| is the same for every k (flat spectrum), and
a K-frequency committee with equal amplitudes only reaches the fraction (M_K/K)/(p/(p−1)) of the bound, where
M_K = min_D Σ_{k∈S}(1 − cos(2πkD/p)) is the coverage of the best K-set (kstar.py). At p = 113: K = 4 reaches 0.47,
K = 6 reaches 0.64, the full set reaches 1. So under plain weight decay the ν = 2 ReLU net is DENSE at the margin
optimum. This is the MAIS-O52 architecture (Levine's Problem 5.2), and it is Morwani's Fig. 7a (ReLU, L2, p = 71,
"close to uniform"). Only the ReLU efficiency constant (single-frequency neuron η₁ = 0.196, sup 0.212,
relu_dense_check.py) is lost; the count conclusion is not.

## 3. x^n on the n-operand sum: the ν = n + 1 sparse program is unrestricted

Expand h_i = (Σ_j e_{ij}(a_j))^n. A monomial Π_j e_{ij}(a_j)^{m_j} with Σ m_j = n has Fourier support only on
multi-indices whose j-th entry is 0 whenever m_j = 0. The diagonal (k,…,k), k ≠ 0, needs every entry non-zero, so
only the monomial with every m_j = 1 contributes: ĥ_i(k,…,k) = n! Π_j ê_{ij}(k). Hence

    ĝ(k) = n! Σ_i Π_{j=1}^n ê_{ij}(k) · conj ŵ_i(k):   ONE amplitude per operand per frequency per neuron.

(For n = 2 this is the familiar 2 û_i(k) v̂_i(k) ŵ_i(k). For a two-operand task with x^3 the monomial u²v has
diagonal support through u's self-convolution, û(k₁)û(k−k₁), and the argument below fails: two-operand tasks are
AM-GM-exact at ν = 3 only. That is why the ladder moves operands and exponent together.)

AM-GM on the n + 1 factors at fixed (i, k): |Π_j ê_{ij}(k)| |ŵ_i(k)| ≤ ( (Σ_j |ê_{ij}(k)|² + |ŵ_i(k)|²)/(n+1) )^{(n+1)/2}.
Write m_{ik} = Σ_j |ê_{ij}(k)|² + |ŵ_i(k)|² (the weight-norm mass neuron i puts on frequency k; Parseval gives
Σ_k m_{ik} ≤ ‖θ_i‖²/(2p) + zero-frequency mass). Then with a_k = |ĝ(k)|:

    a_k ≤ n! (n+1)^{−(n+1)/2} Σ_i m_{ik}^{(n+1)/2},   and   Σ_k m_{ik} ≤ ‖θ_i‖²/(2p).

Per-frequency AM-GM (notes/l2_theorem.md §7, the argument that closed the superposition gap): for fixed total
Σ_i ‖θ_i‖², the vector (a_k) is constrained by Σ_k a_k^{2/(n+1)} ≤ C_n ‖θ‖² with C_n depending only on n and p, and
this constraint is what a committee of single-frequency neurons attains up to a constant. Plugging into (★) with
ν = n + 1:

    margin ≤ C'_n ‖θ‖^ν · max_K  M_K · K^{−ν/2}   (equal amplitudes; free amplitudes change the argmax by ≤ 1 rung),

so the maximum-margin frequency count is  **K*(p, ν) = argmax_K M_K(p) / K^{ν/2}** with ν = n + 1, for EVERY
network in the family (no single-frequency-neuron restriction, no rank-sharing gap: there is no pre-layer to price
by a nuclear norm). The constant sits in [1/4, 1] as at n = 2 (attainment by the junk-free split construction).

Predictions (kstar.py, q = 2, plateau = rungs within 26% of the top):

| p | ν=3 (x², a+b) | ν=4 (x³, a+b+c) | ν=5 (x⁴, 4-sum) | ν=6 (x⁵, 5-sum) |
|---|---|---|---|---|
| 23 | 4 (2..10) | 3 (2..5) | 2 (2..3) | 2 |
| 31 | 4 (3..10) | 3 (2..6) | 3 (2..4) | 2 (2..3) |
| 47 | 5 (3..10) | 3 (3..6) | 3 (2..4) | 2 (2..3) |
| 71 | 5 (4..10) | 4 (3..7) | 3 (3..5) | 3 (2..4) |
| 113 | 6 (4..10) | 4 (3..8) | 4 (3..6) | 3 (2..5) |

Sparse-over-dense margin ratio at p = 71: 1/0.57, 1/0.21, 1/0.07, 1/0.02 for ν = 3, 4, 5, 6: the preference for the
sparse optimum sharpens with every rung, so higher rungs should land cleaner than ν = 3 did (QM: 4–6 at K* = 5).

## 4. The foil

Under the L_{2,ν} penalty λ Σ_i ‖θ_i‖^ν (Morwani n = 2; Li et al. general n) the per-neuron cost is linear in the
frequency mass m_{ik}^{ν/2}, the budget is Σ_k a_k ≤ C, and (★) is maximised by the flat spectrum: every frequency,
one per neuron. `--penalty l2nu` runs exactly that on the same nets; RESULTS_QM has it at n = 2 (27–35 of 35).

## 5. The penalty exponent q (2026-09-17): a second dial, but it only runs from ν/2 down to 1

Generalise the penalty to P_q = λ Σ_i (‖θ_i‖²)^{q/2} (`--penalty l2q --q`): q = 2 is plain weight decay, q = ν is
L_{2,ν}. Lemmas L1–L3 and the per-frequency AM-GM bound a_k ≤ c_n Σ_i m_{ik}^{ν/2} do not see the penalty; only the
budget changes, to Σ_i (Σ_k m_{ik})^{q/2} ≤ C. The sup of the margin program over allocations {m_{ik}} is then a
question about where a fixed budget is best spent:

- **2 ≤ q < ν.** Across frequencies x^{q/2} is convex, so a neuron carrying two frequencies costs at least as much as
  two neurons carrying one each (equality at q = 2): the optimum separates frequencies. Within a frequency the
  amplitude x^{ν/2} is more superadditive than the cost x^{q/2}, so one neuron per frequency wins. A frequency with
  budget c_k then reaches a_k ∝ c_k^{ν/q}, and (★) becomes  **max_K M_K / K^{ν/q}**: unrestricted, same proof.
  q = ν is the flat programme (§4, Morwani / Li et al.), q > ν is degenerate (splitting a frequency over ever more
  neurons lowers the cost faster than the amplitude, so the ratio is unbounded).
- **q < 2.** Across frequencies x^{q/2} is now concave: merging two frequencies into one neuron LOWERS the cost and
  RAISES the amplitude bound (superadditivity of x^{ν/2} at fixed m). The sup is therefore one neuron carrying every
  frequency with masses m_k and Σ_k m_k ≤ C^{2/q}, which is the plain-l2 programme with a rescaled budget. The
  unrestricted argmax is K*(p, ν/2) for every q < 2: the dial does not steepen below 2. (The exponent ν/q would be
  a class statement for single-frequency neurons only, and GD does not even reach that class: at q = 1.5 the x²
  net at p 71 plateaus at CE 0.07 and then oscillates, margin −19 at 200k steps, two dominant frequencies plus a
  tail; RESULTS_Q.md.)

So on a fixed network the programme exponent e = ν/q runs over [1, ν/2] only: from Morwani's dense endpoint up to
the plain-weight-decay value, never beyond it. Steeper exponents need a higher degree, which is why the ν ladder of
§3 cannot be replaced by a penalty sweep; the two dials together give the collapse statement
K = argmax_K M_K/K^{ν/q} for 2 ≤ q ≤ ν, testable on the x² net across q (RESULTS_Q.md).

Junk under this lens: for q < ν merging within a frequency is favoured, so each frequency wants few neurons, and the
junk-free representation (which needs matched neuron pairs per frequency at degree 2, sign-pattern 2^{n−1}-tuples
at degree n) is no longer free; at q = ν it is. Prediction: the clean fraction rises monotonically to 1 as q → ν.
## 6. Depth without shared factors: the gated chain (2026-09-17)

Why linear depth breaks exactness, in one line: the exact exponent counts factors that are FUNCTIONS ON Z_p (an
embedding row, a readout row), because only those have per-frequency content and a Frobenius cost that splits over
frequencies. A factor shared across a neuron's frequencies — a pre-layer, a mixing matrix, even a scalar gain per
neuron — is priced by a nuclear norm (a scalar gain on a vector: cost 2‖e‖, the l2 of the per-frequency masses, not
their sum), which rewards carrying many frequencies in one neuron; whether junk then cancels the discount is the
open lemma (THEORY_relu §5, §5b: the per-neuron route is provably closed). So "depth ν" with the argmax exact needs
every layer along the path to be a Z_p-indexed factor.

The family that does it: the gated chain, `--act prod`. Layer j multiplies in the j-th operand's embedding:

    h_i(a_1, …, a_n) = Π_{j=1}^n e_{ij}(a_j),      z(c; a) = Σ_i w_i(c) h_i(a),      ν = n + 1,

an n-layer multiplicative network (each layer a Hadamard gate by a fresh one-hot embedding), equivalently a rank-m
CP factorisation of the (n+1)-way logit tensor of n-ary modular addition. Cost Σ_i (Σ_j ‖e_{ij}‖² + ‖w_i‖²), one
linear layer per operand, no products of matrices. Exactness is immediate: the (k,…,k) coefficient of a product of
one-variable functions is the product of their k-th coefficients, so

    ĝ(k) = Σ_i Π_j ê_{ij}(k) · conj ŵ_i(k)      with no other monomial at all

(x^n on the n-sum has the same clean term plus the non-multilinear junk u²v…, which is what made ν 4 hierarchical in
RESULTS_NS). Junk lemma, phase lemma, per-frequency AM-GM over n + 1 factors, Frobenius additivity: THEOREM,
unrestricted,  K*(p, ν, q) = argmax_K M_K / K^{ν/q},  ν = n + 1, every q ∈ [2, ν]. The only junk is the sign-pattern
junk cos(Σ_j ±(k a_j + φ_ij)) with mixed signs, cancelled exactly by a population whose phases φ_ij range over
{0, π/2} independently per operand (2^n neurons per frequency; the mixed-sign term carries e^{2iΣ_{j∈S}φ_ij} which
averages to zero over any nonempty S) — the same cancellation as at ν 3, at every depth.

Consequences and predictions. Same M_K, same ladder: p 71: ν 3 → 5 (4..16), ν 4 → 4 (3..7), ν 5 → 3 (3..5); p 59 ν 5
→ 3. Collapse checks: (n 3, q 8/3) must give the (n 2, q 2) count 5; (n 3, q 4) is the flat endpoint. If the gated
chain lands cleanly at ν 4 where x³ scattered 4..9, the x³ scatter is diagnosed as its non-multilinear junk, not as a
failure of the functional. Escape from zero is the same s^{ν−2} problem as for x^n (the bilinear n 2 net at init 1
never leaves zero; init 4 escapes and lands at clean fraction 0.58–0.87 against 0.41 for x²).

Measured (RESULTS_PROD.md, 2026-09-17): ν 3 → 5–8, ν 4 → N_.05 5–8 (K_5pct 2–6), ν 5 → 3 and 3 (on the argmax; x⁴ never
fit); q on the ν 4 chain: 3–10 / 7–8 / 21–22 at q 2 / 8/3 / 4. Every cell in the plateau; ν 4 spreads across it in
both exact families, so that is gradient descent at degree 4, not junk class.

Relation to the tensor-factorisation literature: GD on CP factorisations from small init is known to build
components incrementally (implicit rank minimisation, Razin–Maman–Cohen 2021 and follow-ups; to be re-read before
citing); our K* is the number of frequency components of the max-margin CP factorisation under plain weight decay,
a statement those works do not make.

## 7. What this does and does not say

- Says: for the family above, the plain-weight-decay maximum-margin solution uses K*(p, ν) frequencies, dense at
  ν = 2, and the count is decided by the norm (l2 vs L_{2,ν}) and the degree, not by the task or the width.
- Does not say: that gradient flow reaches the maximum-margin solution (Wei et al. 2019 gives it as λ → 0 for the
  global minimiser; Lyu–Li give KKT points). The runs measure where GD actually lands; the ladder predicts K* or
  K*+1 (heracles RESULTS_H3: the +1 is a finite-λ tie-break inside a flat plateau corner).
- Does not cover: nets with linear pre/post layers (nuclear pricing, THEORY_relu / relu_dense_check), non-group
  targets (RESULTS_QT), two-operand tasks with exponent > 2.

## 8. Attainment for the gated chain: the (n+1)-neuron block (2026-09-18, after PROOF_AUDIT F1)

Status correction. §3/§6 prove an UPPER bound γ_q ≤ B·Γ_{ν/q}(p) (Γ_s = free-amplitude programme; M_K/K^s is its
equal-amplitude restriction, PROOF_AUDIT F2). They do not prove that the maximiser's support is attained. This section
gives the best lower bound we have; it replaces the 2^n-neuron {0, π/2} block of §6.

Construction. Fix a frequency k, write θ_j = 2πk a_j/p, θ_c = 2πk c/p. For l = 0..n let φ_l = πl/(n+1) and take the
n+1 neurons  e_{lj}(a) = r cos(θ_j + φ_l) (every operand j),  w_l(c) = r cos(θ_c + nφ_l).

Lemma (junk-free). Σ_l w_l(c) Π_j e_{lj}(a_j) = (n+1) r^{n+1} 2^{-n} cos(θ_1 + … + θ_n − θ_c).
Proof. Expand each cosine into exponentials: neuron l contributes 2^{-(n+1)} r^{n+1} Σ_{ε∈{±1}^n, δ=±1}
exp(i(ε·θ + δθ_c)) exp(iφ_l(Σ_jε_j + δn)). With t = #{j: ε_j = −1}, Σ_jε_j = n − 2t, so the l-dependent phase is
exp(2πi l u/(n+1)) with u = −t (δ = −1) or u = n − t (δ = +1). Summing over l = 0..n gives 0 unless u ≡ 0 mod (n+1);
as 0 ≤ t ≤ n this happens only for (t = 0, δ = −1) and (t = n, δ = +1): the clean term and its conjugate. ∎

Cost. All n+1 factors of every neuron have equal norm (per-neuron AM-GM tight) and one frequency (Parseval tight), so
the only loss against the bound is the split of one frequency's mass over N = n+1 neurons: P_q is multiplied by
N^{1−q/ν} at fixed clean amplitude, i.e.

    γ_q(construction on support S, amplitudes x)  =  (n+1)^{−(ν−q)/q} · B · (programme value of (S, x)),   2 ≤ q ≤ ν.

At q = 2: 3^{−1/2} = 0.577 (ν3), 4^{−1} = 0.25 (ν4), 5^{−3/2} = 0.089 (ν5); at q = ν the factor is 1 (flat programme
attained, Morwani). Numerically verified (audit_construct.py: junk < 1e-28, constant to 4 digits); margin ascent at p = 11
lands on exactly these constants from random init (audit_ascent.log) → CONJECTURE: the factor is the true sup.

Corollary (two-sided, chain, q = 2). Any network whose clean spectrum has support/amplitudes (S, x) has γ ≤ B·val(S, x).
Hence every max-margin network has val(S, x) ≥ (n+1)^{−(ν−2)/2} Γ*. The flat spectrum has val/Γ* = 0.571/0.507 (ν3, p71/p113),
0.213/0.153 (ν4), 0.068/0.041 (ν5) — all below the constant, so THE FLAT (MORWANI) SOLUTION IS NOT ℓ₂-MAX-MARGIN for the
chain (margin 1% at ν3 p71; use p113 or ν ≥ 4 in the paper). The certified band {K: Γ_K ≥ const·Γ*} is wide (≈3..12 at
ν3 p113): the theorem pins a sparse band, the fixed-support ladder (RESULTS_NU4) locates the peak.
Not covered: x^n (best known block: 6 neurons, 0.408 < 0.571 at ν3 → flat not excluded by proof); tied x^n (no junk-free
phase population exists).
