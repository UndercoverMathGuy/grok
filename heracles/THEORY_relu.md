# THEORY_relu — the ReLU family with linear layers: what is a theorem, what is measured, what is open (2026-09-17)

Companion to THEORY_nsum.md (x^n on the n-sum: unrestricted at every degree) and to relu_dense_check.py /
rank_share.py / sym_factor.py. Written after the 09-17 attempt to make the ReLU theorem unrestricted; the outcome is
one new unrestricted theorem (§3), a sharper statement of what is left (§5), and a measurement (§6).

## 0. Family, cost, normalised margin

tokens → E (d×p) → U, V (m×d) → ReLU → [W₁ (r×m)] → W (p×r) → logits;  no biases;  ν = 2 + n_pre + n_post.
Effective first layer X = [UE; VE] ∈ R^{2m×p} (rows u_i, v_i), effective readout Ω = W W₁ ∈ R^{p×m} (columns ω_i):

    z(c; a, b) = Σ_i ω_i(c) · relu(u_i(a) + v_i(b)).

For fixed (X, Ω) the smallest l2 cost over factorisations is 2‖X‖_* + ‖W‖_F² (ν 3, no post layer) or
2‖X‖_* + 2‖Ω‖_* (ν 4), nuclear norms, attained by balanced factors when d, r ≥ rank. The logit is degree 1 in X and
1 in Ω, so after optimising the free rescaling the ν-homogeneous normalised margin is

    γ₃ = margin / (3^{3/2} ‖X‖_* ‖W‖_F),      γ₄ = margin / (16 ‖X‖_* ‖Ω‖_*)     (relu_dense_check.gamma).

Frequency slices: Π_k = projector onto span{cos k·, sin k·} (Π_0 = constants). "Decoupled" = ‖X‖_* = Σ_k ‖XΠ_k‖_*,
i.e. no left singular vector shared across slices (rank_share.py: ratio_X = 1); "locked" = all neurons carry the same
K-comb with a common phase offset, X of rank 2, ‖X‖_* = Σ_k ‖XΠ_k‖_* / √K.

## 1. Lemmas (all activations, all n; proofs in THEORY_nsum §1 unless noted)

- L1 separability: the (k,…,k) Fourier coefficient of relu(u_i(a)+v_i(b)) is ĥ_i(k,k); the clean logit spectrum is
  ĝ(k) = Σ_i ĥ_i(k,k) · conj ω̂_i(k) (assembly).
- L2 junk lemma: margin(N) ≤ margin of the clean projection g(D) = mean_{a+b−c=D} z. (Proof: §3, step 2.)
- L3 phase lemma: the clean margin is ≤ min_{D≠0} Σ_k |ĝ(k)| (1 − cos 2πkD/p) =: (★).
- L4 Lipschitz budget: 2 Σ_{k≥1} |ĥ_i(k,k)|² ≤ Var(u_i(a)+v_i(b)) = (‖u_i − ū_i‖² + ‖v_i − v̄_i‖²)/p.

## 2. ν = 2 (no linear layers): unrestricted, dense — THEOREM

L1 + L4 + Cauchy–Schwarz + averaging (★) over D:  margin ≤ ‖θ‖²/(2(p−1)) for every ReLU network, every width,
tight only for a flat spectrum; a K-committee reaches (M_K/K)/(p/(p−1)) of it. This is Levine's MAIS-O52 network
and Morwani's Fig. 7a. (THEORY_nsum §2.)

## 3. Symmetrisation — THEOREM, unrestricted, exact (new 2026-09-17; verified to 1e-16 by explicit stacking at p 11)

The task is invariant under (a, b, c) → (a+s, b+t, c+s+t). For a network N with (X, Ω) define the translate N_{s,t}
with u_i(·+s), v_i(·+t), ω_i(·+s+t), and the average N̄ = (1/p²) Σ_{s,t} N_{s,t}: a network with p²m neurons
(i, s, t), first layer rows u_i(·+s), v_i(·+t), readout ω_i(·+s+t)/p².

  (i)   margin(N_{s,t}) = margin(N): the translate permutes the (a, b, c′) triples and preserves correct labels.
  (ii)  margin(N̄) ≥ margin(N): margin is a min of linear functionals of the logit table, hence concave under averaging.
  (iii) z_{N̄}(c; a, b) = g(a+b−c): the average of a character e^{i(ka + k′b + k″c)} over (s, t) picks up
        e^{i((k+k″)s + (k′+k″)t)} and vanishes unless k = k′ = −k″. So N̄ has NO junk; its logit is the clean
        projection of N. (ii)+(iii) is the junk lemma L2.
  (iv)  X̄ᵀX̄ = p² · avg_s T_sᵀ XᵀX T_s, a circulant-averaged Gram, block-diagonal in the Fourier basis with the
        2×2 block (‖XΠ_k‖_F²/2) I at every k ≥ 1 and ‖XΠ_0‖_F² at k = 0. Hence, exactly,
              ‖X̄‖_* = p · D_X,   D_X := ‖XΠ_0‖_F + √2 Σ_{k≥1} ‖XΠ_k‖_F,
              ‖Ω̄‖_* = D_Ω / p,   D_Ω := ‖Π_0Ω‖_F + √2 Σ_{k≥1} ‖Π_kΩ‖_F,     ‖W̄‖_F = ‖W‖_F / p  (ν 3).
        N̄ is decoupled by construction (translation-invariant Gram).

THEOREM. For every network N of the family,   γ(N) ≤ L(N) · γ(N̄),   L(N) = L_X L_Ω,
   L_X = D_X/‖X‖_* ≥ 1,  L_Ω = D_Ω/‖Ω‖_* ≥ 1 (ν ≥ 4; L_Ω = 1 at ν 3),
with N̄ decoupled and junk-free. L = 1 iff every slice is a balanced rank-2 block and no rank is shared; for the fully
locked K-comb population L_X = √K exactly; in general L ≤ √(2·nf).
Proof: (ii) for the margin, (iv) for the cost, and D_X ≥ Σ_k ‖XΠ_k‖_* ≥ ‖X‖_* (a rank-2 slice has nuclear ≤ √2 ·
Frobenius; nuclear norm is subadditive).  ∎

COROLLARY. γ(N) ≤ L(N) · Γ_dec(ν), where Γ_dec(ν) is the sup over decoupled junk-free networks. The rank-sharing
restriction of the class theorem is therefore not a restriction on which networks the bound covers; it is an
explicit, per-network computable factor. For the 432 trained nets of H1/H3 it is 1.10 at the median and 1.78 at the
maximum (§6), against sparse-over-dense margin ratios of 1/0.57 (ν 3), 1/0.21 (ν 4), 1/0.02 (ν 6) at p 71, so no
trained net is within reach of a denser optimum than K*.

## 4. The decoupled junk-free class: where the exponent comes from

For N̄ the per-frequency amplitudes are A_k = |ĝ(k)| (same as N) and the costs are D_X, D_Ω, both l1 sums over
frequency slices. If, per frequency,
    (Dil)   Σ_i |ĥ_i(k,k)|² ≤ c² Σ_i (‖u_iΠ_k‖² + ‖v_iΠ_k‖²)/(2p)        (clean energy at k ≤ input energy at k),
then A_k ≤ c ‖XΠ_k‖_F ‖Π_kΩ‖_F (Cauchy–Schwarz over neurons), and with Σ_k ‖XΠ_k‖_F, Σ_k ‖Π_kΩ‖_F fixed the K-set
programme gives A ∝ (D_X/K)(D_Ω/K) at ν 4 and (D_X/K)(‖W‖_F/√K) at ν 3: margin ≤ Γ · max_K M_K K^{−ν/2}, the ladder
K*(p, ν). (Dil) holds for single-frequency neurons (all clean energy at the neuron's own frequency and its harmonics
d_n = (4/π²)(−1)^{n+1}/(4n²−1), Σ_n |d_n|/|d_1| = 1.5): that is the existing class theorem (l2-theorem-status,
relu-theory-gap), which the symmetrisation theorem now extends to any N whose neurons are single-frequency, with
factor L(N).  Without (Dil), only L4 is available: it bounds the TOTAL clean energy by the total input energy, which
is Frobenius-additive, and the same programme comes out with exponent (ν−1)/2 — the "ν_eff = ν − 1" bound (K* 6
instead of 4 at ν 4, p 113). The trained counts sit at the sharp exponent.

## 5. What is open, precisely, and why the attempt stopped

Two per-neuron statements about ReLU's Fourier coefficients, each with sharp constants:

(Dil) for multi-frequency neurons. FALSE as stated per neuron: a neuron at frequency j has clean content at 2j, 3j, …
   (harmonics, 20%, 8%, … of the fundamental), and a neuron with u at k₁ and v at k₂ makes clean content at every
   common multiple. TRUE in the effective-quadratic approximation (relu_dense_check part B: clean at k ≈ u_k v_k /
   (√(2π) R), R = rms of the pre-activation, error < 5%), where the dilution 0.738/√K is exactly what cancels the
   √K nuclear discount at ν 3: the junk-free locked population reaches only 0.80–0.85 of the sparse committee at
   every K and ρ (part E, r3_clean). A proof needs the exact Bessel expansion of the (k,k) coefficient of
   relu(Σ_j r_j cos(j a + φ_j) + Σ_j s_j cos(j b + ψ_j)) with constants sharp to ~15%; the numerically found sup of
   the l2 clean efficiency over all neurons is 0.212 against 0.196 for a single frequency (part D), so the cushion is
   real but thin.

(QJ) quantitative junk. At ν 4 the junk-free locked population BEATS the sparse committee (r4_clean up to 2.2 at
   ρ = 1: both X and Ω get the √K discount, the dilution cancels only one), so no bound that discards junk can work;
   the true locked margins are negative at every ρ > 0 and every K (r4 from −0.4 at ρ 0.1 to −52 at ρ 1, 501-phase
   search included). Mechanism: for a locked population the clean vector (ĥ_i(k,k))_i and every junk vector
   (ĥ_i(k,k′))_i point along the same neuron-space direction e^{i(Φ_i+Ψ_i)} (effective-quadratic law), so no readout
   can cancel the junk without cancelling the clean part; for translation-shifted (decoupled) populations they are
   distinct characters e^{i(ks_i + k′t_i)} and cancel for free — which is exactly why cancellation costs the
   decoupled price. Needed: margin(N) ≤ margin(N̄)/L(N)^κ (or any bound that makes junk pay at least the discount),
   for all (X, Ω), not just the locked family. Proven only locally (ρ = 0 is a strict optimum: cost saving O(ρ²),
   margin loss O(ρ)) and on the exact grid.

Why symmetrisation alone cannot close it: for the locked family L = √K and γ(N̄) is that of a diluted flat committee,
so the bound γ(N) ≤ √K γ(N̄) is loose by exactly the junk it discards. The two lemmas are hard analysis (Bessel
sums with sharp constants; a neuron-space alignment inequality for the ReLU bispectrum), not something a session
proves. Probability of a full proof in a further week: low (and see §5b: the per-frequency lemma is false). What would substitute: a computer-assisted bound on the
per-neuron (k,k) coefficient over a finite frequency set (exact DFTs, a compactness argument in the phases) — an
option if the paper needs "unrestricted" at ν 4 for ReLU; it does not (§7).

### 5b. The attempt on (Dil), 2026-09-17 evening: the per-frequency lemma is FALSE, not merely unproven (dil_test.py)

Exact structure first (L5, verified numerically to 5 digits): with |x| = (2/π)∫₀^∞ (1 − cos tx) dt/t² and
cos t(u+v) = cos tu cos tv − sin tu sin tv, the clean coefficient of one neuron factorises,

    ĥ(k,k) = −(1/π) ∫₀^∞ [ ĉ_u(t,k) ĉ_v(t,k) − ŝ_u(t,k) ŝ_v(t,k) ] dt/t²,

ĉ_u(t,k), ŝ_u(t,k) the k-th Fourier coefficients of cos(t u(·)), sin(t u(·)). Cauchy–Schwarz gives
|ĥ(k,k)| ≤ √(Φ_k(u) Φ_k(v)) with Φ_k(u) = (1/π)∫ (|ĉ_u|² + |ŝ_u|²) dt/t², and Σ_k Φ_k(u) ≤ (2/π) σ_u exactly (the
integrand is ≤ min(1, t²σ_u²)). So the two-variable problem reduces to single-function spectra, and the total clean
amplitude of any neuron is ≤ (2/π)√(σ_u σ_v). That total bound is 2.2× above the single-frequency value, and it is
the only rigorous constant this route yields.

What the ν 3 programme needs is per frequency: |ĥ(k,k)| ≤ C ρ_k(u) ρ_k(v)/σ with C within ~20% of the single-
frequency constant C₁ = 0.135 (the locked comb sits at 0.81–0.85, so a looser C lets combs "win" in the bound).
Exact DFTs at p 113 on 400 random neurons with K = 1..8 frequencies (amplitudes 0.2–1, random phases):

| K | clean amplitude at frequencies with NO input product (harmonics + intermodulation), share of total: median / max | largest leaked frequency vs largest supported: max | direct constant at supported frequencies / C₁: max, min |
|---|---|---|---|
| 1 | 0.17 / 0.32 (harmonics 2k, 3k, …) | 0.20 | 1.00, 0.86 |
| 2 | 0.27 / 0.42 | 0.33 | 0.96, 0.48 |
| 4 | 0.26 / 0.78 | 0.58 | 1.08, 0.52 |
| 6 | 0.27 / 0.83 | 0.41 | 1.33, 0.30 |
| 8 | 0.25 / 0.79 | 0.47 | 1.84, 0.35 |

Up to 86% of a neuron's clean amplitude can sit at frequencies it has no input at, a single leaked frequency can
carry 58% of the largest supported one, and at supported frequencies the constant runs up to 1.84 C₁. The
effective-quadratic law (part B, "< 5% error") holds for the locked comb family it was checked on (direct/C₁ 0.37–0.83,
leak 16–45%) and fails for generic multi-frequency neurons. Consequence: no inequality of the form "clean at k ≤
constant × input at k" is true with a constant anywhere near C₁, so the l1-over-frequencies cost accounting that
produces the exponent ν/2 cannot be justified per neuron for ReLU. The sharp per-frequency route is closed. What
remains true and rigorous: the total-budget bound (exponent (ν−1)/2, the weak bound) and the symmetrisation
theorem of §3.

Whether the unrestricted CLAIM is nevertheless true (no ReLU network with linear layers beats the K* committee) is
now a question about leaky multi-frequency networks, which no trained net uses (freqs per neuron 1.00) and no
construction has exploited; the phase lemma counts leaked amplitude as if phase-aligned, so any bound built on
|ĝ(k)| overcounts it. Settling it would take a margin-ascent search over multi-frequency networks, not a proof.

## 6. Measured L(N) on the trained cohorts (sym_factor.py, 432 nets, H1 p 59–127 ν 3–6 wd 3e-4; H3 p 113 ν 4/6 wd 3e-5..1e-3)

| | L_X median | L_X max | L_Ω median | L_Ω max | L median | L max |
|---|---|---|---|---|---|---|
| ν 3 (5 primes) | 1.05–1.08 | 1.21 | 1 | 1 | 1.05–1.08 | 1.21 |
| ν 4 (5 primes, 4 wd) | 1.03–1.10 | 1.21 | 1.06–1.33 | 1.69 | 1.11–1.38 | 1.78 |
| ν 5 | 1.03 | 1.08 | 1.04–1.08 | 1.57 | 1.08–1.12 | 1.62 |
| ν 6 (4 wd) | 1.03–1.04 | 1.16 | 1.05–1.08 | 1.67 | 1.08–1.13 | 1.73 |

All 432: median 1.10, 90th percentile 1.32, max 1.78. Most of the excess is the balance factor (a slice whose two
singular values differ has ‖·‖_* < √2 ‖·‖_F) and the readout side at large wd (H3 wd 1e-3: L_Ω 1.33), not rank
sharing (rank_share ratio_X 0.91–0.98). So every trained net satisfies the unrestricted inequality
γ(N) ≤ 1.78 · Γ_dec, and the K* argmax survives any factor below the plateau ratio.

## 7. What the paper can say (and cannot)

- Says: ν 2 ReLU: unrestricted, dense (theorem). ν ≥ 3 ReLU: the sparse programme K*(p, ν) is a theorem for decoupled
  single-frequency networks, and every network obeys γ ≤ L(N) Γ_dec with L computable and equal to 1.10 (median) /
  1.78 (max) on all 432 trained nets; junk-free locking cannot beat the committee at ν 3 (exact grid, 0.85), and at
  ν 4 locked populations have negative margin once their junk is counted (exact grid) and ρ = 0 is a strict local
  optimum.
- Does not say: "provably optimal over all ReLU networks with linear layers". The two missing lemmas are (Dil) and
  (QJ) above.
- Framing: "unrestricted for the activation-degree family (THEORY_nsum), unrestricted up to an explicit coupling
  factor for the linear-layer ReLU family, with the factor measured at 1.1 on trained networks".
