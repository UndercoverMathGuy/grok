# PROOF_AUDIT — hostile referee pass on the theory behind the ICLR abstract (2026-09-18, local CPU only)

Scope: heracles/THEORY_nsum.md (all, §5–§6 first), notes/l2_theorem.md, heracles/THEORY.md, heracles/THEORY_relu.md,
heracles/ABSTRACT_draft.md. Scripts (all exact float64, no training cohorts touched): `audit_common.py`,
`audit_chain.py`, `audit_construct.py`, `audit_program.py`, `audit_program_big.py`, `audit_ascent.py`,
`audit_relu_nu2.py`, `audit_sym.py`; logs in `heracles/logs/audit_*.log`, JSON next to the scripts.

Verdict in one paragraph. Every inequality in the chain (junk lemma, phase lemma, assembly, per-frequency AM-GM,
Parseval, the q-budget for 2 ≤ q ≤ ν, the ν 2 ReLU bound, the symmetrisation theorem) is correct; I could not break
any of them analytically or numerically. What is broken is what the documents SAY the chain proves. The chain is an
upper bound only; it is two-sided only at q = ν. For q < ν the bound is provably not attained, the attainment
constants in the docs are wrong for n ≥ 3, and with the docs' own constructions the theorem cannot even exclude the
flat (Morwani) solution as the optimum. The equal-amplitude formula M_K/K^{ν/q} is not the bound (the free-amplitude
programme is 2–10% larger and sometimes has a different support). "ν 2 ReLU dense — theorem", "q < 2 provably
reverts to the q = 2 count" and "extends to ReLU with linear layers up to a coupling factor" are each stated more
strongly than proven.

Notation used below. p prime, nf = (p−1)/2. g_k(D) = 1 − cos(2πkD/p). a_k = |ĝ(k)| = amplitude of frequency k in
the clean logit g(D). m_ik = Σ_j |ê_ij(k)|² + |ŵ_i(k)|² = squared-weight mass neuron i spends on frequency k
(Parseval: Σ_{k≥1} m_ik ≤ ‖θ_i‖²/(2p)). P_q = Σ_i ‖θ_i‖^q, the penalty; normalised margin γ_q = margin / P_q^{ν/q}.
c_n = κ (n+1)^{−(n+1)/2} with κ = n! for x^n and κ = 1 for the gated chain. Γ_s(p) = max{ min_{D≠0} Σ_k x_k g_k(D) :
Σ_k x_k^{1/s} ≤ 1, x ≥ 0 } is the FREE-amplitude programme with exponent s = ν/q; the docs' ladder is its
equal-amplitude restriction max_K M_K K^{−s}. The bound the chain actually proves is

    γ_q ≤ 2 c_n (2p)^{−ν/2} Γ_{ν/q}(p)   for 2 ≤ q ≤ ν,     γ_q ≤ 2 c_n (2p)^{−ν/2} Γ_{ν/2}(p)   for q < 2.

---

## Findings, worst first

### F1. The theorem is an upper bound; "the maximiser uses K*" is not proven, and the stated attainment [1/4, 1] is wrong for n ≥ 3 — GAP + WRONG constant
Where: THEORY_nsum §3 ("the maximum-margin frequency count is K* … for EVERY network in the family", "The constant
sits in [1/4, 1] as at n = 2"), §6 ("THEOREM, unrestricted, K*(p, ν, q) = argmax…"), §7 ("Says: the plain-weight-decay
maximum-margin solution uses K*(p, ν) frequencies"), RESULTS_PROD header ("Exact for every n and every q"),
abstract v1 ("the ℓ₂ maximum-margin solution … uses K*"). Abstract v2's wording guard is correct; the theory files are not.

Why. Splitting one frequency over N neurons at fixed clean amplitude multiplies P_q by N^{1−q/ν}, i.e. multiplies γ_q
by N^{−(ν−q)/q}. A single neuron per frequency has 75–89% junk and negative margin (`audit_construct.py`, last two
lines), so N ≥ 2 is forced and the bound is strictly unattained for every q < ν. Exact attainment constants of
junk-free constructions (junk energy < 1e-28, γ/bound matches N^{−(ν−2)/2} to 4 digits, `audit_construct.py`):

| family | doc's construction | γ/bound (q=2) | best I found | γ/bound | flat-spectrum UPPER bound / Γ* (kstar.json, p71 / p113) |
|---|---|---|---|---|---|
| chain n=2 (ν3) | 2^n = 4, phases {0,π/2} | 0.500 | 3 neurons, common phase πl/3 | 0.577 | 0.571 / 0.507 |
| chain n=3 (ν4) | 8 | 0.125 | 4 neurons, common phase πl/4 | 0.250 | 0.213 / 0.153 |
| chain n=4 (ν5) | 16 | 0.0156 | 5 neurons | 0.0894 | 0.068 / 0.041 |
| x² (ν3) | 16 (4×4 grid) | 0.250 | 6 neurons, (φ₁,φ₂) = 2πl(1,4)/6 | 0.408 | 0.571 / 0.507 |
| x³ (ν4) | none given | — | 18 neurons, 2πl(1,3,12)/18 (smallest cyclic family) | 0.056 | 0.213 / 0.153 |
| x^n tied, n ≥ 2 | "exists" (nsum.py docstring) | — | no phase population works: junk 33% (x²), 61% (x³, margin < 0) for every G | — | — |

Consequences. (a) With the docs' constructions, lower bound at K* < upper bound of the flat solution in EVERY cell
(0.5 < 0.57, 0.125 < 0.21, 0.0156 < 0.068, 0.25 < 0.57): the theorem as documented does not exclude that Morwani's
dense solution is the ℓ₂ max-margin solution. (b) With the (n+1)-neuron common-phase population (NEW, not in the
docs: every operand of neuron l gets phase φ_l = πl/(n+1), readout phase nφ_l; the term with flipped set S carries
e^{−2i|S|φ_l}, which sums to zero over l for 1 ≤ |S| ≤ n) the chain family does exclude the flat solution, barely at
p 71 (0.577 vs 0.571; 0.250 vs 0.213; 0.089 vs 0.068) and clearly at p 113. For x², Morwani's own network, 0.408 <
0.507: not excluded. For x³: not excluded. (c) Even in the best case the certified set of optimal supports is
{S : Γ_S ≥ c·Γ*}: at ν3 p113 with c = 0.577 that is every K from 3 to far beyond 12 (ladder r(12) = 0.88); at ν4 with
c = 0.25 it is K = 2 … > 12. The theorem pins a plateau, never K*.
Direct margin ascent (`audit_ascent.py`, chain n=2, p 11, width 40) supports c being the truth rather than slack in my
constructions: best of 5 runs γ/bound = 0.5775 at q = 2 (= 3^{−1/2}, junk 0.000, 6 live neurons = 2 frequencies × 3)
and 0.8027 at q = 2.5 (= 3^{−(ν−q)/q} = 3^{−0.2}, 15 neurons = 5 × 3); junk-tolerant local optima were worse
(0.44–0.48). Nothing exceeded the bound.

Fix. State: "THEOREM (upper bound). γ_q ≤ C Γ_{ν/q}(p), attained iff q = ν. PROPOSITION (lower bound, chain family):
the (n+1)-neuron population attains (n+1)^{−(ν−q)/q} of it on any support, hence the flat solution is not max-margin
for p ≥ 71 and the optimal support lies in {Γ_S ≥ (n+1)^{−(ν−q)/q} Γ*}." Replace 2^n by n+1 in §6. Delete "[1/4,1] as at
n = 2" for n ≥ 3. Remove "the maximum-margin solution uses K*" everywhere; "CONJECTURE: sup = (n+1)^{−(ν−q)/q} × bound
for the chain" is supportable by the ascent.

### F2. The equal-amplitude ladder is not the bound — WRONG as an inequality, SOUND-WITH-FIX
Where: THEORY_nsum §3 "margin ≤ C′‖θ‖^ν · max_K M_K K^{−ν/2} (equal amplitudes; free amplitudes change the argmax by
≤ 1 rung)"; l2_theorem §3 "free amplitudes do not improve it"; abstract "functional whose maximiser uses K* =
argmax_K M_K/K^{ν/q}".
What the proof gives is Γ_s with free amplitudes. `audit_program.py` (exhaustive equal ladder + 4000-start free
search + SLSQP) and `audit_program_big.py` (p 71/113, warm-started from kstar.json sets; lower bounds on Γ_free):

| p | s=ν/q | equal-ladder K* | Γ_free / ladder max | free support size |
|---|---|---|---|---|
| 23 | 1.25 / 1.5 / 2 / 2.5 | 6 / 4 / 3 / 2 | 1.104 / 1.077 / 1.029 / 1.019 | 7 / 4 / 3 / 2 |
| 31 | 1.25 / 1.5 / 2 / 2.5 | 6 / 4 / 3 / 3 | 1.067 / 1.038 / 1.089 / 1.085 | 7 / 4 / 3 / 3 (different sets at s ≥ 2) |
| 47 | 1.25 / 1.5 / 2 / 2.5 | (≤6 searched) 5 / 5 / 3 / 3 | 1.065 / 1.015 / 1.059 / 1.044 | 7 / 5 / **4** / 3 |
| 71 | 1.5 / 2 / 2.5 | 5 / 4 / 3 | ≥0.975 (no improvement found) / ≥1.049 / ≥1.046 | 5 / 4 / 3 |
| 113 | 1.5 / 2 / 2.5 | 6 / 4 / 4 | ≥0.992 / ≥1.039 / ≥1.025 | 7 (near-tie) / 4 / 4 |

So a network is NOT bounded by C′ max_K M_K K^{−s}; the right-hand side must be Γ_s(p), which is up to 10% larger, and
the count of the bound's maximiser differs from the ladder argmax in some cells (p 47 s 2: 4 vs 3; p 23/31 s 1.25: 7
vs 6). Since adjacent ladder rungs at the paper's primes differ by 1–3% (p 113 ν 3: K = 5, 6, 8 at 0.974, 1, 0.986), a
support-dependent 2–10% correction is larger than the gaps that define "K*". No network can exhibit the violation
today only because of F1's attainment gap. Fix: define K* as the support size of the maximiser of Γ_s, compute it
(nonconvex, but small), and present M_K K^{−s} as its equal-amplitude approximation with the measured discrepancy.

### F3. "ν 2 ReLU: unrestricted, dense — THEOREM" — upper bound SOUND, density GAP
Where: THEORY_nsum §2 + §7, THEORY_relu §2 + §7, abstract v1; abstract v2 guard cites "Morwani q = ν theorem".
The inequality margin ≤ ‖θ‖²/(2(p−1)) is correct (1-Lipschitz ⇒ Var σ(X) ≤ Var X; independent operands; diagonal
±k Parseval factor ½; Cauchy–Schwarz; AM-GM; Σ_{D≠0} g_k(D) = p — all checked), and the flat spectrum is the UNIQUE
maximiser of the bound (a function constant on D ≠ 0 has equal Fourier coefficients). But "dense is max-margin"
needs (upper bound on every sparse support) < (what a dense net attains). Dense ReLU nets attain η ≈ 0.196 of the
bound (the doc's own number; trained nets 0.17). The rigorous bound for a support S is V_S/V_full of the bound with
V_S the free-amplitude LP: `audit_relu_nu2.py`, p 23: K = 2: 0.41, K = 4: 0.76, K = 8: 0.94, K = 10: 0.955. All exceed
0.196, so the theorem excludes only K = 1. Importing the NUMERICAL per-neuron sup 0.212 (relu_dense_check part D)
the test becomes V_S/V_full < 0.925, which still fails to exclude K ≥ 8 of 11. "Only the ReLU efficiency constant is
lost; the count conclusion is not" is therefore unproven: the constant is exactly what decides the count.
Morwani's theorem is for x² under L_{2,3}; it says nothing about ReLU. Fix: say "the bound's unique maximiser is the
flat spectrum; trained nets are dense (measured)". A real proof would run Morwani's certificate at b = ν = 2 (their
combination step IS valid there) with the single-ReLU-neuron weighted-margin maximiser, which nobody has computed.

### F4. "q < 2 provably reverts to the q = 2 count" — bound SOUND, claim about the count GAP
Where: THEORY_nsum §5 second bullet ("The unrestricted argmax is K*(p, ν/2) for every q < 2"), abstract v2.
Proven and verified (`audit_chain.py`): for q < 2, Σ_i M_i^{q/2} ≤ C ⇒ Σ_i M_i ≤ C^{2/q} (concavity), so the ℓ₂
programme bounds γ_q; and the single-frequency-class budget Σ_k a_k^{q/ν} ≤ c P_q genuinely FAILS for q < 2 (one
balanced neuron carrying all nf frequencies: lhs/rhs = nf^{1−q/2}: 1.50 at q 1.5, 2.24 at q 1, p 11; exactly 1.000
at q 2, < 1 for q > 2). So the doc's validity range 2 ≤ q ≤ ν is right and tight. But the bound's maximiser at q < 2 is
ONE neuron with every frequency, which has margin < 0 (same script: −1.4 vs clean +1.4); nothing near the bound
exists, and margin ascent at q = 1.5 reached only 0.14–0.24 of it. An unattained bound's argmax is not a prediction
of the count; the honest sentence is "for q < 2 this proof technique gives nothing sharper than the q = 2
programme". The abstract sentence must go or be reworded (the doc's own note already says "Do NOT claim q < 2").

### F5. ReLU + linear layers: symmetrisation THEOREM SOUND; corollary and abstract sentence OVERCLAIMED; (Dil) mis-stated
Where: THEORY_relu §3 corollary, §4, §6 last sentence, §7; abstract v2 "extend it to ReLU networks with linear layers
up to an explicit coupling factor".
- §3 theorem: SOUND. Algebra re-derived (X̄ᵀX̄ = p Σ_s T_sᵀXᵀXT_s, circulant ⇒ 2×2 blocks (‖XΠ_k‖²_F/2)·I scaled by p²;
  Ω̄ likewise) and checked independently by explicit stacking at p 7 (`audit_sym.py`: N̄ logits = clean projection to
  4e-16, ‖X̄‖_* = pD_X and ‖Ω̄‖_* = D_Ω/p to 1e-14, margin(N̄) ≥ margin(N)). Cost formulas γ₃, γ₄ (AM-GM 2x + y² ≥ 3(xy)^{2/3},
  2x + 2y ≥ 4√(xy)) correct. Needs d, r ≥ p and width p²m for N̄: say "unbounded width".
- Corollary: γ(N) ≤ L(N)·Γ_dec has content only if Γ_dec is bounded by the sparse programme. §4 needs (Dil) for that and
  §5b shows (Dil) is false. So for general N the chain is γ(N) ≤ L(N)·(unknown). It is a theorem only for N whose
  neurons are single-frequency in u, v AND ω (then N̄ is too, and (Dil) follows from L4 with c = 1). The abstract's
  sentence drops that hypothesis.
- §4 says "(Dil) holds for single-frequency neurons"; §5 says a neuron at j has clean content at 2j, 3j (20%, 8%). As
  literally written (clean energy at k ≤ c² × input energy at k) (Dil) FAILS at k = 2j for a single-frequency neuron.
  It is rescued only because ĝ(2j) also needs ω̂_i(2j) ≠ 0: the class must be defined with a single-frequency READOUT,
  otherwise harmonics buy amplitude at 2j for the price of one factor (readout only), a cheaper exponent than ν/2.
- "the K* argmax survives any factor below the plateau ratio" (§6): false as worded. A factor L = 1.10 leaves every
  support with Γ_S ≥ Γ*/1.10 un-excluded: at ν 3, p 113 that is K = 5…10 (ladder r ≥ 0.909); at L = 1.21 it is K = 4…12 and beyond. The factor excludes
  the flat solution at ν ≥ 4 (1.78 < 4.7) and, at ν 3, only because L_max = 1.21 < 1.75. It never pins K*.
- dil_test.py: the script does what §5b says (exact DFTs; leakage and constant ratios). Verdict on "(Dil) is false": SOUND.

### F6. The junk lemma is a full-table statement; most of the "6,000 networks" are not full-table — GAP in scope
Where: THEORY_nsum L1, l2_theorem §2, abstract last sentence.
L1 averages z over the translation orbit of a row; that needs every row of the orbit to be in the set over which the
margin is taken. Wei et al. / Lyu–Li maximise the TRAINING-set margin. With train fraction < 1 the orbit of a training
row contains test rows with unconstrained margins, so margin_train(z) ≤ margin(g) does not follow (and margin_train
can exceed the bound's value in principle). The nsum/prod/QM/Q runs are full-table (fine); the heracles/nudial/S5/Z
cohorts (frac 0.3–0.8) are outside the theorem. The paper must scope the theorem to full-table training and call the
rest empirical.

### F7. Tied embeddings and constants — SOUND-WITH-FIX
Where: nsum.py docstring ("the clean construction exists with tied embeddings, see THEORY_nsum.md" — it is not in
THEORY_nsum), RESULTS_PROD ν 5 cell and RESULTS_NS ν 4/5 cells (tied).
With one shared embedding the cost is ‖e_i‖² + ‖w_i‖², AM-GM is weighted, and c_n becomes κ n^{n/2}(n+1)^{−(n+1)/2}; the
exponent and hence the programme are unchanged (L1–L3 and B_q verified on tied nets in `audit_chain.py`). But the
phase-population junk cancellation is impossible for tied x^n: the u²w-type term has phase (t₁−1)φ + (t₂−1)φ = 0 for
every φ (junk 33% at x², 61% and negative margin at x³, any population size). Tied chain is fine (common-phase
population is already tied). So for the tied x^n cells there is no lower bound at all.

### F8. Items that survived — SOUND
- L1 junk lemma (group averaging; min ≤ mean), L2 phase lemma (needs the D ↔ −D averaging of l2_theorem §6, which is
  correct; the pointwise inequality is false, so keep that proof, not the one-line version), L3 assembly (relerr
  1e-15 on random nets with DC, tied, multi-frequency), only-multilinear-monomial argument for x^n (DC components and
  k = 0 cannot contribute; any p > n+1), per-frequency AM-GM, Parseval with the ±k factor 2, p odd.
- Budget for 2 ≤ q ≤ ν: a_k^{q/ν} ≤ c^{q/ν} Σ_i m_ik^{q/2} (subadditivity, q/ν ≤ 1) and Σ_k m_ik^{q/2} ≤ (Σ_k m_ik)^{q/2}
  (superadditivity, q/2 ≥ 1). Both directions checked; max lhs/rhs over 40 random/adversarial nets = 1.000 (attained by
  the balanced multi-frequency neuron at q = 2, so the constant is sharp).
- q = ν: flat LP, value p/(p−1), attained exactly (splitting free) — the only two-sided theorem in the package.
- q > ν: unbounded only with unbounded width; at fixed width m it is a finite width-limited programme. Say so.
- Wei et al. Thm 4.1 applies: (Σ_i‖θ_i‖^q)^{1/q} is a norm for q ≥ 1, r = q, f ν-homogeneous (no biases). Lyu–Li does
  NOT apply for q ≠ 2 (GD's implicit geometry is ℓ₂; only the global-minimiser statement is available), and at q = 2
  only to KKT points. THEORY.md's elementwise ℓ_q is a different object from THEORY_nsum's per-neuron q and is already
  withdrawn; do not let the two "q" share a symbol in the paper.
- Gated-chain junk cancellation with {0, π/2}^n: exact (junk 1e-30), cost as claimed (γ/bound = 2^{−n(ν−2)/2} exactly).
  Not minimal (F1).
- Margin ascent never exceeded the bound in any cell (table below).

### F9. Evidence-vs-theory mismatches to keep out of the abstract
- ν 4 and ν 5 trained nets (both exact families) have NEGATIVE margins (−0.5…−8) and 25–55% junk; a max-margin
  theorem is not in its regime there. "measured in the predicted plateau" with plateaus 4..16 of 35 (ν 3, p 71) or
  3..7 is a weak test; say "consistent with".
- l2_theorem §3 "K*(p) ~ 4 log p/(ν−2) already proven" was not re-checked here.
- l2_theorem §7 "no junk-free block with n_k ≤ 5 … (exhaustive on the π/4 grid)" is consistent with my 6-neuron block
  (phases on the π/3 grid); the "n_k = 16 is the smallest" reading elsewhere (memory note, THEORY.md §2 "n_k^{1/3}")
  should be updated to 6.

---

## Numerical attacks run

| script | what | outcome |
|---|---|---|
| audit_chain.py | L1, L2, L3, B_q on 40 nets (p 5–13, n 2–4, x^n and chain; gaussian+DC, tied, one-neuron-all-frequencies), q ∈ {1, 1.5, 2, mid, ν} | no violation for q ≥ 2 (max 1.000, sharp); B_q violated for q < 2 by nf^{1−q/2} as the doc says |
| audit_construct.py | junk-free constructions, exact junk + γ/bound | doc's 2^n chain and 16-neuron x² confirmed; cheaper n+1 (chain), 6 (x²), 18 (x³) found; tied x^n impossible by phases |
| audit_program.py / _big.py | free-amplitude programme vs equal ladder, p 23–113 | Γ_free/ladder = 1.02–1.10 (p ≤ 47), ≥ 1.03–1.05 at p 71/113 for ν 4, 5, no gain found at ν 3; support differs in 3 cells |
| audit_ascent.py | direct margin ascent (annealed softmin / P_q^{ν/q}), p 11 n 2 chain and x², p 7 n 3, q ∈ {1.5, 2, 2.5, ν} | see table; bound never beaten; best = the (n+1)-neuron construction value exactly |
| audit_relu_nu2.py | LP coverage V_S/V_full per K, p 23 | 0.41 (K 2) … 0.955 (K 10): ν 2 bound excludes only K = 1 rigorously |
| audit_sym.py | explicit N̄ stacking p 7 | THEORY_relu §3 (iii), (iv) exact to 1e-14 |

Ascent results (γ/bound; bound uses Γ_free; 4 random inits + 1 warm start from the cheapest junk-free construction):

| cell (p 11, n 2, width 40) | q | bound's Γ | best γ/bound | what the best run is | other runs |
|---|---|---|---|---|---|
| chain | 1.5 | Γ_{1.5} (ℓ₂ programme) | 0.237 | warm start, 2 freqs × 3 neurons, junk 0 | 0.14–0.16, junk 23–53% |
| chain | 2 | Γ_{1.5} | 0.5775 = 3^{−1/2} | 2 freqs × 3 neurons, junk 0.000 (found from RANDOM init too) | 0.44–0.48 with junk 36–39% |
| chain | 2.5 | Γ_{1.2} | 0.8027 = 3^{−0.2} | 5 freqs × 3 neurons, junk 0 | 0.743–0.764 |
| chain | 3 = ν | p/(p−1) | 1.0000 (5/5 runs) | flat spectrum, junk 0: bound attained, two-sided at q = ν | — |
| x² | 1.5 | Γ_{1.5} | 0.118 | junk 30–48% | 0.098–0.103 |
| x² | 2 | Γ_{1.5} | 0.386 | junk-tolerant (junk 17%), below the 6-neuron junk-free value 0.408 on the best support | 0.371–0.378 |

x² at q 2.5 / 3 and the n = 3 cells (p 7) were NOT run to completion: the job was killed at the coordinator's request
(laptop overloaded), so `audit_ascent.json` was never written; the six finished cells are in
`heracles/logs/audit_ascent.log`. To rerun lightly: one cell at a time, `OMP_NUM_THREADS=1 nice -n 19`, steps 4000.
No finished cell has γ/bound > 1.
Caveat: p 11 has nf = 5 and a flat programme (equal ladder argmax = 5 = dense at s 1.5 while Γ_free sits on a 2-set
within 0.1%), so these runs test the INEQUALITY and the attainment constant, not the count.
