# Global-optimality programme: full state as of 2026-09-20 evening

Shared briefing for the parallel proof agents. Read all of it, then the inherited write-ups it points to.

## Why this matters
ICLR 2027 paper "Explaining Feature Sparsity Through the Margin–Regularization Trade-off", full paper due 2026-09-25. Paper logic: (i) a theorem locates the max-margin network and its feature count K*; (ii) exact closed-form evaluation of constructed networks exhibits it; (iii) Wei et al. 2019 Thm 4.1 (global minimiser of the regularised loss → max normalized margin as λ→0) puts training dynamics out of scope, as Morwani et al. 2311.07568 do; (iv) empirics are trends only. Weak link = (i) for q<ν: unrestricted UPPER bound with argmax K*, a construction attaining a constant fraction, no proof the construction is globally optimal.

## Setting
- Task: full-table n-operand modular addition mod prime p (all p^n rows, label = sum mod p).
- Network ("gated chain"): one hidden layer, no biases; neuron i has operand embeddings E_j[i,·] ∈ R^p (j=1..n), readout W[·,i] ∈ R^p. Logit z_c(x) = Σ_i W[c,i] Π_j E_j[i,x_j]. Homogeneity degree ν = n+1.
- Penalty P_q(θ) = Σ_i ‖θ_i‖^q (Euclidean norm of all of neuron i's weights). q=2 plain weight decay; q=ν Morwani's L_{2,ν}. Range 2 ≤ q ≤ ν. r := q/ν ∈ (0,1].
- Normalized margin γ_q = margin / P_q^{ν/q}; margin = min over rows and wrong classes of (z_correct − z_wrong).
- Programme, s = ν/q: Γ_s(p) = max over shares b_k ≥ 0, Σ b_k = 1, of min_{D≠0} Σ_k b_k^s (1 − cos 2πkD/p). K* = number of non-zero shares at the optimum.
- TARGET (T): every network, 2 ≤ q < ν: γ_q ≤ (n+1)^{−(ν−q)/q}·B·Γ_{ν/q}(p), equality at the "(n+1)-neuron common-phase block" on the programme's optimal support (`build_chain` in `fvf_obj.py`: per frequency, neurons l=0..n, operand phase πl/(n+1), readout phase n·πl/(n+1); exactly junk-free). B explicit in THEORY_nsum.md.

## The atom picture (why q<ν is hard)
One neuron at frequency k emits the clean term (depends on the sum of operands) AND 2^n−1 junk terms (sign patterns T≠∅) at equal size. For a population at one frequency with complex atoms c_l and unit phases v_lj:
  clean total C_∅ = Σ_l c_l = A;  junk totals C_T = Σ_l c_l Π_{j∈T} v_lj, T ≠ ∅.
System (KL): C_T = A·[T=∅]. Cost Σ_l |c_l|^r. At r=1 splitting is free; at r<1 it is concave, so splitting to cancel junk costs. Block: n+1 atoms, (n+1)-th roots of unity, equal moduli, cost (n+1)^{1−r}|A|^r.

## Inherited write-ups (all in heracles/)
THEORY_nsum.md (§1 L1–L3, §3, §5, §6, §8), PROOF_AUDIT.md, LEMMA_ATTEMPT.md (agent 1, lemma_*.py), GLOBAL_OPT.md (agent 2, gopt_*.py), GLOBAL_OPT3.md (agent 3, gopt3_*.py), GLOBAL_OPT4.md (agent 4, gopt4_*.py). Outputs in FINAL_RUNS/{LEMMA,GOPT,GOPT3,GOPT4}/.

## Proved
- Unrestricted upper bound γ_q ≤ B·Γ_{ν/q} (junk lemma by orbit averaging → per-frequency AM-GM → budget). Two-sided at q=ν only. Block attains (n+1)^{−(ν−q)/q} of it.
- KEY LEMMA (N = n+1 exactly): (KL) ⇒ Π|c_l| ≥ (|A|/(n+1))^{n+1} ⇒ Σ|c_l|^r ≥ (n+1)^{1−r}|A|^r; equality iff rotated roots of unity, equal moduli. Proof: (KL) ⇔ Σ_l c_l Π_j(u_j − v_lj) = A Π_j u_j; substitute u_j = v_{σ(j),j} for bijections σ onto atoms ≠ l₀; geometric mean over n! bijections; product over l₀ → squared Vandermondes; Hadamard. (Hand-verified.)
- LEMMA Q (unrestricted): margin ≤ mean_a φ_D(a) − |ψ̂_D(m)|, any D≠0, mode m≠0. (Hand-verified.) Pure class: margin ≤ 2^{-n}[|C_∅|(1−cos θ_D) − |C_T||sin(θ_D/2)|].
- Agent 2: exact identity, single-frequency population, h(u) = Σ_l c_l Π_j(1+u_j v_lj): 2^n·margin = min_u [M_1 Re h(u) − sin(2π/p)|Im h(u)|]. Thm A (N ≤ n ⇒ margin ≤ 0, junk allowed). Thm B (N = n+1, junk allowed: Σ|c_l|^r ≥ (n+1)^{1−r}max(R,0)^r, R = min over torus of Re h). Thm C / Lemma C3 (K=1 with junk: (T) exact for p ≥ 23/47/97, 1+O(p⁻²) below). Lemma C4 rigidity. Lemma P (product test measures).
- Agent 3: THEOREM D — ANY chain network (any width, impure neurons, any K); atoms c_ik = Π_j ê_ij(k)·conj(ŵ_i(k)), v_ijk = conj(ê_ij(k))/ê_ij(k); assuming only (H): same-|k| junk J_{k,T} = 0 for all k, T≠∅: γ_q ≤ F·(n+1)^{−(ν−q)/q}·B·Γ, F = 1.000905 (ν3,q2), 1.030841 (ν4,q2), 1.101809 (ν5,q2), 1 at q=ν. Engine Thm E (every N): sorted moduli x_1 ≥ x_2 ≥ …: Σ_{l>h} x_l ≥ D_h(n)|A| via a multi-affine test polynomial killing the h largest atoms (κ_1=1/2, κ_2=3√3/8, κ_3=49/72) + exact vertex enumeration. Purity never needed (atoms = neurons' Fourier data). Peeling CANNOT be sharp (best 1.000521 at ν3,q2).
- Agent 4: Prop D1 exact conic duality: min{max_l |c_l|/x_l : (KL)} · min{Σ_l x_l|Λ(v_l)| : Λ multi-affine, λ_∅=1} = 1; a modulus profile x is realisable iff Σ_l x_l|Λ(v_l)| ≥ |A| for every normalised Λ; phases of c_l carry no information; peeling's loss = replacing |Λ(v_l)| by ‖Λ‖_∞. Theorem O (barrier): for every probability measure π on normalised Λ, sup_v E_π|Λ(v)|^r ≥ 1 — "one averaged test measure + r-subadditivity" can never beat the trivial bound. Theorem J: junk cost relief is QUADRATIC: Cl2_r(η) := min{Σ|c_l|^r : C_∅=1, Σ_{T≠∅}|C_T|² ≤ η²} = (n+1)^{1−r}(1 − βη² + O(η⁴)), β = λ_n(r)/(n+1), λ_n(r) = r(1−r)/[1 − ((n−1)/(n+1))²(1−r)]; rational-certified at n=2 (λ = 3/13 at q=2). With Lemma Q's linear penalty ⇒ (T) on the pure class for junk below η_cross (ν3,q2: 0.672/0.584/0.452/0.407/0.319 for K=2..6) — CONDITIONAL on Cl2_r's measured (SLSQP) global minimum.

## The three open problems
P1 (SHARP CONSTANT): for every N ≥ n+1 and every (KL) system, Σ_l|c_l|^r ≥ (n+1)^{1−r}|A|^r. First open case n=2, N=4: four torus points (a_l,b_l); c = unique Cramer solution of Σc=1, Σca=0, Σcb=0, Σcab=0 (ratios of 3×3 minors of rows (a;b;ab)); claim Σ|c_l|^r ≥ 3^{1−r}, equality iff one c_l = 0 and the other three at cube roots of unity. True at r=1 (triangle inequality) and in the r→0 limit (≥3 non-zero atoms, Thm A). Numerics: min/target = 1.0000009 (q=2, 500 starts). For N>4 at n=2 the system is underdetermined (choose best c for given phases). Equivalent (agent 2): optimal dual Λ_μ ≡ 1.
P2 (JUNK COST CURVE): prove a global lower bound Cl2_r(η) ≥ (n+1)^{1−r}(1 − βη² − Cη⁴) or any explicit proven minorant good enough to keep the trade against Lemma Q's linear penalty up to a useful η. Only the local second-order expansion is proved.
P3 (LARGE JUNK, K ≥ 2): remove hypothesis (H) entirely. Obstruction: margin is a min over rows; all frequencies' polynomials are evaluated at the same row (min of sum ≠ sum of mins); Lemma Q tests one mode at a time so cannot add damage across K frequencies; same-|k| junk of frequency k lives at modes with entries ±2k/0 (distinct across k); only (p−1)(2^n−1)/2 junk coefficients matter. At the top of the junk range the current best bound is V/target = 1.30 (p=11,13). Agent 4 flagged hypercontractivity on Z_p^n as the most promising route; a variance/fourth-moment junk lemma is provably ≤ half of Lemma Q.

## Dead routes (with breaking points) — do not repeat
- Morwani-style dual certificate: lossy step Σ_i τ_i^ν ≤ (Σ_i τ_i^q)^{ν/q} loose by exactly N^{(ν−q)/q}; cannot certify any constant < 1.
- Domination lemma Σ|c| ≥ (n+1)max|c|: refuted (max/sum → 1/2 at n=2).
- Capped-mass / weak-majorisation conjecture Σ_l min(|c_l|,θ) ≥ min((n+1)θ,|A|): FALSE (0.98285 at n=2, θ=1/3). So no r-free majorisation argument can work: the proof must use r.
- Peeling with sup-norm test polynomials: capped at F>1 (true D_1 ∈ [0.6495,0.6536] < 2/3, D_2 = 0.2599 < 1/3 at n=2).
- One averaged test measure (Theorem O); product test measures (Lemma P gives only the grid's convex hull).
- Gauge symmetrisation: circular (cost (n+1)^{1−q/ν} cancels the target factor; controls min over orbit, need sup). Grid rotation: same.
- Induction on n by shifting the last slot: impossible for every n (Jensen on log|1+λv|).
- Gram/Carathéodory–Fejér (loose by 4 at n=2); Mahler measure for N>n+1; Hadamard crowding for N>n+1 (loses > the 10% it needs).
- Carathéodory support counting: (KL) has 2^n complex constraints, gives ≤ 2^n or 2^{n+1}−1 atoms, not n+1.

## Evidence (T) is true
Whole-network smooth NLP over raw weights, no restrictions, p=5,7 (n=2), p=5 (n=3): ratio 1.000000000000 in 16/16 cells, never above. Exact-margin search, pure neurons, n+2 atoms/frequency, junk allowed, p=11..23, K=2..4: best 1.000000000000; forced junk lowers the ratio monotonically to ≤ 0 at ρ=0.8. Local escape from the block never succeeds. Free CE+weight-decay training lands at 0.4–0.6 with 40–60% junk but at strictly higher training objective than the construction (local non-global minima, not counterexamples).

## Rules for every agent
- Code: fvf_obj.py, nu4_program.py, kstar.py/kstar.json, lemma_*.py, gopt_*.py, gopt3_*.py, gopt4_*.py. Run with `uv run python ...`.
- LOCAL CPU ONLY. No pods.
- Do not edit any existing file; do not commit. Write only the new files named in your own brief. Other agents are working in parallel on sibling problems — do not touch their files.
- No background processes left running at hand-back.
- Never call something proven unless every step is written out and fp64-checked. A verified counterexample or a precise "this route dies here" is a valid result. If you find an error in inherited material, say so.
- Write-up style: define every symbol, plain-language reading with each formula.
- Final message: short summary of exactly what is proved/refuted, status (PROVEN / REFUTED / PARTIAL / OPEN), and confirmation that no background jobs remain.
