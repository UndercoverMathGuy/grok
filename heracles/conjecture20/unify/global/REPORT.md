# global: report (from the agent's structured output; the harness blocks subagents from writing report files)
## Headline
Global optimality, used across the whole family b∈[2,ν] with "temperature" s=ν/b, gives one exact identity: the slope of log γ*(s) is minus the cost-share entropy of the global maximisers. Two consequences follow. (i) The IF value, and the single-feature price, can first fail only through a concentrated winner whose cost-share entropy is below the IF optimum's; the certified p=3 counterexamples are exactly such winners. (ii) The count (C) needs value information only up to an explicit programme tolerance ε_tol(s): ≥28% at p=5, b=2 on the n=2 range; infinite at p=3 and in K*=1 regimes; zero at every transition s_c, where (C) forces an exact value identity. So at every p≥5 that has a transition in range, the count cannot be separated from the price–programme value: this holds for all p≥11 (verified for 11≤p≤199) and for p=5,7 when n≥3. Numerically: (C) holds on both sides of the six transitions searched (four on the n=2 and n=3 ranges, two sampled on one side only at p=11, n=2), and single features are direction-uniformly priced at IF. At (5,2) and (7,2), (C) for all b reduces to a compensation bound with 28%/2% slack, and compensation is found to gain exactly nothing.

## Principle
THE TEMPERATURE PRINCIPLE. Treat s=ν/b∈[1,ν/2] as a temperature.

Basic identity. For a fixed network,
  log γ(θ;s) = log h − s·log Σ_i (‖ω_i‖^ν)^{1/s}.
This is a Gibbs free energy:
- the states are neurons, with energies −log‖ω_i‖^ν;
- the Gibbs weights are the cost shares w_i = ‖ω_i‖^b / Σ_j‖ω_j‖^b.
Then
  d/ds log γ = −H(w),  d²/ds² log γ = −Var_w(log w)/s.
γ*(s) is the upper envelope of these concave curves. s=1 (b=ν) is the solved convex end. The programme has the same structure: d/ds log Γ_s = −H(z*), and the IF optimum has entropy log ν + H(z*).

What it unifies.
(a) Value questions become entropy questions (R3, R4). The value can first leave its prediction only through a concentrated winner. This single statement covers:
- the p=3 counterexamples, which have equal neuron costs and m = ν−⌊ν/3⌋ neurons;
- the Note's width theorem;
- the Note's gain/loss accounting: its factor (ν/m)^{s−1} is the integrated entropy deficit, and it reproduces the whole certified excess to 1e-4.
(b) Count questions become value questions with explicit slack (R6). The lower half of (C), "the features contain an optimal support", follows from any decoupling bound accurate to within ε_tol(s) = δ_min/(1−δ_min), computed exactly from the programme. The upper half is a separate, local no-extra-features statement. Three known results are instances:
- Note Thm 'dense': the universal bound has slack ν^{s−1}−1, which reproduces the 1.331 threshold at p=5;
- p=3, and every K*=1 regime: ε_tol=∞, so the lower half is free;
- (p,n)=(5,2),(7,2): K*=F on the whole range, so the upper half is vacuous and (C) is exactly the lower half.
(c) Decoupling, the price–programme theorem and directional (profile) prices are statements about upper envelopes of the same curves (R5, R8).

Which cases it removes.
- From (C) it removes every value question except two: (i) approximate restricted decoupling with tolerance ε_tol(s), which is 10–75% away from transitions at p≤13; (ii) exact identities at the finitely many transitions (R7). The one purely local remainder is "no small extra features".
- From the value conjecture it removes every failure mode except concentrated winners.

What it corrects. The lever "(C) needs only decoupling" is right. But (D) ⟺ [γ* = πB_pΓ_s] + [support identification] (R5). (D) also requires direction-uniform single-feature prices (R8). And (C) on both sides of a transition forces W_{K<}(s_c) = γ*(s_c) (R7). The p=3 lever works because F={1} has no transitions; it does not transfer to any p≥5 that has a transition in range.

Routes inside the angle that stop, and why.
1. Symmetry averaging: G, Σ, torus, dilations.
   - Averages that keep the table cost more, by the factor ν^{1−r}Σ|β|^r; requiring no extra cost is "no cheap features on aggregate", which implies the IF value.
   - Cost-neutral averages (Σ gives θ_split) change the table. Split ratios at the local maxima found are 0.03–0.6.
2. Competitors give only lower bounds on h(θ*). The count needs upper bounds, which come only from test measures.
3. Independent piece translations in class 𝒞: global optimality makes the alignment optimal, which works against us. "No coordination gain" is exactly the class-𝒞 unicorn problem (O1).
4. KKT plus global: log γ = E_w log t_i − (s−1)H(w) − (ν/2) log ν exactly (t_i = singular values of E_q∂g). It is local and θ-dependent.
5. Share floors (R13): rigorous but O(1/log p).
6. Continuation in s: this route succeeds, and gives R1–R4, R6, R7.

## Results

### R1-temperature [PROVED]

Temperature identity.
Setting: θ a network with h(θ)>0, zero neurons dropped. s=ν/b. γ(θ;s) = h(θ)/(Σ_i‖ω_i‖^{ν/s})^s. Cost shares w_i = ‖ω_i‖^b/Σ_j‖ω_j‖^b; H = −Σ w_i log w_i; V = Σ w_i (log w_i)² − H².
Claim: d/ds log γ(θ;s) = −H(w), and d²/ds² log γ(θ;s) = −V(w)/s ≤ 0. So log γ is concave in s. The same identities hold for the cone efficiency e(θ;s) = min_x cm(Φ_1)/(B_p P_b^s) of a single feature.
Reading: a fixed network's normalised margin decays in s at a rate equal to the entropy of its cost shares; e^H is the effective number of neurons.
Proof.
1. Put a_i = ‖ω_i‖^ν, x_i = log a_i, Z = Σ a_i^{1/s}. Then ‖ω_i‖^b = a_i^{1/s}, log γ = log h − Φ with Φ = s log Z, w_i = a_i^{1/s}/Z, and log w_i = x_i/s − log Z.
2. Φ' = log Z − s^{-1}Σ w_i x_i = −Σ w_i log w_i = H.
3. With x̄ = Σ w_i x_i: dw_i/ds = −s^{-2} w_i (x_i − x̄); d log Z/ds = −x̄/s²; dx̄/ds = −s^{-2} Var_w(x).
4. Hence Φ'' = Var_w(x)/s³ = V/s, because Var_w(log w) = Var_w(x)/s². ∎

*Where:* Proof in this output. Check: unify/global/entropy_check.py → entropy_check.out (1)

*Check:* Finite differences on random networks (p=5,n=2 and p=7,n=3; s=1.2, 1.4): dlog/ds matches −H to 7 digits; d² matches −V/s to 5 digits.

### R2-danskin [PROVED]

Slopes of the optimum are entropies of maximisers.
Claim: for s∈(1,ν/2),
- D⁺ log γ*(s) = −min{H(θ,s) : θ global maximiser};
- D⁻ log γ*(s) = −max{H(θ,s) : θ global maximiser}.
The extremes are attained and are the same over maximisers of width ≤ D = p^{n+1} or of any finite width. Consequences: log γ* has only convex kinks; every maximiser satisfies D⁻ ≤ −H(θ,s) ≤ D⁺; at points of differentiability all maximisers have equal cost-share entropy. The same holds for the price π(s), with optimal single features in place of maximisers.
Proof.
1. Width reduction: given θ, write table = Σ μ_i T̂_i with unit rank-ones T̂_i. On the pointed polyhedron {μ ≥ 0 : Σ μ_i T̂_i = table}, the concave function Σ μ_i^r (r = b/ν) is non-decreasing along recession directions, so it is minimised at a vertex, which has ≤ D non-zero entries. Balancing (AM–GM: ‖ω‖² ≥ ν‖T‖^{2/ν}) gives a network with the same table, width ≤ D and no larger P_b.
2. So γ*(s) is the maximum over the compact set 𝒦_D = {width ≤ D, Σ‖ω_i‖² = 1}, and it is continuous in s (P_b ≥ D^{1−b/2} on 𝒦_D).
3. On a compact J ⊂ (1,ν/2), every maximiser has h ≥ η_J > 0 (because γ* ≥ γ_IF > 0). So f(θ,s) = log γ(θ;s) and ∂_s f = −H are continuous on the compact set {h ≥ η_J} × J.
4. Danskin's theorem (Danskin 1967; Bonnans–Shapiro, Perturbation Analysis, Prop. 4.12) gives the one-sided derivatives as the max and min of ∂_s f over the argmax.
5. Enlarging D leaves the left-hand sides unchanged, so the bounds hold for maximisers of every finite width. ∎

*Where:* Proof in this output

*Check:* Near the p=5, n=3 transition s_c = 1.8552, the maximisers found are 2-feature IF on the left (H = log 8) and 1-feature IF on the right (H = log 4). This is the convex kink predicted, with right slope −log 4 > left slope −log 8 (trans_search.out, b = 2.161 vs 2.151).

### R3-entropy-criterion-price [PROVED]

Entropy criterion for the price of a single feature (E1). Fix p, n; π(s) = π_{n,ν/s}(p).
(a) Suppose that for every s∈(1,ν/2], every optimal single feature θ at s with e(θ;s) > ν^{1−s} has cost-share entropy H(θ,s) ≥ log ν. Then π(s) = ν^{1−s} for all b∈[2,ν], i.e. (Q1) holds at every b. Contrapositive: if a single feature can be bought below the IF price at some b, then at some b' ≥ b an OPTIMAL cheap single feature has effectively fewer than ν neurons (H < log ν). So (Q1) reduces to excluding concentrated cheap features: an entropic (robust) width theorem.
(b) Integrated form: log π(s) − (1−s) log ν ≤ ∫_1^s (log ν − H_min(σ))⁺ dσ, where H_min(σ) is the least entropy among optimal single features at σ.
Proof. Let L(s) = log π(s) + (s−1) log ν.
1. L is continuous: use R2's compactness with D_1 = 3p^n; projecting outputs onto span{1, cos, sin} keeps a single-feature table and does not raise the cost.
2. L ≥ 0, from the IF block.
3. L(1) = 0: at b = ν the single-neuron budget Σ_j|β_j(ω)| ≤ B_p‖ω‖^ν gives min cm ≤ Re β_1 ≤ B_p Σ‖ω_i‖^ν.
4. By R2, D⁺L = log ν − H_min.
5. Proof of (a). If L(s_2) > 0, let s_1 = max{s ≤ s_2 : L = 0}. On (s_1, s_2] every optimal single feature beats IF, so by hypothesis D⁺L ≤ 0 there.
6. Dini's lemma: a continuous L with D⁺L ≤ 0 on (a,c) satisfies L(c) ≤ L(a). (If not, take M(s) = L(s) − L(a′) − ε(s−a′) and t = sup{M ≤ 0}; then D⁺M(t) ≥ 0 > −ε ≥ D⁺M(t), a contradiction.) Hence L(s_2) ≤ 0, a contradiction.
7. Proof of (b). H_min is lower semicontinuous (the argmax has a closed graph), so g = (log ν − H_min)⁺ is upper semicontinuous and bounded. Compare L with ∫ g_k for continuous g_k ↓ g. ∎
This is not a value claim. At p = 3 the hypothesis fails, as it must.

*Where:* Proof in this output. Checks: entropy_check.py → entropy_check.out (4); integrated_check.py → integrated_check.out

*Check:* The certified p=3 networks (oned_certified, n=11..18) beat IF and have equal neuron costs with H = log m < log ν (m = 8..13, ν = 12..19), as (a) requires of any counterexample. For these fixed networks the integrated identity is exact: log(γ/γ_IF) at b=2 equals (log ν − H)(ν/2 − s_cross):
- n=11: 0.06117 vs 0.06112;
- n=14: 0.37633 vs 0.37624;
- n=18: 0.79389 vs 0.79376.
The differences are the grid resolution of s_cross.

### R4-entropy-criterion-value [PROVED]

Entropy criterion for the IF value (E2: Conjecture 20 at fixed p, n).
(a) Suppose that for every s∈(1,ν/2] with γ*(s) > γ_IF(s), every global maximiser θ* has H(θ*,s) ≥ log ν + min_{z*} H(z*(s)), where z* are the budget shares of programme maximisers. Then γ* ≡ γ_IF on [1,ν/2], i.e. for all b. Equivalently: the first departure from the IF value, as s grows from the solved convex end, is a winner more concentrated than every IF optimum.
(b) Integrated form: log γ*(s) − log γ_IF(s) ≤ ∫_1^s (log ν + min H(z*(σ)) − H*_min(σ))⁺ dσ.
Proof. Let L = log γ* − log γ_IF.
1. L ≥ 0 and L is continuous.
2. L(1) = 0, because γ*_ν = B_p Γ_1 (paper Cor. dense).
3. Take a programme maximiser y* at s and the fixed IF network θ_IF built from it (paper Prop. block). It is interference-free at every s′, so γ(θ_IF;s′) ≤ γ_IF(s′) (paper Thm 8 at b′ = ν/s′), with equality at s′ = s. Its cost shares are z*_j/ν (ν equal neurons per feature). By R1, D₊ log γ_IF(s) ≥ −log ν − min H(z*).
4. With R2: D⁺L ≤ log ν + min H(z*) − H*_min. Conclude with Dini's lemma as in R3. ∎

*Where:* Proof in this output. Check: entropy_check.out (2), (3)

*Check:* The IF network's entropy equals log ν + H(z*), and d/ds log Γ_s = −H(z*) by finite differences, to 6 digits at (p,s) = (5,1.5), (7,1.3), (7,2.0), (11,1.42), (11,1.8). At p = 3, (a) correctly does not apply: concentrated winners exist.

### R5-decoupling-is-value [PROVED]

Decoupling = price–programme value + support.
Claim: for a global maximiser θ*, decoupling (D) with P_j > 0 ⟺ β_j ≠ 0 holds iff both
(i) γ* = π B_p Γ_s, and
(ii) the feature set of θ* is the support of some programme maximiser.
In particular, (D) at a single maximiser implies Conjecture 20′ at (p,n,b). The push's lever avoids the IF value, not the price–programme value.
Proof.
- (⟹) Note Prop. 'decoupling suffices': z_j = (P_j/P)^s is a programme maximiser with spt z = features, and h ≤ π B_p P^s Γ_s. The reverse inequality γ* ≥ π B_p Γ_s is the price–programme theorem.
- (⟸) Take P_j = (y*_j)^{1/s} P. ∎

*Where:* Proof in this output

*Check:* The global maxima found at n=2 (p = 5, 7, 11; π_{2,b} = 3^{1−s} is proved in Note Thm q1n2) satisfy γ = γ_IF = π B_p Γ_s, and their features equal spt y* (trans_search.out).

### R6-slack-principle [PROVED]

Slack principle: the lower half of (C) from approximate decoupling.
Setting: θ* a global maximiser with feature set S(θ*). Suppose
  (AD_ε)  h(θ*) ≤ (1+ε) π B_p min_δ Σ_{j∈S(θ*)} P_j^s κ_j(δ)  for some P_j ≥ 0 with Σ P_j ≤ P_b(θ*).
Define the support gap δ_min(s) = min{1 − Γ^{(S)}_s/Γ_s : S ⊆ F contains no optimal support} (+∞ if there is no such S), and ε_tol = δ_min/(1−δ_min).
Claim: if ε < ε_tol, then S(θ*) contains the support of a programme maximiser, so |S(θ*)| ≥ min 𝒦*(s). If ε_tol = ∞, no hypothesis is needed.
Proof.
1. z_j = (P_j/P)^s satisfies Σ z_j^{1/s} ≤ 1 and spt z ⊆ S(θ*).
2. The price–programme theorem gives h(θ*) ≥ π B_p Γ_s P^s. With (AD_ε), min_δ Σ z_j κ_j(δ) ≥ Γ_s/(1+ε) > (1−δ_min) Γ_s.
3. If S(θ*) contained no optimal support, the left side would be at most Γ^{(S(θ*))}_s ≤ (1−δ_min) Γ_s, a contradiction.
4. If δ_min = ∞, use that S(θ*) ≠ ∅ (paper Lemma 2 and h > 0). ∎
Corollaries.
(a) Lemma 2 together with |β_j| ≤ Q_j^s gives (AD_ε) for EVERY network, with P_j = Q_j/C_b and 1+ε = 1/π ≤ ν^{s−1}. The resulting condition 1 − δ_min(s) < ν^{1−s} is exactly Note Thm 'dense'. At p=5, ν=3 it reads 6^s < 10.854, i.e. s < 1.3309 (the Note states 1.331).
(b) If K*(s,p) = 1, every singleton is an optimal support (dilations act transitively on F), so ε_tol = ∞. Then (C) ⟺ 'every maximiser has exactly one feature', a pure no-extra-feature statement. p = 3 is the case F = {1}.
(c) If K* = F on all of [1,ν/2] — i.e. (p,n) = (5,2), (7,2) — the upper half is vacuous, and (C) for all b is exactly the lower half.

*Where:* Proof in this output. Tolerances: vertex_exact.py → vertex_exact.out; slack.py → slack.out

*Check:* δ_min was computed two independent ways: exhaustive supports with reweighted LP, and exact vertex enumeration of the coverage polytope. They agree to 4 digits for p = 5, 7, 11, 13 at 10 values of s. Corollary (a) reproduces the Note's dense threshold at p=5 (1.3309).

### R6b-tolerance-table [NUMERICAL]

Tolerances ε_tol(s), computed by exact vertex enumeration of the coverage polytope (exact combinatorics, floating-point arithmetic). Values are listed at s = 1.05, 1.2, 1.35, 1.5, 1.7, 1.85, 1.86, 2.0, 2.3, 2.5.
- p=5: 0.747, 0.575, 0.419, 0.279, 0.114, 0.004 (transition at 1.8552), then ∞ for s > 1.8552 (K*=1).
- p=7: 0.239, 0.161, 0.088, 0.020 (transition at 1.5475), 0.531, 0.383, 0.374, 0.249, 0.017 (transition at 2.3247), ∞ at 2.5.
- p=11: 0.101, 0.050, 0.058, 0.400, 0.376, 0.362, 0.361, 0.350, 0.330, 0.320.
- p=13: 0.077, 0.031, 0.015, 0.074, 0.092, 0.101, 0.101, 0.100, 0.098, 0.097.
Away from transitions, any decoupling bound within 10–40% of the price (up to 75% at p=5) fixes the support. Exactness is needed only as s → s_c.

*Where:* vertex_exact.out, slack.out

*Check:* 

### R7-transition-identity [PROVED]

Transition identity: (C) forces exact value identities at transitions.
Setting: s_c ∈ (1,ν/2), ε > 0, 𝒦*(s) = {K_1} on (s_c−ε, s_c) and {K_2} on (s_c, s_c+ε), with K_1 ≠ K_2 and K_< = min(K_1,K_2). W_K denotes the best normalised margin among networks with at most K features.
Claim: if (C) holds at every s ≠ s_c in (s_c−ε, s_c+ε), then W_{K<}(s_c) = γ*(s_c), and W_{K<}(s) < γ*(s) on the side where K_> is the programme count. The programme switches at s_c in the same way, with Γ_s/Γ^{(≤K<)}_s. So (C) forces the network-level and programme-level switching points to coincide EXACTLY. This is an equality between optimal values of two non-convex network classes, i.e. a value statement.
Proof.
1. W_K is continuous: 'at most K features' is a closed condition preserved by width reduction, so R2's compactness applies.
2. On the K_< side every maximiser has K_< features, so W_{K<} = γ*.
3. On the other side, the maximum defining W_{K<} is attained by a network with ≤ K_< < K_> features. That network is not a maximiser, so W_{K<} < γ*.
4. Let s → s_c. ∎
Transitions in range:
- p=5: s_c = 1.8552 (in range for n ≥ 3);
- p=7: s_c = 1.5475 and 2.3247 (n ≥ 3);
- p=11: s_c = 1.3454 and 1.4895;
- every prime 11 ≤ p ≤ 199 has a transition below s = 1.5, i.e. in range for every n ≥ 2. Computer check: an allocation found by LP exceeds (p/(p−1))F^{1−s}, which bounds every full-support allocation (floating point, by a factor of about 1.08 at p=11 and 2.3 at p=199).
At p = 3 there are no transitions. That is why the p=3 lever works and does not transfer.

*Where:* Proof in this output. transitions_fast.py → transitions.out; vertex_exact.out; dense_end.py → dense_end.out

*Check:* Transition points agree between bisection and vertex enumeration. (C) is observed numerically on both sides of each searched transition (R11). The dense-end certificate holds for all 41 primes from 11 to 199.

### R8-profile-programme [PROVED]

Profile programme: decoupling needs direction-uniform prices.
Definitions. For a single feature θ, its profile is m_θ(l) = min_x [f(x)[y] − f(x)[y−l]]/(B_p P_b^s), for l ∈ Z_p^*. Let Γ^𝒫_s = sup over P ∈ Δ_F and single features θ_j of min_δ Σ_j P_j^s m_{θ_j}(jδ).
Claims.
(a) γ* ≥ B_p Γ^𝒫_s ≥ π B_p Γ_s.
(b) If (D) holds at any maximiser, then Γ^𝒫_s = π Γ_s.
(c) Suppose an OPTIMAL single feature θ_0 (e = π) has m_0(l) > π κ_1(l) for every l ∉ {±1}, and some programme maximiser has at least 2 features. Then Γ^𝒫_s > π Γ_s, so (D) and Conjecture 20′ fail at that (p,n,b), even though the count could survive.
(d) If the averaged directional price π^{avg}(l) = sup_θ (m_θ(l) + m_θ(−l))/(2κ_1(l)) equals π for every l, then Γ^𝒫_s = π Γ_s.
Proof.
(a) The relabelled copy u ↦ u(j·), w ↦ w(j·), scaled to cost P_j P, has gaps g(jx, jδ) ≥ B_p (P_j P)^s m(jδ) (Companion, proof of Thm pp). Gaps of a union add. The cone network has m(l) ≥ π κ_1(l), and κ_1(jδ) = κ_j(δ).
(b) R5 together with (a).
(c) For each δ at most one j ∈ F has jδ ≡ ±1. With two or more features in the support, Σ y*_j m_0(jδ) > π Σ y*_j κ_j(δ) ≥ π Γ_s at every one of the finitely many δ.
(d) The minimum over δ is at most the average over ±δ. ∎

*Where:* Proof in this output. Check: check_profile.py → check_profile.out (B)

*Check:* For perturbed, non-IF single features with non-uniform profiles, the union of relabelled copies with the programme allocation has margin ≥ min_δ Σ_j y_j m_0(jδ) > c_0 Γ_s, strictly, at p=7 and 11 (n=2) and p=7 (n=3). At p=11 the cone price undervalues the decoupled union 15-fold.

### R9-directional-n2 [PROVED]

Direction-uniform IF price at n=2.
Setting: n=2, p ≥ 5, 2 ≤ b ≤ 3, θ a single feature, l ∈ Z_p^*. M = max_T |β_{1,T}|; m(l) = min_x Re(Φ_1(x)(1−ρ^{−l})) (unnormalised).
(i) Arbitrary inputs, own partner allowed: (m(l)+m(−l))/2 ≤ κ_1(l)(Re β_1 − |cos(πl/p)| M).
(ii) Single-frequency θ (inputs and outputs in span{1, cos, sin} at frequency 1): (m(l)+m(−l))/2 ≤ κ_1(l) Re β_1 − |sin(πl/p)| M.
(iii) Hence m(l)+m(−l) ≤ 2·3^{1−s} κ_1(l) B_p P_b^s:
- for every l, for single-frequency single features;
- for every l with |cos(πl/p)| ≥ 1/2, for arbitrary single features (≥ 1−3^{1−s} using the computer-assisted Note Thm c1sharp).
Consequently, the profile programme over single-frequency pieces at n=2 equals 3^{1−s} Γ_s.
Proof.
1. Fix T and χ = 2·1_T. Only the own sign-flipped mode a (|a| = |β_{1,T}|) and the own partner b reach ±χ (Note, proof of Thm q1n2).
2. The coefficient of G(·,l) at χ is (a c_l + b̄ c̄_l)/2, with c_l = 1−ρ^{−l}. The Riesz test 1 − cos(2π⟨χ,x⟩/p + ψ) ≥ 0 has mean 1, so min_x G(·,±l) ≤ E G(·,±l) − |coefficient|. Averaging over ±l, the mean part is Re β_1 κ_1(l).
3. Write c_l = |c|e^{iα}, X = a e^{iα} + b̄ e^{−iα}, Y = a e^{−iα} + b̄ e^{iα}. Then X e^{iα} − Y e^{−iα} = 2ia sin 2α, so |X|+|Y| ≥ 2|a||sin 2α|. With α = π/2 − πl/p the detection is at least κ_1(l)|cos(πl/p)||a|. This proves (i).
4. For single-frequency fields at p ≥ 5 the modes lie in {0,1,2}^n (Companion Lemma supp), and −2 ≡ p−2 ∉ {0,1,2}. So b = 0, and the detection is |c||a|/2 = |sin(πl/p)||a|. This proves (ii).
5. Theorem A (Part I; Note Prop. n2gen) gives Re β_1 ≤ 3^{1−s}Q^s + M/2. With (ii) the excess is M(sin² − |sin|) ≤ 0; with (i) it is κ_1 M (1/2 − |cos|). Finally Q ≤ C_b P_b. ∎
At p = 3, −2 ≡ 1 ∈ {0,1,2}: the DC mode lands on −χ_T. That is exactly the counterexample mechanism, and (ii) correctly fails there.

*Where:* Proof in this output. Check: check_profile.py → check_profile.out (A)

*Check:* 300 random single features at p ∈ {5,7,11,13}, b ∈ [2,3], perturbed from IF at scales 1e-3 to 3. Both inequality steps hold, with relative slack down to 4e-4; no violation.

### R10-directional-numerics [NUMERICAL]

Arbitrary single features are direction-uniformly priced at IF.
Both the min-over-±l price sup min(m(l),m(−l))/κ_1(l) and the ±l-averaged price (which controls the profile programme, R8(d)) equal ν^{1−s} exactly (ratio 1.00000) in every direction tested. This includes the most antipodal direction l = (p−1)/2, where R9 is not proved.
Cases covered:
- n=2 at p = 5, 7, 11, 13, 17, 19, 23;
- n=3 at p = 5, 7, 11;
- n=4 at p = 5;
- b ∈ {2, 2.5, 3}; widths up to 7.
Seeds: IF plus noise, own-partner (frequency-3) content, and Gaussian starts. Every run ends at IF or at a worse local optimum.
The one-sided price reaches exactly 1/sin(π|l|/p) (2.3048 at p=7, l=1). That is a phase rotation of the clean amplitude, and it is irrelevant because every margin contains both δ and −δ.

*Where:* direction.py, dir_verbose.py, run_dir_scan.sh → dir_scan.out, dir_p7_n2.out; direction2.py, run_dir2_scan.sh → dir2_scan.out

*Check:* 

### R11-transitions-numerics [NUMERICAL]

(C) holds on both sides of six transitions.
Search: arbitrary networks (not slot-symmetric; all frequencies; DC allowed). Starts are Gaussian or perturbed IF on random supports of every relevant size; widths up to 16; smooth-min L-BFGS annealed to a relative temperature of 3e-6.
Results:
- p=11, n=2 at b = 2.0, 2.009, 2.019, 2.10, 2.225, 2.235, 2.3 (transitions at 2.0141 and 2.2298): best γ/γ_IF = 1.000000 everywhere; counts 2, 2, 3, 3, 3, 5, 5, all equal to K*.
- p=5, n=3 at b = 2.10, 2.151, 2.161, 2.25 (transition at 2.1561): best 1.000000; counts 1, 1, 2, 2 = K*.
- p=7, n=3 at b = 2.50, 2.58, 2.59, 2.70 (transition at 2.5848): best 1.000000; counts 2, 2, 3, 3 = K*.
Of 336 local maxima, 15 are non-IF (cheapness 1.04–1.28). All of them are ≤ 0.8993 of IF and live on mixed modes: their split keeps only 3–60% of the margin.
Temperature analysis of these non-IF local maxima: most have LOWER entropy than IF (cheap features, fewer effective neurons). So they gain on IF as s grows, at slope log ν + H(z*) − H, but their concave curves peak at ≤ 0.90 and never cross 1.

*Where:* gsearch.py, run_trans.sh → trans_search.out; localmax_entropy.py → localmax_entropy.out (run run_trans.sh first; it regenerates the networks)

*Check:* 

### R12-reduction-52-72 [CONDITIONAL]

Reduction of (C) at (p,n) = (5,2) and (7,2), for all b, to a compensation bound with explicit slack.
At (5,2): K* = F = {1,2} for every b ∈ [2,3]. So (C) ⟺ no global maximiser has a single feature. Via R6 (π_{2,b} = 3^{1−s} is proved; W_{{j}} = W_{{1}} by dilation), this follows from
  W_{{1}}(s) < (1+ε_tol(s)) · 3^{1−s} B_5 κ_min(5)  for s ∈ (1.331, 1.5],
with 1+ε_tol = Γ_s/κ_min ≥ 1.279. The range s ≤ 1.331 is already Thm dense. In words: a network whose only feature is 1 — but which may carry any compensating frequency-2 interference, partners or DC — must not beat the single-feature price by 27.9% or more.
At (7,2): K* = F for all b, and (C) ⟸ W_S < (1+ε_tol) 3^{1−s} B_7 Γ^{(S)}_s for 2-sets S. Here 1+ε_tol = 1.020 at b=2, 1.081 at b=2.2, 1.160 at b=2.5.
Numerical status of the hypothesis: penalised searches (β_j forced to 0 off S, everything else free, 12 starts) give
- W_{{1}}/(3^{1−s} B_5 κ_min) = 1.00000 at b = 2.0, 2.1, 2.2, 2.3;
- W_{{1,2}}/(3^{1−s} B_7 Γ^{({1,2})}) = 0.99999 at b = 2.0, 2.2, 2.5.
Compensation gains nothing, so the needed inequalities hold numerically with the full slack. This is the first concrete place where the slack principle turns (C) into a single bound with room to spare.

*Where:* w1_compensation.py, run_w1.sh → w1_compensation.out. Its n=3 lines use the n=2 normalisation 3^{1−s} by mistake; rescaled by (4/3)^{s−1} they equal 1.0000, and they lie in the K*=1 regime, so they are uninformative.

*Check:* 

### R13-share-floor [PROVED]

Share floor: global optimality applied to sub-networks.
Claim: for a global maximiser θ* (cost P) and any subset θ_A of its neurons (cost P_A), at every minimiser (x_R, δ_R) of the gap of θ*∖θ_A,
  g_{θ_A}(x_R, δ_R) ≥ γ*(P^s − (P−P_A)^s) ≥ s γ* (P−P_A)^{s−1} P_A.
In class 𝒞, every frequency piece therefore has cost share ≥ ψ/(1+ψ), with ψ = ν^{−1}(s Γ_s/2^{n+1})^{1/(s−1)}.
Proof.
1. By optimality h(θ*∖θ_A) ≤ γ*(P−P_A)^s, and h(θ*) ≤ h(θ*∖θ_A) + g_{θ_A}(x_R, δ_R). The second inequality is convexity of t^s.
2. For a single-frequency neuron without DC, |Φ_ij(x)| ≤ (2/p)^{ν/2}‖w‖Π‖u_t‖ ≤ 2^n B_p ‖ω‖^ν. So |g_piece| ≤ 2^{n+1} B_p P_j^s.
3. Combine with γ* ≥ ν^{1−s} B_p Γ_s. ∎
Weak: ψ ≈ 3·10^{−3} at p=11, n=2, b=2. It gives only Θ(log p)-type width and count bounds.

*Where:* Proof in this output

*Check:* Consistent with all IF optima found: piece shares are far above the floor.

### C1-entropic-width [CONJECTURE]

Entropic width lemma, the hypothesis of R3(a): for every p ≥ 5, n and b, every single feature with cone efficiency e(θ;s) > ν^{1−s} has cost-share entropy H ≥ log ν.
- By R3 it implies (Q1), π = ν^{1−s}, at every b.
- The multi-feature analogue (maximisers beating IF have H ≥ log ν + min H(z*)) implies Conjecture 20 by R4.
- It is false at p=3, where the counterexamples have H = log(ν − ⌊ν/3⌋).
Evidence: R10 (no single feature beats IF at all at small p ≥ 5), and the Note's width theorem (≥ ν neurons for shared-sign networks when p > ν).

*Where:* Statement in this output

*Check:* 

## Obstruction
1. Where (C) becomes a value statement. At every prime p ≥ 5 with a transition s_c of K*(·,p) in [1,ν/2], (C) on a neighbourhood of s_c IMPLIES the exact identity W_{K<}(s_c) = γ*(s_c) (R7). This holds at every p ≥ 11 (checked for 11 ≤ p ≤ 199) and at p=5, 7 when n ≥ 3. It is an equality between optimal values of two non-convex network classes, and it encodes an UPPER bound: networks with more features do not beat the best networks with ≤ K_< features at s_c.

2. Why global optimality cannot supply it. Global optimality, symmetry averaging and competitor constructions produce only LOWER bounds on h(θ*), or change the table, or cost more because r < 1 (R14 routes 1–5). The needed upper bound is restricted decoupling with tolerance ε_tol(s), and ε_tol(s) → 0 as s → s_c (R6). At s_c this is Conjecture 20′ restricted to count classes (R5). Its single-feature part is direction-uniform pricing (R8), which is proved only for single-frequency features at n=2 (R9).

3. What remains, in minimal form.
(V) No concentrated winner (R3, R4): no maximiser (or optimal single feature) beating IF may have cost-share entropy below the IF optimum's. This is an entropic width theorem, open for n ≥ 3 and for partner-carrying inputs at n=2.
(T) Exact restricted decoupling at each transition. By R4 it follows from (V) plus rigidity at ties.
(NS) No extra features, the upper half of (C). It is all of (C) in K*=1 regimes. Approximate decoupling excludes extra features of macroscopic share only. Small extra features produced as by-products of multi-frequency neurons at b=2 are charged by no known inequality (compare the Note's p=5, n=6 local maximiser, whose spurious feature has relative size 1.3e-4).

4. Cases where the remaining step is a single concrete inequality. At (5,2) and (7,2), (C) for all b reduces to one restricted compensation bound with 27.9% (respectively 2.0–16%) slack (R12). The numerics show zero compensation gain, but the bound is unproved: it must control foreign-channel modes. At p=5 these are output 2 with DC or frequency-1 inputs, which are paid only quadratically (Note, Lorentz lemma).

## Next steps
- FIRST LEMMA — entropic width lemma for single features (hypothesis of R3(a)). Statement: a single feature with cone efficiency > ν^{1−s} has cost-share entropy ≥ log ν. By R3 this alone gives (Q1), π = ν^{1−s}, at every b for that (p,n). Start with slot-symmetric shared-sign networks at p > ν, where the Note's Laguerre sign-change width theorem already gives ≥ ν neurons. The step to add is cost-weighting: show that neurons with small cost share cannot supply the required sign alternations, e.g. via a weighted Descartes/Laguerre count. The p=3 data show the extremal structure: equal costs, H = log m.
- Prove the (5,2) compensation bound of R12. A one-feature network at p=5, n=2, carrying any frequency-2 interference fields, partners or DC, must have γ < 1.279 · 3^{1−s} B_5 κ_min on s ∈ (1.331, 1.5]. This alone proves (C) at (5,2) for every b. It has 27.9% slack and numerically zero gain. Route: the master inequality at the cone directions ±1. Foreign channels for l=1 from output 2 are (DC on T, frequency 2 off T) and (frequency 1 on T, frequency 2 off T). They are stray-output or riding modes, quadratically paid (Note, Lorentz lemma), so a crude constant may suffice.
- Then (7,2): the same with 2-sets S and slack 2.0% (b=2) to 16% (b=2.5). After that, approximate restricted decoupling inside the R6b tolerance table at p=11, 13. Any inequality of master-inequality type with that much slack proves the lower half of (C) there.
- No-small-extra-features lemma at b > 2: multi-frequency neurons strictly lose budget there (strict superadditivity). Combine this with the share floor (R13) and a small-feature perturbation argument to get (NS). At b=2, charge the by-product mechanism through the unused budget U.
- Partner-aware directional payment at n=2 for directions with |cos(πl/p)| < 1−3^{1−s}. This extends R9 to all inputs and gives Γ^𝒫_s = 3^{1−s} Γ_s at n=2 unconditionally. R10 says it holds with room.
- Use the integrated entropy bounds R3(b)/R4(b) quantitatively. Any lower bound on the cost-share entropy of optimal networks — for example from width theorems combined with a proof that optimal neurons have equal costs, which holds exactly in the certified p=3 data — converts directly into an upper bound on the price and on the value excess.
