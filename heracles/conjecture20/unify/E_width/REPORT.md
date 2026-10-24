# E_width: report (saved by the orchestrator from the agent's structured output)

## Headline
WIDTH-ROUTE (E_width). I did not prove (E_S) in general, and I found no general rigidity for the upper half of (C).

1. The single inequality as requested is FALSE for every network class, in both signs. The requested form is log γ ≤ log D_S + (s−1)[H^dec_S(s) − H(θ,s)] − Δ with Δ ≥ 0. Interference-free networks break it, so no detection term can rescue it:
- the uniform IF network at p=11 and p=13 breaks the + sign by up to e^{0.38};
- the single IF block at (5,2) breaks the − sign for s > 1.4276.

2. The correct form is integrated. For every network, every p, n and b, γ(θ;s) = γ(θ;1)·exp(−(s−1)·H_{1/s}(m_θ)) (W1), where H_{1/s} is the Rényi entropy of order 1/s and m_θ is the vector of neuron mass shares ||ω_i||^ν. This gives a proved all-network inequality of the requested shape (W3):
 log γ ≤ log D_S + (s−1)[H̄^dec_S(s) − H_{1/s}(m_θ)] − Δ_S(θ).
- Δ_S(θ) is the inefficiency at the convex end b = ν. It is bounded below by δ-Fourier detection of every non-clean mode (W4).
- (E_S), (EW_S), this exchange inequality for all θ, and W_S ≡ D_S are all equivalent (W2). The high-entropy half is free; all of the content is in the low-entropy half.

3. Shared neurons are not a bookkeeping obstacle (W5). Sharing lowers the Rényi entropy by exactly the extra cost at b = ν when b = 2, and by at most that much when b > 2. Coupling can therefore act only through the margin (mixed modes).

4. Payment side. For every n, Lemma 7 has a spectral-entropy form (W6: Rotfel'd plus Weyl on the chain matrix; GIBBS generalised to all n via a twisted Hermitian chain plus Schur–Horn). With phase-aligned atoms the payment is quadratic and does not depend on n. This proves a new entropic width theorem (W8). For every p with cos(2π/p) > 0 (all p ≥ 5), every n and every b:
- a single feature with a circular field and phase-aligned atoms is never cheap;
- its effective spectral width is at least ν/(1 + 2k_p x²/ν).
So cheapness requires atom cancellation or non-circular (2j-pair) content.

## Principle
RÉNYI BOOKKEEPING AND THE EXCHANGE RATE.

(1) Whole-network identity, exact for every network with no bookkeeping:
 log γ(θ;s) = log γ(θ;1) − (s−1)·H_{1/s}(m_θ), where m_i = ||ω_i||^ν / Σ_j ||ω_j||^ν.
Equivalently, ∫_1^s H(θ,σ) dσ = (s−1)·H_{1/s}(m_θ): the path average of the cost-share entropy is a Rényi entropy of the masses.
- The problem at temperature s is therefore the solved convex end (b = ν), penalised by Rényi concentration.
- An "entropic width theorem" is exactly an EXCHANGE RATE: inefficiency at the convex end must pay for Rényi concentration at rate s−1.
- This is the only form that can hold for every network. Pointwise-in-s forms are false even for interference-free networks: the decoupled frontier moves with s through H(z*(s)), so cost entropies at one s cannot be compared.

(2) Spectral bridge, for one feature and every n. The chain matrices H = Σ β_l m_l m_l^* (non-Hermitian) and K = Σ ρ_l u_l u_l^* (twisted Hermitian) have spectra that control cost through Rotfel'd (high spectral entropy means expensive). Their entries are interval interference coefficients, so the same spectra are seen by detection. Rényi-2 turns the spectral deficit into a Frobenius (L2) quantity, and the L2 cone-moment bound detects that quantity. The two sides meet with an n-uniform constant exactly when the atoms are phase-aligned.

(3) Shared neurons trade Rényi entropy against cost at the convex end, at rate exactly 1 when b = 2. Coupling gains can therefore come only through the margin.

What this unifies:
- Paper Thm 8: IF networks, via the Rényi form of Lemma 7.
- The Note's gain/loss accounting: (ν/m)^{s−1} is the case of equal masses.
- LARGEN R12's participation number: K_r = e^{H_{1/s}}.
- GLOBAL R3/R4.
- The GIBBS entropy remark, for every n.
- The Note's quadratic payment, for every n.
- p52 R12, extended to every mode.
- The one-neuron lemma, for every n.

What it removes:
- the s-local entropy forms (refuted);
- shared-neuron cost bookkeeping;
- aligned circular single features, for every n.

What it leaves:
- a single exchange inequality, which must use p ≥ 5;
- atom cancellation;
- 2j-pair content;
- the upper half of (C).

## Results

### W1 [PROVED]

Rényi temperature identity. For every product network θ with h > 0 (any p, n, width, inputs or outputs) and every s ≥ 1:
 γ(θ;s) = γ(θ;1)·exp(−(s−1)·H_{1/s}(m_θ)).
Here a_i = ||ω_i||^ν, m_i = a_i / Σ_j a_j, and H_α(m) = (1−α)^{-1} log Σ m_i^α. Equivalently, ∫_1^s H(θ,σ) dσ = (s−1)·H_{1/s}(m_θ). Reading: the s-average of the cost-share entropy is the Rényi-(1/s) entropy of the neuron masses.

**Proof / evidence.**

1. P_b = Σ a_i^{1/s} and P_b^s = P_ν·(Σ m_i^{1/s})^s.
2. s·log Σ m_i^{1/s} = s(1 − 1/s)·H_{1/s}(m) = (s−1)·H_{1/s}(m).
3. Hence γ(θ;s) = h/P_b^s = γ(θ;1)·e^{−(s−1)H_{1/s}}.
4. The integral form follows from GLOBAL R1 (d/ds log γ = −H) integrated over [1,s].

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c3_renyi_detect.py, part (R), output c3_renyi_detect.out. 200 random mass vectors: identity error 8.9e-16; integral form (quadrature) error 6.7e-16.

### W2 [PROVED]

Equivalences. For every p, n and S ⊆ F, the following are equivalent:
(a) (E_S): for almost every s, every maximiser has H ≥ H^dec_S(s).
(b) (EW_S): for all θ in N_S and all s, H(θ,s) < H^dec_S(s) implies γ(θ;s) ≤ D_S(s).
(c) For all θ in N_S and all s: log γ(θ;1) − (s−1)·H_{1/s}(m_θ) ≤ log D_S(s).
(d) Exchange inequality: Δ_S(θ) ≥ (s−1)[H̄^dec_S(s) − H_{1/s}(m_θ)] for all θ in N_S and all s. Here Δ_S(θ) = log(B_p Γ_1^{(S)} / γ(θ;1)) and H̄^dec_S(s) = (s−1)^{-1} ∫_1^s H^dec_S.
(e) W_S ≡ D_S on [1, ν/2].
Under (e), every maximiser has H(θ*,s) = H^dec_S(s) EXACTLY, for almost every s.
Reading: any entropic width theorem that yields (E_S) is the restricted price–programme value identity; nothing weaker suffices.

**Proof / evidence.**

Lipschitz step.
- Width reduction (GLOBAL R2) keeps the table and does not raise P_b, so W_S(s) is a sup over width ≤ D = p^ν.
- There |∂_s log γ| = H ≤ log D, so log W_S is (log D)-Lipschitz.
- Likewise log π (width 3p^n) and log Γ^{(S)}_s (a max of finitely many smooth vertex curves) are Lipschitz.
- So L_S = log W_S − log D_S is absolutely continuous.

Dini bound for D_S.
- Fix σ, an optimal single feature θ_1 and a maximiser y* of Γ^{(S)}_σ.
- Put f(σ') = log e(θ_1;σ') + log G_{σ'}(y*), where G_{σ'}(Y) = min_δ Σ Y_j κ_j(δ) / (Σ Y_j^{1/σ'})^{σ'}.
- Then f ≤ log D_S (because e ≤ π and G ≤ Γ^{(S)}), f(σ) = log D_S(σ), and f'(σ) = −H(θ_1,σ) − H(z*) by the temperature identities.
- Hence D_+ log D_S ≥ −H^dec_S everywhere.

Implications.
- (a)⇒(e): BRIEF_E. Danskin gives D^+ log W_S = −H*_S, so D^+ L_S ≤ 0 almost everywhere on {L_S > 0}. With L_S(1) = 0, L_S ≥ 0 and absolute continuity, L_S ≡ 0.
- (b)⇒(e): on {L_S > 0} every maximiser has γ = W_S > D_S, so (b) gives H ≥ H^dec. Conclude as before.
- (e)⇒(b) and (e)⇒(c): every θ has γ ≤ W_S = D_S. Statement (c) is exactly γ(θ;s) ≤ D_S(s), rewritten through W1. So (c)⇔(e).
- (d)⇒(e): by W1, log γ(θ;s) = log D_S(1) − Δ_S − (s−1)H_{1/s} ≤ log D_S(1) − ∫_1^s H^dec ≤ log D_S(s). The last step is the Dini bound integrated.
- (e)⇒(d) and the equality claim: decoupled networks Θ lie in N_S and have γ ≥ πB_pΓ^{(S)} = W_S, so they are maximisers. At every differentiability point of log W_S (almost every s), Danskin gives all maximisers of any finite width (GLOBAL R2, step 5) a common entropy. Θ has entropy H(θ_1,s) + H(z*) for every admissible choice, so the common value is H^dec_S. Hence (log D_S)' = −H^dec_S almost everywhere, log D_S(s) = log D_S(1) − ∫H^dec_S, and γ ≤ D_S becomes (d). (E) then holds with equality.

*Check:* Logic only. Consistency checks:
- c4_refute.py: the programme temperature identity log Γ_s − log Γ_1 = −∫H(z*) holds at p = 11 and 13 across transitions, to 2e-5 (quadrature).
- c6_integrated.py: statement (d) holds with equality on IF networks.

### W3 [PROVED]

Integrated entropic width inequality: the requested aim, in its correct integrated form. For every p, n, b, every S, every θ in N_S and every s in (1, ν/2]:
 log γ(θ;s) ≤ log D_S(s) + (s−1)[H̄^dec_S(s) − H_{1/s}(m_θ)] − Δ_S(θ), with Δ_S(θ) ≥ 0.
Free half: if H_{1/s}(m_θ) ≥ H̄^dec_S(s), then γ(θ;s) ≤ D_S(s)·e^{−Δ_S(θ)}.
Winner form: γ(θ;s) > D_S(s) forces H_{1/s}(m_θ) < H̄^dec_S(s) − Δ_S(θ)/(s−1).
Shared neurons enter only through the masses a_i; no feature-level bookkeeping is used.

**Proof / evidence.**

1. By W1, log γ(θ;s) = log D_S(1) − Δ_S(θ) − (s−1)H_{1/s}(m_θ).
2. By the Dini bound of W2, integrated, log D_S(1) ≤ log D_S(s) + ∫_1^s H^dec_S.
 - Integration uses absolute continuity of log D_S.
 - H^dec_S is bounded and lower semicontinuous: it is a min of the continuous function H(θ_1,σ) + H(z*) over closed-graph, compact-valued argmax correspondences (Berge).
3. Δ_S ≥ 0 follows from paper Lemma 2 and the single-neuron budget at r = 1: h ≤ Γ^{(S)}_1 Σ_{j∈S} |β_j| ≤ Γ^{(S)}_1 B_p P_ν.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c6_integrated.py, output c6_integrated.out. Run at n = 2, where π = 3^{1−s} and H(θ_1) = log 3; 36 perturbed and over-spread IF networks at p = 5, 7, 11.
- max[log γ − RHS] = 1.8e-15: equality is reached by IF networks.
- 10 networks satisfy the free-half hypothesis; all of them have γ ≤ D_S.

### W4 [PROVED]

δ-Fourier isolation for EVERY mode, and detection at the convex end. Take any network, any j in F, any χ in Z_p^n with χ ≠ 0, and any symmetric probability μ on Z_p^*. Then
 h ≤ Σ_{l∈F} K_l(μ) Re β_l − (p/2)·min_δ μ_δ·|φ_j(χ)|,
where φ_j(χ) = E_x Φ_j(x) ρ^{−⟨χ,x⟩} ranges over every non-clean coefficient: sign-flipped, partner, foreign or mixed.
With μ uniform, K_l = p/(p−1) = Γ_1. Hence, with M(θ) = max_{j, χ≠0} |φ_j(χ)|:
 γ(θ;1) ≤ B_pΓ_1·(1 − M/(2B_pP_ν)), and Δ_F(θ) ≥ −log(1 − M(θ)/(2B_pP_ν(θ))).

**Proof / evidence.**

1. This is p52 R12 with χ_{j,T} replaced by any χ ≠ 0.
2. The identity Σ_δ ρ^{jδ} E_x[g ρ^{−⟨χ,x⟩}] = −(p/2)φ_j(χ) uses only:
 - g = Σ_l Re(Φ_l c_l);
 - Σ_{δ≠0} ρ^{jδ} c_l(δ) = −p·[l = j];
 - Σ_δ ρ^{jδ} conj(c_l(δ)) = 0, since j + l ≢ 0.
3. q_w = p^{−n}[μ_δ + 2Re(w ρ^{jδ} ρ^{−⟨χ,x⟩})] is a probability for |w| ≤ min μ/2, because χ ≠ 0 kills the mass of the oscillating term.
4. Pairing with w = (min μ/2)·conj(φ)/|φ| gives the bound.
5. Corollary: Σ_l Re β_l ≤ Σ_{i,l} |β_il| ≤ B_p P_ν (budget at r = 1), and Γ_1 = p/(p−1) because Σ_{l∈F} κ_l(δ) = p/2 for every δ.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c3_renyi_detect.py, part (D), output c3_renyi_detect.out.
- The identity holds for random (j, χ) on 90 networks at (5,2), (7,2), (5,3), (3,3), (11,2), (7,3), with DC and multi-frequency inputs: max relative error 5.1e-15.
- Convex-end bound: max[h − bound] = −0.237 (≤ 0).
- c6_integrated.py: max[−log(1 − M/(2B_pP_ν)) − Δ_F] = −1.2e-15 (≤ 0).

### W5 [PROVED]

Sharing exchange identity. Let θ_split be the frequency split (SPLIT R1; all β_ij and ξ_itj unchanged). For every s in [1, ν/2]:
 log P_ν(θ) + (s−1)H_{1/s}(m_θ) = s·log P_b(θ) ≥ s·log P_b(θ_split) = log P_ν(θ_split) + (s−1)H_{1/s}(m_split),
with equality at s = ν/2 iff θ has no DC.
Hence (s−1)[H_{1/s}(m_split) − H_{1/s}(m_θ)] ≤ log(P_ν(θ)/P_ν(θ_split)), with equality at b = 2 when there is no DC.
Reading:
- Neurons shared between features lower the network's Rényi entropy by exactly the extra cost at b = ν when b = 2, and by at most that much when b > 2.
- So sharing is never an entropy loophole on the cost side.
- Since γ(θ;s) ≤ (h(θ)/h(θ_split))·γ(θ_split;s) when h(θ_split) > 0, coupling through shared neurons can gain only through the margin. This is SPLIT's (MM).

**Proof / evidence.**

1. The equalities are W1's algebra applied to any positive vector: s·log Σ a_i^{1/s} = log Σ a_i + (s−1)H_{1/s}(a/Σa).
2. The inequality is SPLIT R1: P_b(θ) ≥ P_b(θ_split) for b ≥ 2, with equality at b = 2 iff there is no DC.
3. Zero pieces carry mass 0 and do not affect the Rényi entropies.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c3_renyi_detect.py, part (X). 40 random DC-free networks at (5,2), (7,3), (11,2), (5,4): identity error 1.8e-15. P_2 and the clean atoms are equal for θ and its split.

### W6 [PROVED]

Spectral Lemma 7, for every n ≥ 1, every r = 1/s in (0,1] and any atom system of one frequency. Notation: β_l = ρ_l p_l with |p_l| = 1; unimodular ξ_lt; A = Σβ_l, L = Σρ_l, Q = Σρ_l^r; gain G = |A|/(ν^{1−s}Q^s), cheap iff G > 1.
(a) Chain matrix H = Σ β_l m_l m_l^*, with eigenvalues λ_k, Λ = ν^{−1}Σ|λ_k| ≥ |A| and μ_k = |λ_k|/(νΛ).
 - Σ_k |λ_k|^r ≤ ν^r Q, hence Q^s ≥ Λ·e^{(s−1)H_{1/s}(μ)} ≥ |A|·e^{(s−1)H(μ)}.
 - So G ≤ (|A|/Λ)·exp((s−1)(log ν − H_{1/s}(μ))): the gain is at most e^{(s−1)·spectral Rényi deficit}.
 - If H − AI is nilpotent (e.g. the system is IF), μ is uniform: Lemma 7.
(b) Twisted Hermitian matrix K = Σ ρ_l u_l u_l^*, with u_{l,0} = 1 and u_{l,t} = p_l Π_{u≤t} ξ_lu for t ≥ 1; π = spec(K)/(νL).
 - Q^s ≥ L·e^{(s−1)H_{1/s}(π)}.
 - Schur–Horn: ρ/L is majorised by π, so H_α(mass shares) ≥ H_α(π) for every α.
 - At ν = 3 this is the GIBBS remark 3^{1−s}Q^s ≥ L·3^{−(s−1)(1−H_3(π))}; this is its form for every n.
(c) L2 form.
 - K_{t0} = β_{[1,t]} exactly, and |K_{tt'} − conj(α)β_{[t'+1,t]}| ≤ sqrt(2L(L−|A|)) for t > t' ≥ 1, with α = A/|A|.
 - G ≤ (|A|/L)·(1 + (2/(νL²))·Σ_{t>t'} |K_{tt'}|²)^{s−1}.
 - With phase-aligned atoms (L = |A|), for every n and every b in [2,ν]: G ≤ (1 + (2/ν)Σ_I |β_I|²/|A|²)^{s−1} ≤ exp(Σ_I |β_I|²/|A|²), summed over the n(n+1)/2 interval coefficients. This is a quadratic payment that does not depend on n.

**Proof / evidence.**

(a)
1. Rotfel'd/McCarthy: ||Σ X_l||_{S_r}^r ≤ Σ ||X_l||_{S_r}^r for 0 < r ≤ 1 and arbitrary matrices. X_l = β_l m_l m_l^* has the single singular value νρ_l.
2. Weyl's majorant theorem: Σ|λ_k|^r ≤ Σσ_k^r, since t ↦ e^{rt} is convex and increasing.
3. Then Σ μ_k^r ≤ (νΛ)^{−r} ν^r Q, i.e. Q ≥ Λ^r Σμ^r. Since s·log Σμ^r = (s−1)H_{1/s}(μ), raising to the power s gives Q^s ≥ Λ e^{(s−1)H_{1/s}(μ)}.
4. H_{1/s} ≥ H_1, and νΛ ≥ |tr H| = ν|A|.
5. If H − AI is nilpotent, every λ_k = A.
(b)
1. Rotfel'd for the PSD rank-one terms ρ_l u_l u_l^* (trace νρ_l), with tr K = νL; same algebra as (a).
2. Schur's theorem on the Gram side V^*V, V = [sqrt(ρ_l) u_l]: its diagonal is (νρ_l), and its nonzero spectrum equals that of K.
(c)
1. (u_l u_l^*)_{t0} = p_l Π_{u≤t} ξ, and (u_l u_l^*)_{tt'} = Π_{u=t'+1}^t ξ.
2. Twist error: |Σρ_l(1 − p_l conj(α))Πξ| ≤ sqrt(L·Σρ_l|p_l − α|²) = sqrt(2L(L−|A|)).
3. H_{1/s}(π) ≥ H_2(π) = log ν − log(1+V), with V = 2Σ_{t>t'}|K_{tt'}|²/(νL²) (Frobenius norm). Hence ν^{1−s}Q^s ≥ L(1+V)^{−(s−1)}.
4. Finally (s−1)·2/ν ≤ 1 − 2/ν < 1.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c1_spectral.py, output c1_spectral.out. 7980 systems (n = 1..6, generic, near-aligned and heavy-tailed atoms, plus atoms of random networks at (5,2), (7,3), (5,4), (11,2), (3,3)). Worst relative excesses (all ≤ 0 up to rounding):
- (a) 6.7e-16 and 8.9e-16;
- (b) 4.4e-16; Schur–Horn 0;
- twist error −3.7e-15; L2 bound 7.8e-15.
IF blocks n = 1..7: chain eigenvalues equal A to 2e-15, and the twisted spectrum is flat. c9_bipartite.py (n = 1..8): prefix entries exact to 5e-15; twist and gain bounds hold to 3e-15.

### W7 [PROVED]

Spectral network inequality, for every network θ in N_S, every p, n and s. For each feature j:
- H_j = max{H_{1/s}(μ_j), H_{1/s}(ρ_j/L_j)}, over any slot order;
- q_j = |A_j|^r / Σ_l |A_l|^r;
- H̃ = (1−r)^{−1}·log Σ_j q_j e^{(1−r)H_j};
- Δ'(θ,s) = log(Γ^{(S)}_s (Σ_j |A_j|^r)^s / h) ≥ 0.
Then
 log γ(θ;s) ≤ log γ_IF^{(S)}(s) + (s−1)[log ν − H̃] − Δ' = log D_S(s) + (s−1)[log ν − H̃] − Δ' − log(πν^{s−1}).
Shared neurons enter only through the budget split Σ_j ρ_ij^r ≤ (B_p a_i)^r.
Winner-width corollary: if γ(θ;s) ≥ G·γ_IF^{(S)}(s) with G ≥ 1, then e^{H̃} ≤ ν·G^{−1/(s−1)}·e^{−Δ'/(s−1)}. A network that beats the IF value of its class contains a feature that is spectrally narrower than an IF block, in every slot order.

**Proof / evidence.**

1. W6(a) and the exact identity Q_j^s = L_j e^{(s−1)H_{1/s}(ρ_j/L_j)} give Q_j ≥ |A_j|^r e^{(1−r)H_j}.
2. The budget gives Σ_j Q_j ≤ B_p^r P_b, so P_b^s ≥ B_p^{−1} (Σ|A_j|^r)^s e^{(s−1)H̃}.
3. Lemma 2 with y_j = |A_j|/(Σ|A|^r)^s gives h ≤ Γ^{(S)}_s (Σ|A_j|^r)^s, i.e. Δ' ≥ 0.
4. Combine, with γ_IF^{(S)} = ν^{1−s}B_pΓ^{(S)} and D_S = πB_pΓ^{(S)}.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c2_network.py, output c2_network.out.
- 91 perturbed IF networks (p = 5, 7, 11; n = 2, 3): min slack −1.8e-15.
- Local optima at (5,2,2), (7,2,2.5), (5,3,2), (7,2,2): slack ≥ 0.
- Exact p = 3 certificates: n = 11 has γ/γ_IF = 1.0544, H̃ = 2.0748 ≈ log 8, entropy term 2.05 = Δ' 0.82 + slack 1.18 + gain 0.053; spectral width 7.96 ≤ 10.08 as the corollary requires. n = 13 is analogous.

### W8 [PROVED]

Entropic width theorem for circular single features, uniform in n.
Hypotheses:
- p is an odd prime with cos(2π/p) > 0 (every p ≥ 5), n ≥ 1, b in [2,ν], and k_p = sec(2π/p) − 1 ≤ √5;
- θ is a single-feature network (outputs at frequency j, DC allowed);
- its field is circular, E(Φ_j − A)² = 0. This holds, e.g., when no input slot carries a 2j-pair (SPLIT R6(c)): class C, or single-frequency inputs with DC;
- its cone margin is positive: c = min_x cm(Φ_j(x)) > 0;
- its atoms are phase-aligned.
Put x = 1 − c/|A| in [0,1). Then:
(i) Σ_{T≠∅} |β_{j,T}|² ≤ k_p x² |A|².
(ii) The effective spectral width satisfies e^{H_2(π)} ≥ ν/(1 + 2k_p x²/ν), and the Rényi-2 effective number of atoms is at least that. It exceeds ν − 2k_p ≥ ν − 2√5 for every n.
(iii) γ(θ;s) ≤ γ_IF^{({j})}(s)·(1−x)(1 + 2k_p x²/ν)^{s−1} ≤ γ_IF^{({j})}(s), with equality iff x = 0, i.e. IF.
So for every n and every b, a cheap circular single feature must have atom cancellation (L > |A|).
Honesty note on the hard rules:
- This is a VALUE statement, so by BRIEF §4 it must use p ≥ 5. It does so only through cos(2π/p) > 0. That is the single arithmetic input; nothing is sent to a limit.
- At p = 3 the cone-moment step is unavailable, and (C) is trivial there.
- It concerns single features (the IF price), not the coupling content of (E).

**Proof / evidence.**

(i) Cone moment bound.
1. Φ − c lies in the sector of half-angle ϑ = π/p, so Re(Φ−c)² ≥ cos(2ϑ)|Φ−c|².
2. Circularity gives E(Φ−c)² = (A−c)². Parseval gives E|Φ−c|² ≥ |A−c|² + Σ_T|β_T|², since the characters 2j·1_T are distinct and nonzero.
3. Write A − c = ϱe^{iφ} with |φ| ≤ ϑ, a = |A|, t = ϱ/a, u = t·cos φ. Then
 Σ_T|β_T|² ≤ a²t²(cos2φ − cos2ϑ)/cos2ϑ = a²(2u² − 2t²cos²ϑ)/cos2ϑ ≤ a²·2u²sin²ϑ/cos2ϑ = a²k_p u².
4. c/a = cos ψ − u ≤ 1 − u (ψ = arg A), so u ≤ x.
(ii) W6(b),(c) with L = a: V = (2/(νa²))Σ_I|β_I|² ≤ 2k_p x²/ν, and e^{H_2(π)} = ν/(1+V). Schur–Horn transfers this to the masses, and ν/(1 + 2k/ν) > ν − 2k.
(iii)
1. Cone lemma: take jδ ≡ ±1 at the minimising x, giving h ≤ κ_min·c.
2. Budget: P_b^s ≥ B_p^{−1}Q^s.
3. W6(c): ν^{1−s}Q^s ≥ a(1+V)^{−(s−1)}.
4. Hence γ/γ_IF ≤ (c/a)(1+V)^{s−1} ≤ (1−x)·exp(k_p x²(1 − 2/ν)) ≤ (1−x)·e^{√5 x²}.
5. Elementary lemma: −log(1−u) ≥ √5u² on [0,1), with equality only at u = 0.
 - F'(u) = 1/(1−u) − 2√5u is negative exactly on (u_1, u_2), where u_{1,2} = (1 ∓ sqrt(1 − 2/√5))/2.
 - So F > 0 on (0, u_1], and F ≥ F(u_2) on [u_1, 1).
 - F(u_2) = 1.086070 − 0.981305 = 0.10476 > 0, evaluated in closed form; no certificate needed.
6. Equality forces x = 0, i.e. Φ ≡ A.
7. k_p decreases in p, and k_5 = √5 because sec 72° = 1 + √5.

*Check:* Scripts in /home/user/grok/heracles/conjecture20/unify/E_width/:
- c8_aligned.py, output c8_aligned.out: 51 positive-margin aligned class-C cases at (5,2), (7,2), (5,3), (7,3), (11,2), (5,4). Cone-L2 slack ≥ 5.6e-3; γ/γ_IF − bound ≤ 1.1e-15; γ/γ_IF ≤ 0.78.
- Optimisation over aligned class-C single features at (5,2,2), (5,3,2), (7,3,2), (5,4,2), (7,2,2.5) reaches 0.99999–1.00000 and never more.
- c10_dc.py: 28 single-frequency networks with DC: cone-L2 slack ≥ 0.43; bound holds.
- c11_aligned_bign.py at (5,5), (5,6), (7,4), (5,7), b = 2: (i) and (iii) hold, worst −1.3e-4.
- c7_sup.py: the envelope sup over (ϱ,φ) is 0 for p = 5, 7, 11, 13, 17, 101 (attained only at ϱ → 0).

### W9 [PROVED]

For every n ≥ 1 and every odd p, a single product neuron has h ≤ 0. This extends the n2mix lemma (n = 2) to every n.

**Proof / evidence.**

1. If h > 0, then Π_t u_t(x_t) ≠ 0 for every x.
2. Every input with a positive product has the strict argmax of w as its label; every input with a negative product has the strict argmin.
3. So only two labels occur, but y takes all p ≥ 3 values. Contradiction.

*Check:* c3_renyi_detect.py, part (O): random and optimised single neurons at (3,2), (5,2), (3,4), (5,3), (7,2), (3,5), (5,4). Max normalised margin −9e-11.

### X1 [REFUTED]

The requested s-local inequality, valid for every network:
 log γ(θ;s) ≤ log D_S(s) + (s−1)[H^dec_S(s) − H(θ,s)] − Δ(θ), with Δ ≥ 0.

**Proof / evidence.**

Counterexample at n = 2, where π = 3^{1−s} (Note Thm q1n2) and every optimal single feature has H = log 3. So D_F = 3^{1−s}B_pΓ_s and H^dec_F = log 3 + min H(z*).
- Take the uniform IF network on all of F at p = 11 (s = 1.35, 1.45, 1.49): log γ − RHS = +0.182, +0.181, +0.378.
- At p = 13: +0.124, +0.232, +0.232.
- At s = 1.2, where the programme optimum is uniform, equality holds.
The network is IF, so any detection term vanishes.
Mechanism: its free energy stays at the s = 1 optimum, log Γ_1 = log(p/(p−1)). The decoupled frontier's free energy is log Γ_1 − ∫_1^s [H(z*(σ)) − H(z*(s))] dσ, which is smaller once H(z*) has dropped at a transition. Only the integrated form W3 can hold.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c4_refute.py, output c4_refute.out. Programme by exact vertex enumeration; the γ of the IF networks is computed from their tables.

### X2 [REFUTED]

The reversed s-local form: log γ(θ;s) ≤ log D_S(s) − (s−1)[H^dec_S(s) − H(θ,s)]. In words, concentration costs at least e^{−(s−1)·entropy deficit}.

**Proof / evidence.**

At (5,2), take the single IF block on feature 1 (H = log 3 < H^dec = log 6). The inequality reduces to κ_min ≤ 5·2^{−2s}. It fails for s > log2(5/κ_min)/2 = 1.4276: log γ − RHS = +0.0033, +0.031, +0.087 at s = 1.43, 1.45, 1.49. So the exchange between entropy and value is not linear in the pointwise deficit either.

*Check:* c4_refute.out

### N1 [NUMERICAL]

The exchange inequality on known extremal networks: single feature, IF reference, b = 2. By W1,
 log(γ/γ_IF) = −Δ_conv + (s−1)(log ν − H_{1/s}).
- p = 3 winners: n = 11 gives −1.9750 + 2.0280 = +0.0530 = log 1.0544; n = 13 gives −1.9158 + 2.0194 = +0.1036. Their concentration gain only barely exceeds their convex-end loss.
- The p = 5, n = 6 width-6 network (LARGEN R10b): −2.1244 + 0.3941 = log 0.1772. Its convex-end loss is about 5 times its concentration gain.
- The p = 3 winners have atom cancellation 1 − |A|/L = 0.228 (n = 11) and 0.222 (n = 13), and DC content (a 2-pair at p = 3). Both escape routes of W8 are present.

**Proof / evidence.**

Exact margins on count vectors (Fractions for the certificates). The decomposition is the identity W1 and matches log(γ/γ_IF) to 1e-4.

*Check:* /home/user/grok/heracles/conjecture20/unify/E_width/c5_exchange.py, output c5_exchange.out; c10_dc.out

### C1 [CONJECTURE]

(DDI_S), the missing exchange inequality. For every odd p, n ≥ 2, b in [2,ν], S and θ in N_S:
 Δ_S(θ) ≥ (s−1)[H̄^dec_S(s) − H_{1/s}(m_θ)].
In words, inefficiency at the convex end pays for Rényi concentration at rate s−1.
- By W2 it is equivalent to (E_S) and to W_S ≡ D_S.
- It is trivial at p = 3: one frequency, so coupled = decoupled.
- Supporting evidence: N1, GLOBAL R11, SPLIT R13 (no network found above D_S at p ≥ 5).
A sub-conjecture: W8 without phase alignment. Every circular single feature has γ ≤ γ_IF for p ≥ 5, every n and every b. Evidence: LARGEN R10 (IF optimal at p = 5, n ≤ 12) and GLOBAL R10.

**Proof / evidence.**

Open. See the obstruction.

*Check:* 

## Obstruction
The precise general obstruction for the width route.

1. EQUIVALENCE.
- Every entropic width statement strong enough for (E_S) is equivalent to W_S ≡ D_S, the restricted price–programme value identity (W2).
- The high-entropy half (H_{1/s}(m_θ) ≥ H̄^dec_S implies γ ≤ D_S) is free (W3). All of the content is the low-entropy half.
- Under (E), maximiser entropy equals H^dec almost everywhere. So no slack exists to absorb small errors in an entropy bound.

2. NO POINTWISE-IN-s FORM.
- The requested inequality with H(θ,s) and H^dec(s) fails in both signs on interference-free networks (X1, X2), so no detection term can save it.
- The exchange is necessarily integrated: Rényi-(1/s) entropy of the masses against the convex-end deficit Δ_S (W1, W3).

3. WHOLE NETWORK VERSUS FEATURES.
- The gain CAN be tied exactly to an entropy deficit of the whole network, but only in the Rényi/convex-end form. There shared neurons are harmless on the cost side: W5 gives the exact exchange at b = 2.
- In the per-feature spectral form (W7), shared neurons are handled by the budget split. There the network's cost-share entropy plays no role and cannot: Schur–Horn gives spectral entropy ≤ mass entropy, the wrong direction for "low network entropy implies spectral deficit".
- So the shared-neuron complication is real only through the margin (mixed modes and compensation), i.e. SPLIT's (MM), which is false at p = 3 and fails for non-maximisers (SPLIT R10).

4. THE MISSING INEQUALITY (C1), Δ_S ≥ (s−1)·(Rényi deficit), must satisfy all of the following.
- (a) It must fail at p = 3. The n = 11 certificate pays 1.975 for a gain of 2.028.
- (b) At b = 2 the rate is s − 1 = (n−1)/2. A network missing a fixed fraction of effective width therefore needs a convex-end loss linear in n (an exponentially small efficiency).
 - One δ-isolated test (W4) certifies at most log 2 for class C.
 - The qualitative width theorems are Hartley-type (they count neurons regardless of mass) and do not force m ≥ ν. LARGEN R10b has width 6 < ν = 7 at p = 5, n = 6, yet satisfies the exchange with fivefold room.
- (c) For circular single features, W8 proves the exchange uniformly in n when atoms are phase-aligned: the payment is quadratic (spectral Rényi-2), against cone-moment L2 detection. The two remaining sources of cheapness are precisely:
 - ATOM CANCELLATION (L > |A|). In W6(c) the n(n−1)/2 non-prefix chain entries carry the twist error sqrt(2L(L−|A|)). This buys up to e^{O(n²)(1−|A|/L)} of spectral room against the factor |A|/L. The certified LARGEN k* growth and the p = 3 winners (22% cancellation) both live here.
 - NON-CIRCULARITY: 2j-pairs, i.e. DC at p = 3, partner or foreign channels at p ≥ 5.
- (d) For several features, the single-feature cone geometry behind W8 is lost. Joint detection is the relations core (O1, GIBBS's L2 route stops at p = 31 at b = 2) together with compensation (O2).

5. UPPER HALF OF (C). Entropy and value identities are first-order. Extra features of vanishing value contribution, especially inside shared neurons at b = 2 (cost-neutral by W5), change neither the value nor the Rényi bookkeeping. No entropic width theorem can exclude them; a strict second-order (local) argument is needed. I found none that is general.

## Next steps
- Cancellation-robust W8. Remove phase alignment for circular single features, for every n. Either find a chain construction whose off-diagonal entries are exact interference coefficients without the sqrt(2L(L−|A|)) twist error, or charge cancellation through higher cone moments. LARGEN R2(c) controls E|Φ−c|^k up to k ≤ (p−1)/2, and W6's Rényi-2 should then be replaced by Rényi-k. Target: γ ≤ γ_IF for every circular single feature at every p ≥ 5, n and b. That is conjecture C1's sub-statement, and it would settle O3 for circular features.
- Multi-feature version of W8 in class C with aligned atoms. Joint fields are proper (SPLIT R6(b)), so Var_x g(·,δ) = Σ_l κ_l(δ)E_l exactly. Since W6(c) makes the payment quadratic (IF price ≥ |A_l| − E_l/|A_l|), it suffices to prove depth E_δ − h ≥ Σ_l κ_l(δ)E_l/|A_l| at the binding δ. That is a joint L2 detection with weights 1/|A_l| and constant 1. Compare with GIBBS's sharp plane Bhatia–Davis (p−1)/(p+1), and try to pass the phase alignment to class-C maximisers.
- Non-circular content. The 2j-pair inputs are exactly SPLIT's 2|F|−1 channels. Combine W5 (cost-neutral sharing at b = 2, strictly paid for b > 2) with the channel classification to price the pseudo-variance term that W8 drops.
- A cost-weighted (Rényi) version of the Note's width theorem for shared-sign 1-D families. Replace Laguerre's sign-change count by a Turán/Nazarov-type lemma for exponential sums, so that missing mass converts into a quantitative loss of cone margin. That would turn the Hartley-type width bound into an exchange inequality of type C1 for that family, including large n at b = 2.
- Run the continuation in the Rényi frame. W3 makes each network's curve log γ(θ;1) − (s−1)H_{1/s}(m_θ) a function of the s-free pair (convex-end efficiency, Rényi profile). Look for a monotone (in s) functional on the convex-end near-optimal set (IF class-C networks with uniform amplitudes, KM2-rigid) that propagates C1 from s = 1, where it holds, using KM2's rigidity for a quadratic stability constant.
- Upper half of (C). Look for a strict second-order argument at maximisers for b > 2, where sharing is strictly paid (SPLIT R1(b)). Show that an extra feature carried in a shared neuron costs first order in its piece energy, while its margin contribution is O(energy^{ν/2}). Then treat b = 2 by continuity of the count, not of the value.
