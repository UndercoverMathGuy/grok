# Unification push: shared brief

Goal of this push: a proof, as general as possible, that the ansatz's feature count K* is the feature count of every
margin-optimal network, for every n, p and b. Fewer cases is better, and elegance helps. A new object only counts if it
does real work in a proof and has been checked.

All repo paths are relative to `/home/user/grok`. "Note" = `heracles/conjecture20/mechanism_note.tex`,
"Unified" = `heracles/conjecture20/unified.tex`, "Companion" = `heracles/conjecture20/interference_framework.tex`,
"Disproof" = `heracles/conjecture20/conjecture20_disproof.tex`, "Paper" = `heracles/paper/iclr2027/sections/{approach,cyclic,appendix}.tex`.

## 1. Setting and notation

- p is an odd prime, rho = e^{2 pi i/p}, F = {1, ..., (p-1)/2} (frequency j is identified with -j), n >= 2 operands,
  nu = n+1, b in [2, nu], s = nu/b in [1, nu/2], r = 1/s = b/nu.
- A product network theta has neurons omega_i = (u_i1, ..., u_in, w_i) in (R^p)^nu and logits
  f(x)[c] = sum_i w_i(c) prod_t u_it(x_t), for x in Z_p^n and c in Z_p. The label is y(x) = x_1 + ... + x_n mod p.
- Cost: ||theta||_{2,b}^b = sum_i ||omega_i||_2^b, with ||omega||_2^2 = sum_t ||u_t||^2 + ||w||^2.
- Gaps: g(x,delta) = f(x)[y(x)] - f(x)[y(x)-delta] for delta != 0. Margin: h = min_{x, delta} g.
  Normalized margin: gamma = h / ||theta||^nu. The optimum gamma*_b is the sup over all widths.
- Coverage: kappa_j(delta) = 1 - cos(2 pi j delta / p).
- Clean coefficient: beta_j = 2 p^{-nu} fhat(j,...,j; -j). For one neuron, beta_ij = 2p^{-nu} conj(what_i(j)) prod_t uhat_it(j).
  Frequency j is a **feature** iff beta_j != 0.
- Sign-flipped coefficient beta_{j,T}, for T a non-empty subset of [n]: input index -j on T. Paper Lemma 6:
  beta_{ij,T} = beta_ij prod_{t in T} xi_itj, with |xi| = 1. This holds for arbitrary inputs.
- Paper Lemma 2: h <= sum_j Re(beta_j) kappa_j(delta) for every delta.
- Single-neuron budget: sum_j |beta_ij|^r <= C_b ||omega_i||^b, with C_b^s = B_p = 2 (2 p nu)^{-nu/2}.
- Programme: Gamma_s(p) = max { min_{delta != 0} sum_j y_j kappa_j(delta) : y >= 0, sum_j y_j^r <= 1 }.
  K*(s,p) = |spt y*|.
- Universal bound: gamma* <= B_p Gamma_s.
- Interference-free (IF) value: nu^{1-s} B_p Gamma_s (Paper Thm 8). It is attained by blocks of nu neurons per feature
  (Prop 9). Lemma 7: an IF feature costs sum_i |beta_ij|^r >= nu^{1-r} |beta_j|^r.
- Feature fields: Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc}. Then g(x,delta) = sum_l Re(Phi_l(x)(1 - rho^{-l delta}))
  and E_x Phi_l = conj(beta_l).
  - Single-feature cone (Companion Lemma 3): margin = kappa_min * min_x cm(Phi(x)), where cm(z) = Re z - cot(pi/p) |Im z|.
  - Slack identity (Note): Phi_j(x) = (2/p)(h - sum_{delta != 0} sigma(x,delta) rho^{j delta}), where sigma >= 0 is the slack.
- Price of a feature: pi_{n,b}(p) is the best single-feature normalized cone margin over B_p.
  - It satisfies nu^{1-s} <= pi <= 1.
  - Price-programme theorem: gamma* >= pi B_p Gamma_s. It is attained by relabelled copies of one single-feature network,
    scaled by y*.

**TARGET (C).** For every odd prime p, every n >= 2 and every b in [2, nu], every global maximiser of the L_{2,b} margin
uses exactly |spt y*| features for some maximiser y* of Gamma_s(p). (Maximisers with different support sizes can coexist at
isolated s, e.g. p=5, s ~ 1.855.)

**KEY REDUCTION (Note, Prop "decoupling suffices").** The price is irrelevant to (C). (C) follows from **decoupling (D)**:
- (D): h(theta*) <= pi B_p min_delta sum_j P_j^s kappa_j(delta), for some split sum_j P_j <= ||theta*||^b with
  beta_j != 0 iff P_j > 0.
- At p=3 (one frequency), (C) is trivially true, although the VALUE conjecture (gamma* = IF) is false there.
- So a proof of the COUNT need not prove the value. That freedom is the main lever of this push.

## 2. What is proved (do not re-prove; cite)

- p=3: (C) holds trivially.
  - Conjecture 20 (value = IF) is FALSE at p=3, b=2, n >= 11 (Disproof; Note Table 2: excess 2.21x at n=18).
  - The mechanism: DC input components make fields non-circular (2 = -1 mod 3).
  - The 1-D shared-sign family, with field e^{i pi tau/p} sum_i V_i e^{lambda_i tau}.
  - Width theorem (Note): m >= nu - floor(nu/p) for shared-sign single-feature networks.
- Prop "no cheap features => exact count" (Note), for all n, p, b. A feature is **cheap** if |beta_l| > a_l = nu^{1-s} Q_l^s,
  where Q_l = sum_i |beta_il|^r.
- Master inequality (MI), and the payment / detection split.
  - Unified: (L) local payment + (G) global detection give (C) for all n, p, b.
  - Class C reduces (G) to a statement about trigonometric polynomials ("no unicorns").
- Dense regime (Note Thm "exact count in the dense regime"), and b in [b_0, nu].
- Count = Theta(log p) for every near-optimal network.
- n=2:
  - Single features are never cheap, at any p and for arbitrary inputs (Note Thm "price of a feature at n=2").
  - Sharp payments: P12, with c_12 = 1 - (2/3)^{s-1} (0.1835 at b=2), and c_1 = 1 - 3^{1-s} (0.4226; computer-assisted).
  - Riesz theorem: cheap features force exact additive relations or spectral coincidences.
  - Class C with (D3) and (I2) dissociation holds, with a 39% surplus (quadratic tests).
  - Class C holds at p in {5, 7}.
  - Relations case (all interference on T={1,2}) holds for p <= 59, and at any p if sum_{L'} kappa < 29.7.
  - Near-IF rigidity holds at every p, for any number of cheap features.
  - Efficiency–Sidon: allocations with e <= 0.759 are closed.
  - Dilution lemma: thin blocks are closed.
- n >= 3:
  - Single-frequency single features cannot be cheap for p >= pi(1 + sqrt2 K_{nu,s}). The constant is factorial and crude.
  - Circularity theorem (Companion Thm 14): when k_p = sqrt(cot^2(pi/p) - 1) >= k*(n,s), single-frequency networks lose.
    k* is known only numerically: 0.35, 0.54, 0.67, 0.76, 0.93 for n = 2, 3, 4, 5, 7 at b=2, and > k_5 = 0.946 from n=8.
- Compensation (Note):
  - Channels lemma: other modes reach a feature l's sign-flipped character 2l 1_T only via the own partner (input 3l) or
    foreign channels (input 2l +- j, output j).
  - Paid compensation is quadratic (Lorentz/Aczel).
  - Chebotarev dichotomy.
  - The single-cheap-feature game is won numerically for p <= 41.

## 3. Open (Note, end of the "Status" section)

- (O1) Relations: at n=2, flat interference polynomials on efficient covers. This is the arithmetic core. Status:
  - The trilemma: no set found that is simultaneously flat, an efficient cover, and small.
  - Flat glued blocks need K ~ 11.6 ln p frequencies at coverage 0.3, against K* ~ 1.4 ln p.
  - Thin/costly/blind trichotomy: the blind case (iii) is open.
- (O2) Compensation with several cheap features. Free shared neurons (F2) need relations of the form 2l = +-j +- q.
- (O3) Single-feature constants for n >= 3.
- (O4) p = 5 at large n. If a multi-frequency-input mechanism won there, it could create spurious features and break (C).

## 4. Dead ends (do not re-propose without naming the new idea that answers why each died)

- One-sided Sidon constants (spikes break them).
- The uniform Lemma' with K_r, in the form "for every u <= tau a" (thin flat blocks break it).
- KKT or local optimality alone. A p=5, n=6, b=2 local maximiser has a cheap feature and a spurious feature
  (mechanism_scripts/kkt/).
- Cost-free compensation. It defeats Theorem A, so costs must be used.
- Linear Riesz products without dissociation at large p. The capacity log_3 p ~ 0.91 ln p is below K*.
- Moduli-only arguments (tilts) under Theorem A's constant.
- Product tests in the relations case (leakage).
- "Stability" in the form "near-optimal covers have few short relations" (false: about 3% efficiency per relation).
- H_Delta ("over-covering a Bohr set is cheap") is false at p=1009.
- Pointwise payments over collision-free families: false at n=4, b=2 (c_1 >= 1.127).
- Any p-free inequality that would imply the IF VALUE at p=3. The p=3 counterexamples refute it. Every candidate value
  inequality must use p >= 5 somewhere, or be a statement about the COUNT only.

## 5. Seed observations from the orchestrator (checked numerically: unify/seed/split_check.py; re-verify before relying on them)

**(S1) Tensor form.** A neuron's table is the rank-one tensor T_i = w_i (x) u_i1 (x) ... (x) u_in.
- After balancing, ||theta||^nu = nu^{nu/2} N_r(f), where N_r(f) = (inf { sum_i ||T_i||_F^r : f = sum_i T_i, T_i rank one })^{1/r}.
- Hence gamma*_b = nu^{-nu/2} sup_f h(f) / N_r(f).
- Read this as maximising a concave, translation-invariant objective h over the unit ball of a translation-invariant,
  NON-convex quasi-norm (r < 1).
- The translation group G = Z_p^n acts by x -> x+a, c -> c + sum a. The G-invariant part of f is exactly its clean part.
- At r=1 (b = nu) the problem is convex, so averaging over G is free: this is Morwani's dense optimum. For r < 1,
  averaging costs.

**(S2) Frequency split lemma.** Let P_k be the orthogonal projection of R^p onto the frequency-k plane span{cos, sin}
(k in F), and P_0 the projection onto constants.
- For each neuron put omega^{(k)} = (P_k u_1, ..., P_k u_n, P_k w), and let theta_split = { omega_i^{(k)} : all i, k in F }.
- Then theta_split is in class C, cost(theta_split) <= cost(theta) for every b >= 2, and theta_split has exactly the same
  beta_j and beta_{j,T} as theta.
- Proof of the cost claim: ||omega||^2 = sum_k ||omega^{(k)}||^2, and sum_k m_k^{b/2} <= (sum_k m_k)^{b/2} for b >= 2.
- The equality case is b=2 with no DC: then mixing frequencies in one neuron is FREE.
- f(theta) = f(theta_split) + (all-DC constant table) + R(theta). Here R is the sum of the **mixed modes** (factor
  frequencies not all equal). R has zero clean part: it is pure interference.

**(S3) Sign-flip averaging identity.** For k in F and a slot t, sigma_{k,t} negates the frequency-k component of input
slot t and of the output, in every neuron.
- These maps generate Sigma = (Z_2)^{F x n}, a group of cost-preserving maps that fix every own-frequency coefficient.
- The average of the tables over Sigma is f(theta_split) plus a constant. So by concavity, h(theta_split) >= E_{sigma}
  h(sigma theta).
- Hence: the max over all networks equals the max over class C if and only if mixed modes never help ("(MM)").
- At p=3 the counterexamples are exactly networks in which mixed (DC) modes help. So (MM) must use p >= 5, through the
  channel arithmetic.

**(S4) Slot-wise product Riesz tests.** Take w(x) = prod_t (1 + eps_t cos(2 pi (2j) x_t/p + psi_t)), with eps_t <= 1.
- For a class C network, E[w g(.,delta)] equals the clean value of every feature l != j. For feature j it equals the
  value of the sign-flip generating polynomial P_j(z) = sum_T beta_{j,T} z^T at z_t = (eps_t/2) e^{-i psi_t}.
- So the test is blind to the other features, with no dissociation hypothesis.
- It sees all T of feature j at once: a polydisc sup-norm detection.
- Collisions appear only when several features are tested in the same slot.

## 6. Tools

Run everything as `uv run --with numpy --with scipy --with mpmath python script.py`. Prefix heavy runs with
`OMP_NUM_THREADS=1`. There are only 4 CPUs, shared by several agents. Keep each computation under about 5 minutes.

- Programme Gamma_s(p), K*, binding set: `heracles/conjecture20/mechanism_scripts/prog.py`.
- Supports at large p: `heracles/conjecture20/mechanism_scripts/relations3/supports.json` and `large_*.json`.
- p=3 counterexamples:
  - `heracles/conjecture20/exact_certificate.py` (weights for n=11 and n=13).
  - `heracles/conjecture20/oned_certified/p3_n*.json`, checked with `certify_1d.py`.
- Single-feature searches: `heracles/conjecture20/interference_scripts/cone.py`.
- Cross-feature compensation at p=5: `mechanism_scripts/joint.py`.
- The KKT local maximiser: `mechanism_scripts/kkt/p5n6_pol.npz`.
- Payments: `mechanism_scripts/payment/`, `nearif/`. Relations: `relations2/`, `relations3/`.
- Falsification: `falsify/`. Unicorn configurations: `unified/unicorn/`.

## 7. Rules

- Labels. Use exactly one per claim:
  - PROVED: complete proof, every step justified, hypotheses explicit.
  - CONDITIONAL: proved from an explicitly stated hypothesis.
  - NUMERICAL: computer search; this is evidence, not proof.
  - CONJECTURE.
  - REFUTED: with an explicit counterexample.
  - A computer-assisted proof must use exact or interval arithmetic and give a re-runnable script.
- Every PROVED claim needs a numerical sanity check on small exact cases. A failed check kills the claim.
- Before claiming anything about the VALUE, check it against the p=3 counterexamples.
- Plain language. Define every symbol at first use, and give a one-line reading of every key formula.
- Write only inside your own directory `heracles/conjecture20/unify/<your-dir>/`. Never edit existing files. No git commands.
- An honest null or a precise obstruction beats an inflated positive.
