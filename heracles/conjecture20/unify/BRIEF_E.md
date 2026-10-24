# Brief E: a general proof of the count via temperature continuation

**HARD RULES FROM THE USER**
- General proofs only, valid for EVERY odd prime p, EVERY n >= 2 and EVERY b in [2, nu] at once.
- No case analysis by (p, n).
- No "for p >= p_0", "for b near nu" or "for large n" statements.
- No computer-assisted certificates for specific parameters.
- Numerics are allowed ONLY as sanity checks or to look for counterexamples.

Notation, the target (C), the known results and the dead ends are in `unify/BRIEF.md`. The confirmed general tools
are in `unify/STATUS.md` section 3. In particular use:
- the frequency split and Sigma-averaging (SPLIT R1-R9);
- delta-Fourier isolation (p52 R12; any p, n, network);
- the temperature identity, Danskin's theorem and the entropy criteria (GLOBAL R1-R6);
- KM1-KM3.

## 1. The continuation framework (GLOBAL R1-R4, referee-confirmed; restricted-class version stated here)

**Temperature parametrisation.**
- s = nu/b runs over [1, nu/2]: s = 1 is b = nu, s = nu/2 is b = 2.
- For a network theta with h(theta) > 0, put a_i = ||omega_i||^nu (independent of s).
- Then log gamma(theta;s) = log h(theta) - s log sum_i a_i^{1/s}.

**Cost shares and entropy.**
- w_i(theta,s) = a_i^{1/s} / sum_j a_j^{1/s}, and H(theta,s) = -sum_i w_i log w_i.

**Identities.**
- d/ds log gamma(theta;s) = -H(theta,s).
- d^2/ds^2 log gamma(theta;s) = -Var_w(log w)/s <= 0.
- dH/ds = Var_w(log w)/s >= 0.

**Restricted classes.** For S a subset of F, let N_S be the networks with beta_j = 0 for every j not in S (features in S).
- W_S(s) = sup over N_S of gamma(theta;s). It is attained (width reduction keeps the table: GLOBAL R2) and is
  continuous in s.
- By Danskin: D^+ log W_S(s) = -H*_S(s), where H*_S(s) is the minimum entropy over maximisers in N_S.

**Decoupled value.** D_S(s) = pi(s) B_p Gamma^{(S)}_s, where pi(s) = pi_{n, nu/s}(p) is the single-feature price.
- The restricted price-programme theorem gives W_S >= D_S.
- At s = 1 the problem is convex and solved, with pi(1) = 1, so W_S(1) = D_S(1) = B_p Gamma^{(S)}_1.

**Decoupled networks.** Take an optimal single feature theta_1 at s and a maximiser y*_S of Gamma^{(S)}_s, with cost
shares z*_j = (y*_j)^{1/s} / sum.
- Relabelled, scaled copies of theta_1 form a network Theta in N_S with entropy H(theta_1, s) + H(z*).
- So D_+ log D_S(s) >= -H^dec_S(s), where H^dec_S(s) = min over (theta_1, y*_S) of [H(theta_1,s) + H(z*)].

Putting these together, L_S = log W_S - log D_S >= 0 satisfies L_S(1) = 0 and D^+ L_S <= H^dec_S - H*_S.

**(E_S), the target.** For almost every s in (1, nu/2), every maximiser of gamma(.;s) on N_S has cost-share entropy at
least H^dec_S(s). In words: within any feature class, an optimal network is never more concentrated than the most
concentrated decoupled optimum. Coupling cannot win by concentrating.

**What (E_S) gives** (Dini / Vitali-Caratheodory: GLOBAL R3/R4 and the referee fix).
- W_S = D_S on all of [1, nu/2] for every S. In particular gamma* = pi B_p Gamma_s for all p, n, b (Conjecture 20').
- The LOWER half of (C). If theta* is a global maximiser with feature set S*, then
  gamma* <= W_{S*} = pi B_p Gamma^{(S*)}_s <= pi B_p Gamma_s = gamma*. So Gamma^{(S*)}_s = Gamma_s, i.e. S* contains the
  support of some programme maximiser.
- The UPPER half of (C), "no extra features", is NOT given by value identities alone. It needs a separate rigidity
  argument; finding one is part of the task.

**Why this is general.**
- Every quantity above is defined for all p, n, b. Nothing is sent to a limit.
- At p = 3 (one frequency), coupled = decoupled, so (E) is vacuous, consistent with (C) being true there while the IF
  value fails.
- The free-mixing obstruction (p52 R17: shared neurons at b = 2 pay nothing) lives only at the endpoint s = nu/2. For
  b > 2, shared neurons pay strictly (SPLIT R1(b)). Since (E) is needed only for almost every s, the endpoint b = 2
  enters through continuity, at least for the value.

## 2. Equivalent and useful forms (check them)

**Free energy.** F(theta;s) := log gamma(theta;s) + (s-1) H(theta,s).
- F is invariant under replication: k copies, each scaled to keep the cost, give gamma -> k^{1-s} gamma and
  H -> H + log k.
- dF/ds = (s-1) Var_w(log w)/s >= 0.
- At a pair of co-maximisers, (E) is the statement F(theta*) >= F(Theta).

**KKT form (GLOBAL R14 route 4).** At a maximiser with Fritz-John measure q:
- log gamma* = E_w[log t_i] - (s-1) H(w) - (nu/2) log nu, where t_i is the value of the multilinear form E_q[g] at
  neuron i's unit direction.
- Also t_i is proportional to a_i^{(b-nu)/nu}.

**All-network sufficient form (EW_S).** For every s and every theta in N_S: if H(theta,s) < H^dec_S(s), then
gamma(theta;s) <= D_S(s). This "entropic width theorem" implies (E_S), since maximisers have gamma = W_S >= D_S.

**Exact integral identity.** log gamma(theta;s) = log gamma(theta;1) - integral_1^s H(theta,sigma) dsigma, and
gamma(theta;1) <= B_p Gamma_1. Good networks must be concentrated on average. (E) asks for the opposite bound: too
much concentration cannot pay.

## 3. Where spreading comes from (lower bounds on entropy)

Entropy lower bounds must come from the MARGIN CONSTRAINT. Known instances:
- Paper Lemma 7 / KM2: an interference-free feature needs nu equal neurons, so its entropy is log nu.
- Width theorems (Note; LARGEN R1, R4).
- A single neuron has h <= 0 at n = 2 (n2mix lemma).
- The programme: covering every offset needs several frequencies, which gives H(z*).

Coupling mechanisms all lower the entropy:
- shared neurons, i.e. two features in one neuron;
- cheap features with fewer than nu neurons, which carry interference that other features must compensate or that
  must avoid the binding offsets.

(E) says these never pay.
