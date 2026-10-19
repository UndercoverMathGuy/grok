# The (E) round: what a general route proves, and the one statement it reduces to

Five routes were each followed by a hostile referee with a generality audit:
- reduction;
- split and replication;
- KKT and critical values;
- entropic width;
- duality.

No PROVED claim was found false. Every result below holds for every odd prime p, every n >= 2 and every b in
[2, nu], with no case split. Reports and verdicts are in `E_<route>/REPORT.md` and `E_<route>/VERIFY.md`; proof
files are `E_reduce/reduce_proofs.tex`, `E_width/E_width_proofs.tex` and `E_dual/e_dual_proofs.tex`.

## 1. One identity organises the problem (E_width W1, E_dual D1)

For every product network with positive margin:

    gamma(theta; s) = gamma(theta; 1) * exp( -(s-1) * H_{1/s}(m_theta) ),      m_i = ||omega_i||^nu / sum_j ||omega_j||^nu .

So the problem at exponent b is the solved convex problem (b = nu) penalised by the Renyi entropy of order
b/nu of the neuron masses. The masses do not depend on s.
- Decoupled networks are product distributions (feature allocation x within-feature masses). For these the Renyi
  entropy is additive.
- So Conjecture 20' (price-programme value) is an additivity statement, of the same type as the
  Amosov-Holevo-Werner multiplicativity question.
- Integrated over s, it is the temperature identity d/ds log gamma = -H.

## 2. General theorems obtained

| statement | source |
|---|---|
| **Equivalence (no free lunch).** For every class S these are all equivalent: (E_S) a.e.; (E_S) for every s < nu/2; (EW_S); the exchange inequality Delta_S >= (s-1)(Hbar^dec_S - H_{1/s}(m)); the restricted value identity W_S = pi B_p Gamma^{(S)} on [1, nu/2] | E_reduce Thm A, E_split E4, E_width W2, E_dual D4 |
| **Convex-end rigidity.** At b = nu the maximisers in N_S are exactly the aligned, IF, class-C block unions with programme allocation, and H* = H^dec | E_split E3 |
| **No cheap feature => decoupled.** For b in (2, nu), every maximiser on N_S with no feature bought below the IF price is a union of Lemma-7 blocks with programme allocation, so it satisfies (E_S). Class-C networks with no cheap feature lose margin linearly in their interference | E_kkt R4, R5 |
| **Static gain bound.** gamma <= B_p M(yhat_theta) K_{1/s}(m_theta)^{1-s}. A network beats D only if it is effectively narrower than its own decoupled projection | E_dual D7 |
| **First-order rigidity** (given the value identities). The output-frequency usage nu E(theta*) of every maximiser is a convex combination of the programme maximisers' budget shares. Consequences: the feature set is a union of optimal supports; the count is exact for a.e. feature weighting; some global maximiser satisfies (C) | E_reduce Thm B, Cor U, Thm G, Cor S |
| **Dilation obstruction.** Value and first-order data cannot exclude a "straddler", i.e. a maximiser whose feature set is a union of >= 2 optimal supports. The upper half of (C) needs a pointwise, strictly concave inequality such as (FE) | E_reduce Prop O, Thm C |
| **Spectral Lemma 7, every n.** Chain-matrix spectra control cost through Rotfel'd, and their entries are interval interference. Aligned circular single features are never cheap, for every n (p >= 5 enters only through cos(2pi/p) > 0) | E_width W6, W8; E_dual D5; E_split E2 |
| One product neuron has h <= 0, every n | E_width W9 |
| Mode budgets with constant (b-2)/(2nu), impossible at b = 2. Shared neurons are cost-neutral in the Renyi bookkeeping, so coupling acts only through the margin | E_split E7, E_width W5 |
| Linear (test-measure) duality is blind to s: no single-tensor dual certificate beats B_p Gamma_1 | E_dual D3b |

## 3. Refuted general forms (certified or exact examples)

- Pointwise-in-s entropy-deficit inequalities, in either sign: IF networks break them (E_width X1, X2).
- An all-network Forster-type inequality with decoupled extremals: fails at (5,2,2) (E_dual D8).
- The restricted identity on amplitude-profile classes (E_dual D9).
- Canonical-split decoupling (SPLIT R10).
- "Neurons are dual best responses" (E_kkt R13).
- (FE) at non-global local maxima (E_reduce verify v4). (FE) is a genuinely global statement.

## 4. Where a general proof of (C) stands

**Theorem T (proved reduction, every p, n, b).**
- Assume (RD^lambda_S): sup over N_S of gamma_lambda <= pi B_p Gamma^{lambda,(S)}_s, for every S and every feature
  weighting lambda near 1. Then:
  - the value gamma* = pi B_p Gamma_s;
  - the lower half of (C);
  - the union-of-supports statement;
  - the exact count for generic weights.
- Adding (FE), H(theta*) >= H(nu E(theta*)) + H^(1)_min (or any other non-straddling statement), gives (C).
- At p = 3 every hypothesis is vacuous. The theorem yields the price value there, not the refuted IF value.

**The core statement is RD**, a value inequality. RD_F is Conjecture 20'. By section 2, no entropic, dual, split or
KKT reformulation is weaker. Any general proof of RD must in particular prove:
1. at n = 2, the class-C IF value at every prime (relations core: "flat + large implies compressible"; open for
   p > 31 at b = 2);
2. for |S| = 1, that foreign zero-mean output fields never help one feature (open even at (5,2); best bound 1.273 D);
3. at n >= 3, control of the two remaining sources of cheapness: atom cancellation and non-circular (2j-pair)
   content.

**Why it is hard, in one line.** Payment adds up over features but detection does not.
- A network's total gain over IF is at most the cost-weighted average of its per-feature gains.
- Any single test measure detects interference either one feature at a time (delta-Fourier isolation) or as an
  L2 aggregate.
- Compensation can make the total interference flat-bottomed in x.
- A general proof therefore needs a *joint* detection inequality that matches the joint payment.
