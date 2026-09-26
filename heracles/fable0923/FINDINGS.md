# fable0923 FINDINGS: (T) on the pure class at the paper's supports, without any mixture inequality

Date 2026-09-23. Local CPU, fp64 + mpmath interval arithmetic + sympy exact arithmetic. New files only
(this directory). Nothing inherited was edited.

**Status: PROVEN (computer-assisted at two finite steps), pure class, n = 2, q = 2, on every support S whose
K-mode aiming depth is certified >= 0.3476.** This covers all eight supports certified in `HOLES_aiming.md`
(the five `astar_papers` K* supports at p = 59, 71, 97, 113, 127 and the three LADDER supports at p = 59, 71, 113),
in particular **p = 113, where the inherited singleton-partition route is provably insufficient**
(`HOLES_aiming.md` sec 10). No mixture-of-partitions inequality (P3C's PFP-lambda, threshold ~0.304, unproved) is used.
The unrestricted (impure) target (T) is NOT settled; sec 6 says what changes for impure neurons.

## 0. Symbols (plain reading with each)

| symbol | meaning | plain reading |
|---|---|---|
| p | odd prime | the modulus; the task is a + b = c mod p on all p^2 rows |
| nu = 3, r = q/nu = 2/3 | degree, penalty ratio | two operands, plain weight decay |
| omega = e^{2 pi i/p} | root of unity | the task's basic phase |
| S, K = \|S\| | frequency support of the network | which cosines the neurons use; S subset {1..(p-1)/2} |
| pure network | every neuron's three slots are cosines at ONE common frequency k in S | no mixed-frequency neurons, no DC |
| ehat(k) = sum_a e(a) omega^{-ka} | unnormalised DFT of a slot | |
| c_l = ehat_{l1}(k) ehat_{l2}(k) conj(what_l(k)) | atom of neuron l (at its frequency k) | the neuron's signed clean contribution |
| v_{lj} = conj(ehat_{lj}(k))/ehat_{lj}(k) | slot phase datum | unit modulus |
| G_k = sum_{l at k} c_l | clean coefficient | what the level-set average sees |
| J_{k,T} = sum_{l at k} c_l prod_{j in T} v_{lj}, T in {{1},{2},{1,2}} | same-frequency junk | the three sign-flipped Fourier coefficients of feature k |
| eta_k = (sum_T \|J_{k,T}\|^2)^{1/2} | junk size (l2) | |
| cost_k = sum_{l at k} \|c_l\|^{2/3} | per-frequency price | |
| w_{kD} = 1 - cos(2 pi k D/p) | coverage weight | how much offset D is penalised by frequency k |
| Gamma_S = max_{b>=0, sum b=1, supp b in S} min_{D!=0} sum_k b_k^{3/2} w_{kD} | share programme restricted to S | Gamma_S <= Gamma*(p) := Gamma_{3/2}(p) |
| B = 2(6p)^{-3/2} | prefactor | as in THEORY_nsum / ELEGANT |
| junk modes of k | m_{k,1} = (2k,0), m_{k,2} = (0,2k), m_{k,12} = (2k,2k) in Z_p^2 | the row modes at which feature k's junk shows in the gap |
| what(m) = sum_x w(x) omega^{-m.x} | moment of a probability density w on rows Z_p^2 | |
| Omega_S | conv{(omega^{-m.a})_{m junk modes of S} : a in Z_p^2} subset C^{3K} | the achievable junk-mode moment vectors |
| A*(S) | K-mode polydisc inradius (P3A/HOLES): min_{z!=0} max_a Re sum_k z_k omega^{-2ka} / sum_k \|z_k\| | inherited, certified in `HOLES_aiming.md` |
| A_ball(S) | sup{A : Omega_S contains prod_k {t in C^3 : \|\|t\|\|_2 <= A}} | the depth to which ONE density can aim at all three junk patterns of every frequency, l2 per frequency |

## 1. The theorem

> **THEOREM F (pure class, n = 2, q = 2).** Let p be an odd prime and S a support with K-mode aiming depth
> A*(S) >= 0.3476. Then every pure gated-chain network on S (any width, any amount of same-frequency junk) satisfies
>
>     gamma_2 := margin / P_2^{3/2}  <=  3^{-1/2} . B . Gamma_S ,
>
> and the share-weighted 3-neuron common-phase block on a Gamma_S-maximising share vector attains it. If moreover
> A*(S) > 0.3476 strictly, every network attaining the bound is junk-free at every frequency it uses, hence (by
> ELEGANT Thm 1's equality case) a share-weighted block on a Gamma_S-optimal sub-support.

> **COROLLARY.** With the certified depths of `HOLES_aiming.md` sec 6 (min 0.358249 at the old p = 59 support; all others
> >= 0.3734), Theorem F holds on all eight certified supports. On the Gamma_{3/2}-optimal K* supports (where Gamma_S = Gamma*(p))
> this is (T) itself on the pure class, in particular at p = 113 with S = {1,5,12,15,38,46} (A* >= 0.413104).

Plain reading: on the supports the paper actually uses, no pure network -- however many neurons, however it spreads
junk -- beats the constructed block; and anything that ties it is the block.

Threshold arithmetic: the proof needs A_ball(S) >= A_eta where A_eta = 0.272 is a depth at which the population
inequality (sec 2.3) is proved; Lemma 2 gives A_ball(S) >= A'(2 + A')/3 with A' = min(A*(S), 0.47); solving
A'(2 + A')/3 >= 0.272 gives A' >= -1 + sqrt(1.816) = 0.34759.

## 2. Proof structure (three lemmas + assembly)

    Lemma 1 (reduction, general density)   margin <= (2/p^3) min_D sum_k w_kD [ Re G_k + Re sum_T J_{k,T} what(m_{k,T}) ]  for every density w
    Lemma 2 (transfer)                     A_ball(S) >= A'(2 + A')/3,   A' = min(A*(S), 0.47)
    Lemma 3 (population inequality, eta-form, depth 0.272)   cost_k >= 3^{1/3} ( |G_k| - 0.272 eta_k )_+^{2/3}
    Assembly                               Lemma 1 with the aiming density from Lemma 2, then Lemma 3, then Gamma_S and the neuron budget.

### 2.1 Lemma 1 (the reduction with an arbitrary row density)

For every probability density w on Z_p^2 and every offset D != 0, margin <= E_w[gap_D], because every gap_D(x) is a gap
to a wrong class. For a PURE network the gap tensor has Fourier mass only at the clean row mode 0 and the 3K junk
modes, and exactly (checked to 1.9e-13 on five random networks, `reduce_verify.py`, convention verified):

    E_w[gap_D] = (2/p^3) sum_k Re[ (1 - omega^{-kD}) ( G_k + sum_T J_{k,T} what(m_{k,T}) ) ].

Averaging D and -D ((1 - omega^{-kD}) + (1 - omega^{kD}) = 2 w_kD, real):

    margin <= (2/p^3) sum_k w_kD . y_k(w),    y_k(w) := Re G_k + Re sum_T J_{k,T} what(m_{k,T}),   for every D != 0.   (L1)

This is P3C Theorem P3C-1 steps 1-3 with the partition-product family replaced by ALL densities; nothing else changes.
Block check: G_k = 3(p/2)^3 for the unit block and (2/p^3) . w_kD . G_k = (3/4) w_kD = its true gap.

### 2.2 Lemma 2 (transfer from the certified K-mode depth to the l2-polyball depth)

Write z = (z^1, z^2, z^3) in (C^K)^3 for a direction in the 3K junk coordinates (patterns {1}, {2}, {12}) and
h(z) := max_{(a,b)} F(a,b), F(a,b) = f_1(a) + f_2(b) + f_3(a+b), f_i(t) = Re sum_k z^i_k zeta_k^t, zeta_k = omega^{-2k}.
By the support-function characterisation of convex sets (real pairing Re sum_m z_m t_m),

    Omega_S contains the polyball of radius A   <=>   h(z) >= A sum_k ||(z^1_k, z^2_k, z^3_k)||_2  for all z.

*Step 1 (two uses of the K-mode certificate).* (a, c := a+b) is a bijection of Z_p^2, so
h(z) = max_a [ f_1(a) + max_c ( f_3(c) + f_2(c - a) ) ]. For fixed a, c -> f_3(c) + f_2(c-a) is a K-mode function
with coefficients z^3_k + z^2_k zeta_k^{-a}; the certificate gives max_c >= A* sum_k |z^3_k + z^2_k zeta_k^{-a}| >=
A* Re sum_k conj(d_k)(z^3_k + z^2_k zeta_k^{-a}) for any unit vector d fixed in advance. The a-dependent part combines with f_1:
Re sum_k [z^1_k + A* d_k conj(z^2_k)] zeta_k^a, again a K-mode function, whose max over a is >= A* sum_k |z^1_k + A* d_k conj(z^2_k)|. Hence

    h(z) >= A* [ Re sum_k conj(d_k) z^3_k + sum_k |z^1_k + A* d_k conj(z^2_k)| ]   for every unit d,           (*)

and by symmetry the same with the three patterns in any of the 6 role assignments (full, outer, coupled) = (3,1,2)
above. Any A' <= A* may replace A* (the certificate at depth A* implies depth A').

*Step 2 (per-frequency moduli).* With (x, y, w) = (|full|, |outer|, |coupled|) of one frequency, maximising the k-th
term over d gives beta >= y + |x - A'w| (take d = x/|x|: x + y - A'w; take d aligning y with A' d conj(w): y + A'w - x).
Summing (*) over the 6 assignments, each coordinate is "outer" in 2 of them and each ordered pair is (full, coupled)
in exactly 1:

    sum_sigma beta_sigma >= 2(a1+a2+a3) + sum_{i!=j} |a_i - A' a_j|,    a = (|z^1_k|, |z^2_k|, |z^3_k|).

*Step 3 (a 3-variable inequality).* For a >= 0 and 0 <= A' <= 47/100,

    2(a1+a2+a3) + sum_{i!=j} |a_i - A' a_j| >= 2(2 + A') ||a||_2 .                                          (3V)

PROOF (`tri_ineq2.py`, exact): both sides are homogeneous; on the simplex sum a = 1 the 6 planes a_i = A'a_j and the 3
faces cut convex cells on which the left side is linear and the right side is convex, so the difference is concave
per cell and minimised at cell vertices; the 21 distinct vertices are rational functions of A' (listed by the script);
for each, sign changes in A' are located by exact root finding, and on each piece the squared inequality is a
rational function of A' verified nonnegative by exact root counting and endpoint evaluation. Result: PROVED on
[0, 47/100]. (It is FALSE for A' >= 0.4975: vertex (1, A', A')/(1+2A'); hence the cap at 0.47. The exact per-k
optimum beta does not need the cap -- numerically c(A) := min (1/6) sum_sigma beta_sigma/||.||_2 = (2+A)/3 for all A <= 1/2,
`cavg.py` -- but the cap costs nothing here since every certified A* <= 0.4738.)

Averaging (*) over the 6 assignments and using (3V): h(z) >= A' . (2 + A')/3 . sum_k ||z^{(k)}||_2.  QED

Reading: aiming one density at the three junk patterns of every frequency simultaneously costs at most a factor
(2 + A')/3 ~ 0.78 relative to aiming at one pattern per frequency. Numerically the true A_ball(S) equals or exceeds
A*(S) on all ten supports tested (`aball.py`; the worst polyball direction is single-pattern), so the transfer is lossy but safe.

### 2.3 Lemma 3 (the eta-form population inequality at depth 0.272)

For atoms c_l in C, unit v_{l1}, v_{l2}, with G, J_T, eta as in sec 0 and r = 2/3:

    sum_l |c_l|^{2/3} >= 3^{1/3} ( |G| - 0.272 eta )_+^{2/3} .                                             (ETA)

PROOF. Normalise G = 1, x_l = |c_l|, c_l = x_l e^{i alpha_l}, e = sum x_l >= 1, u = sqrt(2e(e-1)). Following
`astra/P3_ANALYTIC.md` sec 1 verbatim with one change: the PSD matrix M = sum_l x_l m_l m_l*, m_l = (1, e^{i alpha}v_1, e^{i alpha}v_1v_2),
has diagonal e, lower entries J_1, J_12 and K := sum x_l v_{l2} with |K - J_2| <= u (Cauchy-Schwarz), so

    tr M = 3e,   tr M^2 = 3e^2 + 2(|J_1|^2 + |J_12|^2 + |K|^2) <= 3e^2 + 2 eta^2 + 4 eta u + 2u^2 =: B(e,eta)

(P3_ANALYTIC bounds this further by 3 delta^2 with delta = max|J_T|; keeping eta^2 is the whole point). Rotfel'd/Jensen and Hoelder
give sum x_l^{2/3} >= 3^{1/3} e^{4/3} (3/B)^{1/3}; also sum x_l^{2/3} >= e^{2/3}. So (ETA) at depth A follows from

    Claim(A):  max( e^{2/3}, 3^{1/3} e^{4/3}(3/B(e,eta))^{1/3} ) >= 3^{1/3}(1 - A eta)_+^{2/3}   on e in [1,sqrt3], eta in [0, 1/A]

(outside: e >= sqrt3 makes the first branch >= 3^{1/3}; eta >= 1/A makes the right side 0). Claim(A) is proved for
A = 0.272 (also 0.275, 0.28) in `eta_cert3.py`:
* Corner lemma (analytic; the equality point (e,eta) = (1,0) sits here and no box test can close it). Cubing, the second
  branch suffices iff Delta := 3e^4 - (1-A eta)^2 B >= 0. With e = 1 + x, u^2 = 2x + 2x^2, the exact expansion (verified symbolically
  with u an independent symbol, `corner_verify.py`) and 4 eta u <= 4 eta^2 + u^2 give
  Delta >= 9x^2 + 6A eta - eta^2 [6 + 3A^2 + 10A^2 x + 7A^2 x^2 + 2A^2 eta^2 + 4A^2 eta u] (the discarded terms
  4A eta^3 + 8A eta^2 u + 14A eta x^2 + 20A eta x + 3x^4 + 12x^3 are all >= 0). On R0 = [0, 0.2] x [0, 0.25] the bracket is
  <= 6.4699 (interval evaluation) and 0.25 . 6.4699 = 1.617 < 1.632 = 6A, so Delta >= eta(6A - eta . bracket) >= 0 on R0.
* Rest of the domain: 5279 boxes, each certified by the monotone test max(e0^{2/3}, 3^{1/3}e0^{4/3}(3/B(e1,eta1))^{1/3}) >
  3^{1/3}(1 - A eta0)^{2/3} evaluated in mpmath interval arithmetic (outward rounding, 25 digits), B increasing in both
  arguments and the right side decreasing in eta. 0 failures.  QED

The true threshold of (ETA) is >= 0.2440 (single-atom population, eta = sqrt3, cost 1 -- this is P3A's rho_need); the PSD
route reaches 0.2701 numerically and 0.272 is what is certified. So Lemma 3 is within 0.03 of best possible.

### 2.4 Assembly

Let the network be pure on S, A' = min(A*(S), 0.47), A_S := A'(2+A')/3 >= 0.272. For each k put
t^{(k)} := -A_S . conj(J^{(k)})/eta_k if eta_k > 0 (else 0). By Lemma 2 the vector (t^{(k)})_k lies in Omega_S, so there is a
probability density w on Z_p^2 with what(m_{k,T}) = t_{k,T} for all k, T, and then Re sum_T J_{k,T} t_{k,T} = -A_S eta_k.
Lemma 1 gives, for every D != 0,

    margin <= (2/p^3) sum_k w_kD y_k,    y_k = Re G_k - A_S eta_k <= |G_k| - 0.272 eta_k ,

and, dropping frequencies with y_k <= 0 (w_kD >= 0), with y+_k = (y_k)_+. Lemma 3 per frequency:
cost_k >= 3^{1/3} (y+_k)^{2/3}. The programme, with shares b_k = (y+_k)^{2/3}/sum_j (y+_j)^{2/3}:

    min_D sum_k w_kD y+_k = (sum_k (y+_k)^{2/3})^{3/2} . min_D sum_k b_k^{3/2} w_kD <= Gamma_S (sum_k (y+_k)^{2/3})^{3/2}
                          <= Gamma_S ( 3^{-1/3} sum_k cost_k )^{3/2}.

Neuron budget: |ehat(k)|^2 <= (p/2)||e||^2 (Parseval, real slot), so |c_l| <= (p/2)^{3/2}(||theta_l||^2/3)^{3/2} (AM-GM), i.e.
||theta_l||^2 >= (6/p)|c_l|^{2/3} and P_2 >= (6/p) sum_k cost_k, tight for balanced pure cosines. Hence

    margin <= (2/p^3) Gamma_S 3^{-1/2} (p/6)^{3/2} P_2^{3/2}  =>  gamma_2 <= 3^{-1/2} . 2(6p)^{-3/2} . Gamma_S = 3^{-1/2} B Gamma_S.  QED

Attainment: ELEGANT Prop 8 (share-weighted block on a Gamma_S maximiser; the same algebra with Gamma_S in place of Gamma).
Rigidity: if A_S > 0.272 and some eta_k > 0 with y_k > 0, then y_k < |G_k| - 0.272 eta_k and the chain is strict; if
y_k <= 0 the frequency has positive cost and zero credit, again strict. So equality forces eta_k = 0 at every frequency
with positive cost, then ELEGANT Theorem 1's equality case (r < 1) applies.

## 3. What is new relative to the inherited state

| inherited | here |
|---|---|
| P3C: partition-PRODUCT densities, moments (t1,t2,t1t2) or (0,0,t); needs the mixture criterion PFP-lambda at A ~ 0.304, unproved; singleton alone provably fails at p = 113 (HOLES_aiming sec 10) | GENERAL densities on Z_p^2; the three junk moments of a frequency are independent; no mixture, no partition |
| P3A/P3C aim at the polyDISC (l1 in the junk coordinates) | aim at the l2-polyBALL, which is what the PSD population inequality actually needs (it controls tr M^2, an l2 quantity of J); the polyball contains the polydisc of the same radius, so its inradius is larger |
| Theorem DA / HOLES_logic: exact rational certificate of the singleton-partition inequality at A = 1/2, 0.47 | the eta-form inequality (ETA) at depth 0.272, proved by P3_ANALYTIC's PSD/Hoelder argument + corner lemma + 5279 interval boxes |
| certified K-mode depths A*(S) (HOLES_aiming) used directly, one pattern at a time | transfer lemma A_ball >= A*(2+A*)/3, so the SAME certificates feed the stronger criterion |
| pure-class theorem certified only for p = 71 (depth 0.47) | pure-class theorem on all eight certified supports, p = 59..127, incl. both p = 113 supports |

The inherited scope limits remain: full table; pure single-frequency neurons; q = 2; n = 2; support fixed to S.

Relation to astra's new PHASE_FRAME_EXTENSION sec 7 obstruction ("no matrix-only argument reaches q = 2"): Lemma 3 IS a
chain-matrix (PSD) argument, but it is only ever applied after the margin side has been priced through a density
chosen AFTER seeing the junk (Lemma 1 + Lemma 2); for pure networks the gap tensor has no discarded components, which
is exactly why the obstruction (built from a full tensor with components Q discards) does not bite here.

Relation to s5theory's group finding (junk hides on groups with a sign-fixed irrep; odd p has none): in the language of
this note, junk of frequency k can be aimed at only through the character omega^{-2k .} at the junk modes; a density's
moment at a REAL character (a sign character, present iff the group has even order) is forced real, so the polydisc
inradius of that mode is 0 in the imaginary direction and imaginary junk there is invisible to every test density.
For odd p no nontrivial character is real, every single junk mode has inradius cos(pi/p), and the only obstruction to
aiming is the arithmetic of S (Theorem R's Dirichlet collapse on dense S). That is a precise reading of "odd order
leaves junk nowhere to hide", and it is the mechanism Lemma 2 exploits.

## 4. Numerical record (all local, fp64 unless stated)

| script | what | result |
|---|---|---|
| `eta_psd.py` | threshold of Claim(A) by dense grid; true (ETA) threshold by population optimisation | A_PSD ~ 0.2701 (binding e ~ 1.03, eta ~ 1.3); truth >= 0.2440 (single atom); ||J||_1-form truth ~ 0.1835 (M_2 population) |
| `eta_cert3.py` (+ `corner_verify.py`) | rigorous proof of Claim(A): corner lemma (interval-evaluated bracket) + adaptive box cover re-checked in mpmath interval arithmetic | A = 0.272: 5279 boxes, 0 failures; A = 0.275: 2075; A = 0.28: 1064; corner algebra verified symbolically (Delta - hand = 0; residual has all-positive coefficients) |
| `eta_cert.py`, `eta_cert2.py` | earlier versions: pure interval bisection cannot close at the corner (e,eta) = (1,0) | superseded |
| `cavg.py` | c(A) = min over C^3 of the 6-assignment average of the exact beta | c(A) = (2+A)/3 to 5 digits for A in [0.30, 0.50], worst direction single-pattern |
| `tri_ineq.py`, `tri_ineq2.py` | the 3-variable inequality (3V): numerics, then exact vertex-enumeration proof | 21 vertices; PROVED on A' in [0, 47/100]; fails from A' ~ 0.4975 at (1,A',A') |
| `aball.py` (`aball_papers.json`) | LP-ascent UPPER bounds on A*, Adisc(3K), Aball(3K) for 10 supports; torus single-frequency limit | A_ball >= A* on 10/10 (e.g. p=113: 0.4752 vs 0.4749; p=71: 0.5478 vs 0.4796); torus: polydisc 0.5000, polyball 0.8660 |
| `reduce_verify.py` | identity (L1) against brute-force gap tensors, random pure networks, random densities, every D | max relative error 1.95e-13; convention: what(m) = sum_x w(x) omega^{-m.x}, atoms c_l = ehat1 ehat2 conj(what) |

Reproduce (from this directory): `uv run --with mpmath --with sympy python eta_cert3.py 0.272`;
`uv run --with sympy python tri_ineq2.py 47/100`; `uv run --with sympy python corner_verify.py`;
`uv run python reduce_verify.py`; `uv run python aball.py --papers --starts 6`; `uv run python cavg.py`.

## 5. What this does NOT do, and honest dependencies

* It depends on the inherited certified lower bounds for A*(S) (`HOLES_aiming.md` sec 6, `holes_aiming_certificate.py`,
  `FINAL_RUNS/HOLES_aiming/`). I did not re-run those certificates; I take them as given.
* It depends on P3_ANALYTIC's sec 1.1 matrix argument (Gram/Jensen + Hoelder), which I re-derived and checked but did not
  re-verify numerically beyond the inherited checks.
* Pure class only. Support fixed. q = 2 (the r-dependence of Claim(A) is easy to re-certify for other r in [2/3, 1) but the
  transfer and thresholds would have to be recomputed; not done).
* The transfer lemma is lossy (factor (2+A*)/3). What would really help is certifying A_ball for supports with SMALL A*,
  but those are exactly the ones where HOLES' enumeration is out of reach.

## 6. Most promising next steps (my ranking)

1. **Impure neurons on the support** (the genuine gap). With general densities, one can additionally demand what = 0 on the
   cross-frequency ("X") row modes that an impure network on +-S can populate, and the same LP ascent computes the
   resulting constrained polyball depth (add free variables on the X-modes to the polar LP). Two things then decide the
   outcome: (a) whether the constrained depth stays >= 0.272 on the paper supports; (b) the collision problem -- junk
   modes 2k that are also X-modes (2k in +-S +- S, e.g. 10 = 15 - 5 at p = 113) cannot be aimed at without charging the
   X-component that shares the mode. (b) is astra's "price the discarded components" in concrete form; the Hoelder-slack
   accounting of IMPURE.md sec 4 is the natural tool. I did not start this.
2. **Close the 0.03 gap in Lemma 3** (0.272 -> 0.244) by a better phase placement in the PSD matrix (the loss is entirely
   |K - J_2| <= u at e slightly above 1); this would lower the required A* from 0.3476 to about 0.32.
3. **q in (2, 3)**: re-certify Claim(A) for r in (2/3, 1); the transfer lemma is r-free; Gamma_{3/q} replaces Gamma_{3/2}.
4. **n >= 3**: the transfer lemma generalises (2^n - 1 patterns, partial sums as coordinates), but the PSD
   population inequality at q = 2 is FALSE for nu >= 4 in max-form (GENERAL_DEPTH sec 6) and its eta-form threshold is unknown.

No background processes were left running.
