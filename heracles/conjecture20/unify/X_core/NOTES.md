# One path: the single-set core RT_n

All workflows are stopped. This is the orchestrator's own work. The path is the single-set core RT_n (BRIEF_PROVE
1(A)), attacked by compression. Every reduction so far ends at RT_n, and any proof of (C) through RD must prove it.

## RT_n, and what "compression" means

Notation:
- a = alpha + beta is the clean allocation; beta is the single-set (cheap) amplitude;
- U(w) = sum_l beta_l cos(2 pi l w/p + phi_l) is the interference;
- D_delta = max_w |U(w) - U(w - delta)| is the lag depth;
- lambda = (n/(n+1))^{1 - 1/s} and c_n = 1 - lambda^s.

RT_n is

    min_delta [ sum_l a_l kappa_l(delta) - D_delta ]  <=  Gamma^{(S)}_s * ( sum alpha^r + lambda sum beta^r )^s .

To prove it, it suffices to COMPRESS: exhibit an allocation a' on S with sum a'^r <= sum alpha^r + lambda sum beta^r and
M(a') >= the left side.

## Results so far (checked in `legendre_check.py` / `.out` and `l2rt.py` / `.out`)

**1. CD_C (C_ccfactor CC17, "exact factorisation at the network's own allocation") is FALSE.**
- Take the Legendre single-set network: U = chi (the Legendre symbol), beta_l = 2/sqrt p on all of F.
- Its left side is exactly sqrt p - 2 (checked for p up to 227).
- CD_C demands that this be at most (1 - c_n) sqrt p. That fails as soon as c_n sqrt p > 2: p >= 127 at (n,b) = (2,2),
  p >= 101 at (3,2), p >= 37 at (10,2).
- So any proof of RD must use the GLOBAL efficiency of the programme (Gamma_s). The network's own allocation is not
  enough. The p <= 13 numerics behind CC14 were too small to see this.

**2. RT_n holds for the whole Legendre family at every prime p = 1 mod 4, for every n and s.**
- Compression: the field row at w = 0 is supported exactly on the quadratic residues (a "QR half" of F).
- Its coverage is (p - chi(delta) sqrt p)/4 >= (p - sqrt p)/4 (checked).
- Hence Gamma_s >= 2^{s-1} (1 - 1/sqrt p) U_s, where U_s is the value of the uniform allocation.
- RT_n then reads sqrt p - 2 <= (2n/(n+1))^{s-1} (sqrt p - 1), which is true.

**3. Field-row averaging gives a universal partial theorem.**
- Averaging over w gives E_w sum_l r_l(w)^r <= sum alpha^r + m_r sum beta^r, with m_r = E_theta (1 + cos theta)^r.
- So PW, and hence RT_n, holds for EVERY U whenever m_r <= lambda.
- Near s = 1 that condition reads (1 - log 2) >= log(1 + 1/n), i.e. n >= 3.
- At b = 2 it needs n >~ 10.
- It fails at small n, which is where the core lives. This refines the "p-uniform gain cap" of C_joint J8.

**4. The hard instance is pinned down.**
- Where: the Legendre networks at SAFE primes p = 3 mod 4 (no even multiplicative subgroup to compress onto), small
  n, and s - 1 ~ 5/sqrt p.
- What is needed: Gamma_s/U_s >= (1+1/n)^{s-1} (1 - 2/sqrt p).
- What that requires: a two-level (+-1) even allocation on F whose cosine sums at all p-th roots are <= about
  2.4 sqrt((p-1)/2).
- This is a flat-Littlewood-polynomial (Rudin-Shapiro type) constant. The best known whole-circle constant for
  truncated Rudin-Shapiro is sqrt 6 ~ 2.45, so the instance is razor-thin. It explains the "safety factor 3.4" of J11e.
- Order-m character allocations (m > 2) are flat but continuous-valued, and their gain m_r^{-s} is too small for n = 2.

**5. [CORRECTED below: these 'violations' were artifacts of a weak Gamma_s estimate] The phase-free L2 version is FALSE.**
- The L2 version replaces D_delta by sqrt(sum beta^2 kappa), a valid lower bound on the depth.
- With an LP lower bound for Gamma it fails: 1.065 at (29,3,2), 1.032 at (23,3,2), 1.011 at (29,2,2.5).
- So RT_n genuinely needs the sup-norm (phase) structure of U at moderate p, i.e. an additive-structure / flatness
  statement: on those supports no phase choice is as flat as its L2 norm suggests.

## Update: the Legendre family is closed, and a candidate general mechanism

- `legendre_theorem.tex` proves that RT_n holds for the Legendre family at every prime p, every n and every s.
  - p = 1 mod 4: the quadratic-residue half.
  - p = 3 mod 4: the sine-weighted quadratic residues, with exact coverage and Fourier control of the cost bias.
  - For s >= 3: a single frequency.
  - The only non-analytic step is one explicit two-variable calculus inequality in (p, s). It is checked on a dense
    grid for real p in [29, 1e9] and handled asymptotically beyond.
- Unified mechanism: both compressions are y_l = beta_l (cos theta_l(w0))_+, the positive part of the field-row
  oscillation at one point (`unified.py`).
- General-interference test (`general_pp.py`, `pp_adversarial.py`, `pp_adversarial2.py`):
  - the compression alpha + c beta (cos theta(w0))_+ (best w0, c) certified RT_n on every configuration found, without
    knowing Gamma_s;
  - with the cheap share pinned at 0.3, 0.6 and 1.0 (p <= 19), the minimum certificate was >= 1.39;
  - unpinned, the adversary only reaches 1.000, at the interference-free equality case.
- Candidate general theorem (the compression theorem): for every (alpha, beta, phi) there are w0 and c such that
  M(alpha + c beta (cos theta(w0))_+) (sum alpha^r + lambda sum beta^r)^s >= LHS (sum y^r)^s. This implies RT_n.

## The compression dichotomy (orchestrator, in progress)

Pure cheap case (alpha = 0). L = min_delta [sum beta kappa(delta) - D_delta]. Two compressions:
- **Linear**, y = beta. Its value is >= L/B^s iff L <= (1 - c_n) M(beta). This holds, for example, if the depth at the
  binding offset of beta is >= c_n M(beta).
- **Positive part**, y = 2 beta (cos theta(w0))_+ = r(w0) - e(w0), where r is the field row and
  e = beta (1 - |cos theta(w0)|) is the "even remainder". By FR, L <= M(r(w0)). Its value is >= L/B^s iff
  (1 - E_cov/L) >= Psi(w0), with E_cov the worst-offset coverage of e and
  Psi(w0) = 2 tau_n (sum beta^r pi^r / sum beta^r)^s.
- **Generic phases.** Psi ~ (2/pi) e^{-(log 2pi - 1 - log(1+1/n))(s-1)} = (2/pi) e^{-0.433(s-1)} at n = 2, and the
  remainder fraction is 1 - 2/pi.
- **The dichotomy.** If linear fails, then L > (1 - c_n) M(beta), and the positive part certifies whenever
  (Q): max_delta sum beta (1 - |cos theta(w0)|) kappa(delta) <= (1 - Psi(w0)) (1 - c_n) min_delta sum beta kappa(delta).
  (Q) is depth-free. It holds when the coverage profile of beta is flat in delta and some w0 makes the phases
  reasonably real (e.g. Legendre at p = 1 mod 4: remainder 0).
- **Gap.** (Q) fails for allocations whose coverage profile is far from flat, e.g. few features. There, depth has to
  be brought back in at the offsets where the coverage is large.
- **Numerics** (`dichotomy.py`): at p = 17..37 the adversary cannot push the combined certificate below ~1.4 (cheap share
  >= 0.4). Small p is easy; the hard regime is large-p flat interference.

## Harmonic analysis of the positive-part compression (orchestrator)

Write |cos t| = (2/pi)[1 + 2 sum_{j>=1} (-1)^{j+1} cos(2jt)/(4j^2-1)], and set U_k(w) = sum_l beta_l cos(k theta_l(w)).

The coverage of y = beta f(theta(w0)) at delta is sum_k fhat_k [U_k(w0) - (U_k(w0 + delta/k) + U_k(w0 - delta/k))/2]:
- k = 0: the clean term fhat_0 sum beta kappa(delta);
- k = 1: the signed field-row term, bounded by the depth D_delta (it is the only harmonic the network controls);
- k >= 2: the phase-multiplied interferences U_k, not controlled by D.

Choices of w0 tried:
- **w0 = argmax U_2 or argmin U_2** makes the k = 2 term favourable (Fejer-type, as C_1 <= 0 was for Legendre). But the
  realised phases then cluster where f is large, so the cost goes up. The trade-off is intrinsic.
- **Random w0.** Coverage concentrates around (2/pi) sum beta kappa with fluctuation ~ (sum beta^2 kappa^2)^{1/2}. A union
  bound over offsets costs sqrt(log p), the same log gap as random halving.

So in the window s - 1 ~ 1/sqrt p with K_eff ~ p, a DERANDOMISED compression is needed. Legendre provides one through
arithmetic (exact profiles). In general it must come from the flatness of U.

Useful fact: depth <= D at ALL offsets is equivalent to osc(U) <= D (take delta = w - w'), which gives sum beta^2 <= 2 D^2.
Fully flat networks are therefore small-oscillation functions. Networks need flatness only at their binding offsets,
however, so the general case mixes the two regimes.


## Correction and the two extremes of the core (orchestrator)

**Item 5 above is withdrawn.** The phase-free L2 core is

    L2-RT: min_delta [ sum (alpha+beta) kappa(delta) - sqrt(sum beta^2 kappa(delta)) ] <= Gamma_s (sum alpha^r + lambda sum beta^r)^s .

Its apparent violations disappear once Gamma_s is computed with a strong multi-start solver (`l2rt_recheck.py`,
`l2rt_strong.py`): 1.008 at (23,3,2) becomes 0.9993. Further evidence:
- the maximum is always at the IF equality case (ratio 1.000);
- with the cheap share pinned at 0.2-1.0 the ratio is <= 0.83 (`l2rt_pinned.py`);
- with beta uniform on subsets of F, p <= 199, s in the window, the ratio is <= 0.80.

L2-RT implies RT_n, because the true depth is at least the L2 depth. It involves only allocations and the programme: no
trigonometric polynomials.

**The proof of the Legendre theorem gives more than it states.** Its programme lemma
(Gamma_s / U_s >= T(s)(1 - c/sqrt p)) does not depend on U. Together with the L2 depth sqrt(p/2) beta, it gives RT_n
for EVERY network whose cheap part is uniform on all of F, for any phases. Only the constants need rechecking with
c = sqrt 2.

**Danger test** (`optsupport_test.py`/`.out`): an all-cheap network sitting on a programme-optimal allocation y*,
which is where the L2 bound is weakest.
- K* grows like log p: 10, 13, 16, 19, 23 for p = 101..1601 at s = 1.5; 27 and 39 at s = 1.25.
- The L2 ratio stays below 1 but creeps up (0.78 -> 0.90). Heuristically it would cross 1 once K* >~ 1/(2 c_n^2), at
  astronomically large p.
- The TRUE depth with the flattest phases found gives a ratio of only 0.07-0.39. Optimal supports behave like Sidon
  sets: no phase choice makes them flat, and the depth is ~ l1, not l2.
- So this is not a counterexample, and not even close.

**Shape of a general proof of RT_n:**
(i) spread allocations: phase-free L2 depth plus programme compression (the Legendre / uniform-on-F mechanism);
(ii) concentrated, efficient allocations: Sidon-type sup-norm depth (Kronecker / Riesz products), with
    depth >= sum beta kappa.
The bridge between (i) and (ii) is the remaining work.

## Attempting L2-RT directly: what it reduces to (orchestrator)

1. **L2-RT cannot be the final general statement.** It is phase-free, and the heuristic says it fails once K* >~ 1/(2 c_n^2)
   on optimal supports (astronomically large p). There the true sup-norm depth (Sidon behaviour) must be used. So a
   general proof of RT_n has to be phase-sensitive somewhere.
2. **Flat interference and low-discrepancy halvings are dual.**
   - A flat U is a phase choice making sum beta_l e^{i phi_l} e^{ilx} small everywhere.
   - A good halving is a sign choice making sum beta_l eps_l cos(2 pi l delta/p) small everywhere.
   - The positive-part compression turns the first into the second: eps_l = sign cos theta_l(w0).
   - Its discrepancy is (signed part) + (even remainder). The signed part is exactly
     U(w0) - (U(w0+delta) + U(w0-delta))/2, which is <= osc(U).
   - **Real phases** (phi_l in {0, pi}): at w0 = 0 the remainder vanishes. The sign pattern of U is then a halving whose
     discrepancy is <= osc(U), which is about the depth. This is the general form of the QR argument (p = 1 mod 4) and
     works for EVERY real-phase U.
   - **Complex phases:** the remainder beta |cos theta(w0)| needs a "positive-type" coverage profile, as the sine
     profile had for Legendre at p = 3 mod 4 (C_1 <= 0). For unstructured phases no single w0 provides this.
3. **So the bridge is now precise.** For complex-phase flat U on a dense support, find a w0 (or a small family of
   points) whose |cos theta_l(w0)| profile has nonpositive cosine transform up to O(depth). Equivalently: the phase
   field of a flat polynomial can be "realified" at some point.
4. **A potential danger to check:** random dense supports S carrying near-flat U in the window s - 1 ~ C/sqrt p. Neither
   arithmetic compressions nor random halvings (sqrt log loss) obviously apply there. Whether near-flat U even exists
   on random dense S is itself the question: flatness and halving discrepancy are governed by the same harmonic
   analysis of S.

## New general results this round (orchestrator)

- **Theorem (real phases)**, `compression_theorems.tex`: if the interference U is symmetric about some point w0, i.e.
  all phases are real there, then RT_n holds for EVERY p, n, s, alpha, beta.
  - Proof in two lines: take y = alpha + beta (1 +- eps) with eps = sign pattern of U at w0, whichever half is cheaper.
  - Coverage is >= L by the signed field-row identity.
  - Cost is <= sum alpha^r + 2^{r-1} sum beta^r <= budget, because 2^{r-1} <= lambda is equivalent to 1/2 <= n/(n+1).
  - Checked on 3000 random real-phase configurations: min ratio 1.003 (`realphase_check.py`).
  - This contains Legendre at p = 1 mod 4.
- **Even-part compression** (complex phases, any w0): RT_n holds if (E) holds. The loss is the remainder
  sum beta (1 - |cos theta(w0)|) kappa(delta).
- **Uniform cheap part on all of F, ANY phases** (depth >= L2 = sqrt 2 in Legendre units):
  - the explicit bounds (sine-QR, QR, single frequency) prove it for all p >= 81;
  - for 29 <= p <= 80 they are too lossy in narrow s-windows (`uniformF_anyphase2.out`), and sharper estimates are
    needed there;
  - this is a bonus result, not needed for the main chain.
- **What remains for RT_n:** complex-phase interference on general supports. A single w0 must keep the even-part
  remainder below its average (1 - 2/pi) sum beta kappa, up to O(depth), at the binding offsets. A random w0 loses
  sqrt(log p), so derandomisation is needed. This is "realifying" the phase field of a flat polynomial.
