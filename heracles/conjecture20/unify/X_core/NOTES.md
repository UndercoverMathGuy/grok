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

**5. The phase-free L2 version is FALSE.**
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
