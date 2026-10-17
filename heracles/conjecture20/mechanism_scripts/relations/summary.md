# Relations in the Riesz argument (fork R)
Literature
- Riesz products: unique {0,+-1}-representations give nu = prod(1-cos(chi_k+phi_k)) >= 0, E nu = 1, and one-sided constant 1/2
  (Rudin; Bourgain, Ann. Inst. Fourier 35 (1985) 137). Peyriere (arXiv:1003.4600) gives two-sided Sidon constant <= 4.27 for one quasi-independent set.
- Pisier (Bull. AMS 8 (1983) 87), Bourgain (1985), Hare-Yang (arXiv:1808.03128): Sidon <=> proportionally quasi-independent.
  These are qualitative only: no explicit one-sided constants for |S| ~ log p in Z_p.
- Chang's lemma and dissociated spectra (arXiv:0712.1074, math/0610554) bound structure from above; they do not give the lower bound needed here.
- Tao/Biro/Meshulam: |spt f|+|spt f^| >= p+1 on Z_p (arXiv:math/0308286), equivalent to Chebotarev. Exact only; I found no explicit robust form.
Proved (forkR/lemma.tex)
1. Quantitative Riesz: min Y <= -sum a_k(eps/2 - W_k)/(1+W_0), with W the exact weighted relation counts.
2. Coverage does NOT bound the one-sided constant. S=F with Gauss-sum phases has |D_delta| <= 2 sqrt(p)(1+ln p),
   while sum kappa_l(delta) = p/2, so the constant is <= 4(1+ln p)/sqrt(p) -> 0. A valid lemma must use |S| << p.
3. The 1-D relaxation is exact for a realisable family: two aligned neurons per feature with xi_2 = -xi_1. All interference is then on
   T={1,2}, |beta_12| = |beta| = (3/2)^{s-1}a, and the margin depends only on x1+x2. IF is safe iff phases cannot push the
   multi-direction one-sided constant below 1-(2/3)^{s-1} (0.1835 at b=2).
4. Floors:
   - If all cheap interference sits on one T, the IF bound holds when sum_{L'} kappa(delta*) <= 12 (Thm 23 had 4).
   - In general the multi-direction one-sided constant is >= 1/sqrt(min_B sum_S kappa).
Numerics (s=1.5, programme supports; relw.py, onesided.py)
- eps-Riesz, single T (1-D, the network's worst case):
  - eps=1: W_0 = 1.4 / 9.3 / 18.5 / 326 at p = 31 / 101 / 211 / 401.
  - eps=0.6: the guaranteed constant is negative from p=31 on.
- Greedy 2-D T choice (only if the proof may choose T): constant 0.43 for p <= 211, but it fails at p=401.
  So eps-Riesz cannot reach the K* ~ 3 ln p scale.
- True multi-direction one-sided constant (adversarial phases and amplitudes; local search, so upper bounds):
  0.83 (p=31), 0.79 (p=101). Runs for p=211 and 401 are still writing os211.out and os401.out.
- Rigorous L2 floor: 0.49 / 0.36 / 0.35 / 0.28 (p = 31 / 101 / 211 / 401).
- Requirements: 0.1835 (Lemma-3 family), 0.289 (single T, Theorem A), 0.5 (multi-T, Theorem A with max_T).
  - The floor proves the Lemma-3 family IF-optimal whenever min_B sum kappa <= 29.7. Extrapolating the computed sums (about 2.3 ln p),
    I estimate that covers p up to ~1e5.
  - It proves single-T networks safe for p <= 211.
Verdict
- Not closable by exact dissociation (capacity n log_3 p < K*), by eps-Riesz products (the relation weights explode), or by a coverage-only lemma.
- The true constant (~0.8) exceeds every requirement by 1.6-4.4x, so the gap is in proof technique, not in the truth of the statement.
- What is needed: a uniform lower bound, >= 0.29 (single T) or >= 0.5 (general), on max_{delta in B}(-min_w D_delta)/sum a_l kappa_l(delta)
  over sparse polynomials with |S| ~ 3 ln p and adversarial phases.
- L2/L4 (Paley-Zygmund, additive energy) arguments only reach the sqrt(K) scale, i.e. the floor in item 4. Beating it needs extreme-value
  information about (lw/p)_{l in S} at p ~ e^{K/3} points, where equidistribution in T^K is unavailable.
- What is realistic: a per-prime certificate (an exact LP over phases on a grid) for any finite range of p, plus the L2 floor where it suffices.
