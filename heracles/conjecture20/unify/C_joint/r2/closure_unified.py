# Unified closure: aligned class-C (single-set blocks on arbitrary sets, IF parts), gain_l <= w_l sqrt(E_l),
# w_l = min(c_n, C theta_l); closure if sum_l kappa_l(d*) w_l^2 <= psibar(d*)^2, psibar = (psi(t_d*) + psi(t_-d*))/2.
import numpy as np, itertools
from closure_check import Fn, setup, margin_parts
psi = lambda t: (np.sqrt(t*t+4) - t)/2
rng = np.random.default_rng(5); tested = 0; worst = -1e9; inter = -1e9; skewed = 0
for trial in range(1500):
    p = int(rng.choice([7, 11, 13])); n = int(rng.choice([2, 3])); nu = n+1; F = (p-1)//2; s = rng.uniform(1, nu/2)
    sets = [T for k in range(1, n+1) for T in itertools.combinations(range(n), k)]
    k = int(rng.integers(2, min(F, 6)+1)); ls = rng.choice(np.arange(1, F+1), size=k, replace=False)
    # bias towards common sets to create additive triples / skew
    fam = [(int(l), sets[rng.integers(min(len(sets), 2))] if rng.random() < 0.5 else sets[rng.integers(len(sets))]) for l in ls]
    X, Z, KAP, CD = setup(p, n, fam)
    al = rng.exponential(size=k); be = rng.exponential(size=k)*rng.choice([0.1, 0.5, 2.0]); ph = rng.uniform(0, 2*np.pi, k)
    h, mean, D = margin_parts(al, be, ph, Z, KAP, CD)
    C = 2*(s-1)/nu; cn = 1 - (n/nu)**(s-1); L = al+be; th = be/L; at = L*Fn(th, n, s)
    ds = int(np.argmin(KAP @ at)); dm = p-2-ds
    sig = np.sqrt(np.mean(D[ds]**2)); tp = np.mean(D[ds]**3)/sig**3; tm = np.mean(D[dm]**3)/sig**3
    pb = (psi(tp) + psi(tm))/2
    inter = max(inter, (h - (KAP[ds] @ L - sig*pb))/(KAP[ds] @ L))
    w = np.minimum(cn, C*th)
    if KAP[ds] @ w**2 <= pb**2:
        tested += 1; worst = max(worst, h/(KAP[ds] @ at)); skewed += (abs(tp) > 1e-6 or abs(tm) > 1e-6)
print(f"intermediate bound h <= sum L kappa - sigma psibar: max violation {inter:.2e} (<= 0)")
print(f"{tested} networks meet the unified closure hypothesis ({skewed} with nonzero skew): max h / sum at kappa(d*) = {worst:.5f} (<= 1)")
