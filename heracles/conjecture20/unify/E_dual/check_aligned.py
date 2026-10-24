"""D5' check: class-C single-frequency single features with ALIGNED atoms (common phase).
Field Phi(z) = sum_i conj(beta_i) prod_t (1 + xi_it z_t), z in mu_p^n;  cone level c = min_z cm(Phi(z)).
Checks (a) sum_{T!=0}|beta_T|^2 <= (Re A - c)^2 tau_p, tau_p = (1-cos(2pi/p))/cos(2pi/p)  [circularity, p>=5]
       (b) sum_i (|beta_i|/L)^2 <= 1/nu + (2/nu^2) sum_I |beta_I|^2/L^2                [entropic Lemma 7]
       (c) c <= (K_r/nu)^{s-1} L = nu^{1-s} Q^s  whenever 2(s-1) tau_p <= nu            [no cheap aligned feature]"""
import itertools, numpy as np
rng = np.random.default_rng(3)
worst = {'a': -np.inf, 'b': -np.inf, 'c': -np.inf}; ncpos = 0; tight = np.inf
for trial in range(4000):
    p = int(rng.choice([5, 7, 11, 13])); n = int(rng.integers(2, 6)); nu = n + 1
    if p ** n > 20000: continue
    taup = (1 - np.cos(2 * np.pi / p)) / np.cos(2 * np.pi / p)
    k = int(rng.integers(1, 3)); bl = []; xl = []
    for q in range(k):                               # union of k KM2 blocks + noise, aligned (real positive) atoms
        wq = rng.exponential(1); c0 = np.exp(2j * np.pi * rng.random(n)); om = np.exp(2j * np.pi / nu)
        for l in range(nu):
            bl.append(wq / nu * np.exp(rng.normal(0, rng.uniform(0, 0.6)))); xl.append(c0 * om ** l)
    beta = np.array(bl); xi = np.array(xl) * np.exp(1j * rng.normal(0, rng.uniform(0, 0.4), (len(bl), n)))
    L = beta.sum(); A = beta.sum()
    Z = np.exp(4j * np.pi * np.arange(p) / p)        # z = rho^{2x}
    idx = np.indices((p,) * n).reshape(n, -1).T                 # all z in mu_p^n
    Zs = Z[idx]                                                  # (p^n, n)
    Phi = (np.conj(beta)[None, :] * np.prod(1 + xi[None, :, :] * Zs[:, None, :], axis=2)).sum(1)
    cmin = np.min(Phi.real - np.abs(Phi.imag) / np.tan(np.pi / p))
    # all sign-flipped sums beta_T and interval sums
    bT = {}
    for T in itertools.product([0, 1], repeat=n):
        bT[T] = np.sum(beta * np.prod(np.where(np.array(T)[None, :] == 1, np.conj(xi), 1.0), axis=1))
    S2 = sum(abs(v) ** 2 for T, v in bT.items() if any(T))
    I2 = 0.0
    for k0 in range(n + 1):
        for k1 in range(k0 + 1, n + 1):
            T = tuple(1 if k0 <= t < k1 else 0 for t in range(n)); I2 += abs(bT[T]) ** 2
    worst['b'] = max(worst['b'], np.sum((beta / L) ** 2) - (1 / nu + 2 / nu ** 2 * I2 / L ** 2))
    if cmin > 0:
        ncpos += 1
        worst['a'] = max(worst['a'], S2 - (A.real - cmin) ** 2 * taup)
        for b in [2.0, 2.5, nu - 0.5]:
            s = nu / b; r = 1 / s
            if 2 * (s - 1) * taup <= nu:
                Q = np.sum(beta ** r); rhs = nu ** (1 - s) * Q ** s
                worst['c'] = max(worst['c'], cmin / rhs - 1); tight = min(tight, 1 - cmin / rhs)
print('positive-margin aligned features: %d' % ncpos)
print('(a) max[sum|b_T|^2 - (ReA-c)^2 tau_p] = %.3e   (b) max[collision - bound] = %.3e   (c) max[c/(nu^{1-s}Q^s) - 1] = %.3e (min gap %.3e)'
      % (worst['a'], worst['b'], worst['c'], tight))
