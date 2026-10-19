"""Static gain bound at p = 3 (certified 1-D networks, b = 2): gamma/gamma_IF <= (nu / K_r(p_theta))^{s-1}."""
import json, numpy as np
from elib import renyi, shannon, shares
for nn in range(11, 19):
    d = json.load(open('/home/user/grok/heracles/conjecture20/oned_certified/p3_n%d.json' % nn))
    p, n = d['p'], d['n']; nu = n + 1; s = nu / 2.0; r = 1 / s
    lam, lv, av = map(np.array, (d['lam'], d['logV'], d['argV'])); lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], dtype=float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    a = []
    for i in range(len(lam)):
        u = eps * np.exp(lam[i] * e); w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        t2 = (np.linalg.norm(w) * np.linalg.norm(u) ** n) ** (2.0 / nu)
        a.append((nu * t2) ** (nu / 2))
    a = np.array(a); pp = a / a.sum(); Kr = np.exp(renyi(pp, r))
    print('n=%d  gamma/gIF=%.4f  K_r=%.4f  bound (nu/K_r)^(s-1)=%.4f  H(w_s)=%.4f log nu=%.4f' %
          (nn, d['numerical_ratio'], Kr, (nu / Kr) ** (s - 1), shannon(shares(a, s)), np.log(nu)))
