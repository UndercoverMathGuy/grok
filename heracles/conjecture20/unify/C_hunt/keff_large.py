# Uniform-eta safety factor of the L2 relaxation at the STORED b = 2 programme optima for large p (relations3/*.json):
#   ell = max_{delta in B} (sum y*^2 kappa)^{1/2} / Gamma, K_eff = 1/ell^2, sigma = theta(s) / (rho_p * ell),
#   theta(3/2) = (1 - 3^{-1/2})/sqrt3 (sharp L2 payment, GIBBS R6), rho_p = sqrt((p-1)/(p+1)) (sharp joint detection, GIBBS R3).
#   sigma < 1  <=>  uniform L2-cheapening of the optimum does not pay in the relaxation.
import json, glob, numpy as np
D = '/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/'
rows = []
sup = json.load(open(D + 'supports.json'))
for k, v in sup.items(): rows.append((int(k), v))
for f in glob.glob(D + 'large_*.json'):
    d = json.load(open(f))
    for k, v in d.items():
        if k.isdigit() and isinstance(v, dict) and "y" in v: rows.append((int(k), v))
s = 1.5; theta = (1 - 3 ** (1 - s)) / np.sqrt(3)
seen = set()
for p, v in sorted(rows, key=lambda z: z[0]):
    if p in seen: continue
    seen.add(p)
    S = np.array(v['S']); y = np.array(v['y']); G = v['Gamma']
    F = (p - 1) // 2; d = np.arange(1, F + 1)
    K = 1 - np.cos(2 * np.pi * np.outer(d, S) / p)
    cov = K @ y
    B = np.nonzero(cov < cov.min() * (1 + 1e-6))[0]
    L2 = np.sqrt((K[B] * y[None, :] ** 2).sum(1))
    ell = L2.max() / cov.min()
    rho = np.sqrt((p - 1) / (p + 1))
    print(f"p={p:6d} K*={len(S):2d} Gamma(stored)={G:.5f} min cov={cov.min():.5f} |B|={len(B):2d} K_eff={1/ell**2:6.2f} sigma={theta/(rho*ell):.3f}")
