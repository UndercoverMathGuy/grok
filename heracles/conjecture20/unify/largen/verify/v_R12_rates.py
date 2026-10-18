# Side check of the heuristic rate numbers quoted under R12 (not part of its proved identity): for the certified p=3
# 1-D networks and the Note's p=5 1-D networks, compute c/L and (K_r/nu)^{s-1} with masses M_i = B_p ||omega_i||^nu
# (balanced neurons).  For a 1-D neuron, M_i = |V_i| (Z(lam_i)^2/(2p))^{n/2}, Z(lam)^2 = sum_a e^{2 lam e_a}.
# Consistency: c / (nu^{1-s} (sum M_i^r)^s) must equal the stored gamma/gamma_IF (single feature, K* = 1, b = 2).
import json, math, glob
import numpy as np
ROOT = '/home/user/grok/heracles/conjecture20/'
def run(J):
    p, n = J['p'], J['n']; nu = n + 1; s = nu / 2; r = 1 / s; th = math.pi / p
    lam = np.array(J['lam']); V = np.exp(np.array(J['logV']) + 1j * np.array(J['argV']))
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], float)
    Z2 = np.array([np.sum(np.exp(2 * l * e)) for l in lam])
    M = np.abs(V) * (Z2 / (2 * p)) ** (n / 2)
    taus = np.arange(0, (p - 1) * n + 1)          # every tau in 0..L is realised by some count vector
    Phi = np.array([np.sum(V * np.exp((lam + 1j * th) * t)) for t in taus])
    c = (Phi.real - np.abs(Phi.imag) / math.tan(th)).min()
    L = M.sum(); Q = np.sum(M ** r); K = Q ** (1 / (1 - r)) / L ** (r / (1 - r))
    return c / (nu ** (1 - s) * Q ** s), c / L, (K / nu) ** (s - 1), K
for f in sorted(glob.glob(ROOT + 'oned_certified/p3_n*.json'), key=lambda x: int(x.split('_n')[1].split('.')[0])) + \
         [ROOT + 'mechanism_scripts/kkt/oned_p5_n%d.json' % k for k in (3, 4, 6)]:
    J = json.load(open(f)); ratio, cL, rhs, K = run(J)
    print(f"p={J['p']} n={J['n']}: c/IFprice={ratio:.5f} (stored {J['numerical_ratio']:.5f});  c/L={cL:.3e} = e^({math.log(cL)/J['n']:+.3f} n);"
          f"  (K_r/nu)^(s-1)={rhs:.3e} = e^({math.log(rhs)/J['n']:+.3f} n);  K_r/nu={K/(J['n']+1):.3f}")
