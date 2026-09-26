import sys, numpy as np
sys.path.insert(0, '..')
from common import random_net, table, gaps, spectrum
rng = np.random.default_rng(0); p = 7; w = np.exp(2j * np.pi / p); a = np.arange(p)
z = table(p, random_net(p, 10, rng, pure=False)); g = gaps(z); Zh = np.fft.fftn(z); G, J = spectrum(z)
for k in (1, 2, 3):
    u = (-2 * k) % p
    Phi_gap = np.array([p ** 3 * (g[:, :, D] * w ** (-u * a)[:, None]).sum() / p ** 2 for D in range(1, p)])
    Phi_Z = np.array([sum(Zh[(u - ga) % p, (-ga) % p, ga] * (1 - w ** (-ga * D)) for ga in range(1, p)) for D in range(1, p)])
    rec = -(1 / p) * sum(Phi_Z[D - 1] * w ** (-k * D) for D in range(1, p))
    print(k, np.abs(Phi_gap - Phi_Z).max(), abs(rec - J['1'][k]), abs(J['1'][k]))
