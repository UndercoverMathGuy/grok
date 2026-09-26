# Profile-level X-free functional with the PROVED linear eta-form cost bound:
#   Ftilde = B_XB(G,J) / (gamma_sym ((6/p) sum_k 3^{1/3}(|G_k| - 0.272 eta_k)_+^{2/3})^{3/2})
# versus the true F = B_XB / (gamma_sym P^{3/2}), on the block with small phase perturbations (junk spread over patterns).
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
from xblind_bound import bxb
p = 5; Gam, ys = programme(p, 1.5); rng = np.random.default_rng(1)
B = [np.array(n) for n in block_net(p, ys)]
for eps in (0.01, 0.03, 0.1, 0.3):
    worst_t, worst_f = 0, 0
    for t in range(30):
        net = []
        for n in B:
            new = []
            for v in n:
                Fv = np.fft.fft(v); ph = eps * rng.normal()
                Fv[1:3] *= np.exp(1j * ph); Fv[3:5] *= np.exp(-1j * ph)[::-1] if False else np.exp(-1j * ph)
                new.append(np.real(np.fft.ifft(Fv)))
            net.append(tuple(new))
        f, _ = bxb(p, net); z = table(p, net); G, J = spectrum(z)
        eta = np.array([np.sqrt(sum(abs(J[T][k]) ** 2 for T in J)) for k in (1, 2)])
        phi = sum(3 ** (1 / 3) * max(0, abs(G[k]) - 0.272 * eta[k - 1]) ** (2 / 3) for k in (1, 2))
        ft = f * cost2(net) ** 1.5 / ((6 / p) * phi) ** 1.5
        worst_t, worst_f = max(worst_t, ft), max(worst_f, f)
    print(f"phase noise {eps}: max true F {worst_f:.5f}   max Ftilde (eta-form cost) {worst_t:.5f}")
