# End-to-end margin side on a support whose signed sums are distinct (Riesz depth 1/2 in both slots):
#   margin <= (2/p^3) min_D sum_k cov_k(D) m_k,  m_k = min_{|t1|=|t2|=1/2} Re h_k(t1,t2)
# and the assembly constant: (2/p^3) Gamma ((p/6) P / 3^{1/3})^{3/2} = gamma_sym P^{3/2}.
import numpy as np
from common import *
from f3_region import m_of
rng = np.random.default_rng(8); worst = -9
for p, S in ((11, [1, 4]), (13, [1, 3]), (17, [1, 2]), (19, [1, 5, 7])):
    h = (p - 1) // 2
    for t in range(40):
        y = np.zeros(h); y[np.array(S) - 1] = rng.uniform(0.5, 1.5, len(S))
        net = block_net(p, y) + [tuple(rng.uniform(0.1, 0.9) * v for v in n) for n in random_net(p, int(rng.integers(1, 4)), rng, pure=True, kset=S)]
        z = table(p, net); m = margin(z)
        mk = np.zeros(h + 1)
        for k in S:   # per-frequency population: atoms c = e1^(k) e2^(k) conj(w^(k)), v_j = conj(ehat)/ehat
            cs, v1s, v2s = [], [], []
            for e1, e2, w in net:
                E1, E2, W = np.fft.fft(e1)[k], np.fft.fft(e2)[k], np.fft.fft(w)[k]
                if abs(E1 * E2 * W) > 1e-12:
                    cs.append(E1 * E2 * np.conj(W)); v1s.append(np.conj(E1) / E1); v2s.append(np.conj(E2) / E2)
            if cs:
                mk[k] = max(0.0, m_of(np.array(cs), np.array(v1s), np.array(v2s), 0.5, 0.5))
        A = cover_matrix(p)
        bound = 2 / p ** 3 * (A @ mk[1:]).min()
        worst = max(worst, (m - bound) / max(abs(m), 1e-9)); npos = globals().get("npos", 0) + (m > 0); globals()["npos"] = npos
Gam = programme(13, 1.5)[0]; P = 7.3
print(f"max relative (margin - bound): {worst:.2e}  (<= 0 needed); positive-margin nets: {npos}/160")
print(f"assembly constant check: {2 / 13 ** 3 * Gam * ((13 / 6) * P / 3 ** (1 / 3)) ** 1.5:.12f} vs {gamma_sym(13, Gam) * P ** 1.5:.12f}")
