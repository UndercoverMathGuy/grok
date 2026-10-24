# Referee check v9 (CC16(5)): Legendre flat design, V_l = |2 chi_hat(l)/(1-rho^{-l})|^2 (as in c4_facts (5)).
# (i) is the pair cone moment vacuous at EVERY delta (not only delta=1)?  (ii) is every off-diagonal entry of
# sum_l V_l Psi^(l) nonnegative (=> no entrywise-nonnegative copositive Q on any Delta detects the aggregate)?
import numpy as np
for p in [61, 101, 401]:
    F = (p - 1) // 2
    chi = np.array([0] + [1 if pow(int(x), (p - 1) // 2, p) == 1 else -1 for x in range(1, p)], float)
    ch = np.fft.fft(chi) / p; l = np.arange(1, F + 1)
    V = np.abs(2 * ch[l] / (1 - np.exp(-2j * np.pi * l / p))) ** 2
    pcm = [(V * (1 - np.cos(2*np.pi*l*d/p)) * np.cos(2*np.pi*l*d/p)).sum() / (V * (1 - np.cos(2*np.pi*l*d/p))).sum() for d in range(1, p)]
    d = np.arange(1, p)
    kap = lambda a: 1 - np.cos(2 * np.pi * np.outer(a, l) / p)   # (len a, F)
    A = d[:, None]; Bm = d[None, :]
    S = (kap(d) @ V)[:, None] + (kap(d) @ V)[None, :] - (kap(((A - Bm) % p).ravel()) @ V).reshape(p - 1, p - 1)
    off = S[~np.eye(p - 1, dtype=bool)]
    print(f'p={p}: pair cone moment fraction: max over delta = {max(pcm):+.4f} (at delta={1+int(np.argmax(pcm))}), at delta=1: {pcm[0]:+.4f};'
          f' min off-diagonal of sum_l V_l Psi^(l) = {off.min():+.4e} (relative {off.min()/np.abs(off).max():+.3f})')
