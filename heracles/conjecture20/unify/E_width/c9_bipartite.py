# Bipartite twist K0 = sum rho_l u_l u_l^*, u_{l,0} = 1, u_{l,t} = p_l prod_{u<=t} xi_lu (t >= 1), any atom system:
#   K0_{t,0} = beta_{[1,t]} exactly;  |K0_{t,t'} - conj(alpha) beta_{[t'+1,t]}| <= sqrt(2L(L-|A|)) for t > t' >= 1;
#   gain G = |A|/(nu^{1-s} Q^s) <= (|A|/L) (1 + 2 sum_{t>t'} |K0_{tt'}|^2/(nu L^2))^{s-1}
import numpy as np
from elib import *
rng = np.random.default_rng(9)
w1 = w2 = w3 = -np.inf
for n in range(1, 9):
    nu = n + 1
    for trial in range(300):
        N = rng.integers(1, 2 * nu + 2)
        spread = [0.0, 0.1, 0.5, 3.0][trial % 4]
        beta = rng.exponential(size=N) * np.exp(1j * spread * rng.normal(size=N))
        xi = np.exp(2j * np.pi * rng.random((N, n)))
        rho = np.abs(beta); pl = beta / rho; L = rho.sum(); A = beta.sum(); al = A / abs(A)
        m = chain_vectors(xi)
        u = m.copy(); u[:, 1:] = u[:, 1:] * pl[:, None]
        K0 = np.einsum('l,la,lb->ab', rho, u, np.conj(u))
        eps = np.sqrt(max(2 * L * (L - abs(A)), 0))
        off2 = 0
        for t in range(1, nu):
            bI = np.sum(beta * np.prod(xi[:, 0:t], axis=1))
            w1 = max(w1, abs(K0[t, 0] - bI))
            off2 += abs(K0[t, 0]) ** 2
            for tp in range(1, t):
                bI = np.sum(beta * np.prod(xi[:, tp:t], axis=1))
                w2 = max(w2, abs(K0[t, tp] - np.conj(al) * bI) - eps - 1e-12 * L)
                off2 += abs(K0[t, tp]) ** 2
        for s in [1.0, 1.4, nu / 2]:
            Q = np.sum(rho ** (1 / s))
            G = abs(A) / (nu ** (1 - s) * Q ** s)
            w3 = max(w3, G / ((abs(A) / L) * (1 + 2 * off2 / (nu * L ** 2)) ** (s - 1)) - 1)
print(f"prefix entries exact: max err {w1:.1e}; non-prefix twist error excess {w2:.1e} (<=0); gain bound excess {w3:.1e} (<=0)")
