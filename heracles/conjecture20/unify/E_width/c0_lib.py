# Convention checks for elib.py
import numpy as np, itertools
from elib import *

rng = np.random.default_rng(0)
for (p, n, m) in [(5, 2, 4), (7, 2, 3), (5, 3, 3), (3, 2, 4), (7, 3, 2)]:
    net = Net(rng.normal(size=(m, n, p)), rng.normal(size=(m, p)))
    F = (p - 1) // 2
    beta, xi = net.atoms()
    cl = net.clean(); sf = net.signflip(); ph = net.fields()
    e1 = np.max(np.abs(cl - beta.sum(0)))
    e2 = max(abs(sf[(j, T)] - np.sum(beta[:, j - 1] * np.prod(np.where(np.array(T)[None, :] == 1, xi[:, :, j - 1], 1), 1)))
             for j in range(1, F + 1) for T in itertools.product([0, 1], repeat=n))
    e3 = max(abs(ph[l - 1][(0,) * n] - np.conj(cl[l - 1])) for l in range(1, F + 1))
    e4 = max(abs(ph[l - 1][tuple((2 * l * T[t]) % p for t in range(n))] - np.conj(sf[(l, T)]))
             for l in range(1, F + 1) for T in itertools.product([0, 1], repeat=n))
    # single-neuron budget sum_j |beta_ij|^r <= (B_p a_i)^r at r = 1/s
    for s in [1.0, 1.3, (n + 1) / 2]:
        r = 1 / s
        lhs = (np.abs(beta) ** r).sum(1); rhs = (Bp(p, n) * net.mass()) ** r
        assert np.all(lhs <= rhs * (1 + 1e-12)), (p, n, s)
    print(f"p={p} n={n}: clean=sum atoms {e1:.1e}; signflip=atoms*xi {e2:.1e}; phi(0)=conj beta {e3:.1e}; phi(2l1_T)=conj beta_T {e4:.1e}; budget ok")

# IF networks attain gamma_IF = nu^{1-s} B_p Gamma_s
for (p, n, s) in [(5, 2, 1.5), (7, 2, 1.2), (5, 3, 1.7), (11, 2, 1.45)]:
    G, ys = programme_vertices(p, s)
    U, W = if_network(p, n, ys[0])
    net = Net(U, W)
    g = net.gamma(s); gif = (n + 1) ** (1 - s) * Bp(p, n) * G
    print(f"IF network p={p} n={n} s={s}: gamma/gamma_IF = {g / gif:.10f}; y* = {np.round(ys[0], 5)}; #maximisers {len(ys)}")
# programme sanity: p=5, Gamma_s = (5/2) 2^{-s}; p=7, (7/2) 3^{-s}; Gamma_1 = p/(p-1)
for s in [1.0, 1.2, 1.5]:
    print(f"s={s}: Gamma(5)={programme_vertices(5, s)[0]:.10f} vs {(2.5) * 2 ** -s:.10f};  Gamma(7)={programme_vertices(7, s)[0]:.10f} vs {3.5 * 3 ** -s:.10f}")
for p in [5, 7, 11, 13]:
    print(f"Gamma_1({p}) = {programme_vertices(p, 1.0)[0]:.12f}  p/(p-1) = {p / (p - 1):.12f}")
