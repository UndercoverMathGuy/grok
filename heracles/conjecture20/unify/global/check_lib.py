# Validation of lib.py: IF networks attain gamma_IF; gradient check; atoms vs clean/sign-flipped coefficients; split.
import numpy as np, itertools
from lib import *

rng = np.random.default_rng(0)
print("== programme: reweighted LP vs exhaustive")
for p in [5, 7, 11, 13]:
    for b_s in [1.2, 1.5, 2.0, 2.5]:
        G, y, _ = programme(p, b_s)
        ex = exhaustive_programme(p, b_s)
        Gex = max(v[0] for v in ex.values())
        kbest = [k for k, v in ex.items() if abs(v[0] - Gex) < 1e-9 * Gex]
        print(f"p={p} s={b_s}: LP {G:.6f} exhaustive {Gex:.6f} K*={kbest} supp={[ex[k][1] for k in kbest]} y={np.round(ex[kbest[0]][2],4)}")

print("== IF network attains gamma_IF")
for (p, n, b) in [(5, 2, 2.0), (7, 2, 2.0), (7, 3, 2.0), (11, 2, 2.5), (5, 3, 3.0)]:
    s = (n + 1) / b
    ex = exhaustive_programme(p, s)
    Gex = max(v[0] for v in ex.values()); k = [k for k, v in ex.items() if v[0] == Gex][0]
    yS = ex[k][2]; S = ex[k][1]; F = (p - 1) // 2
    y = np.zeros(F); y[np.array(S) - 1] = yS
    net = if_network(p, n, y, b)
    print(f"p={p} n={n} b={b}: gamma/gamma_IF = {net.gamma()/gamma_if(p,n,b,Gex):.10f}  width {net.m}  "
          f"max|beta_jT| {max(abs(v) for (j,T),v in net.signflip().items() if any(T)):.2e}  |beta|={np.round(np.abs(net.clean()),5)} y*={np.round(y,5)}")

print("== gradient check of softgamma")
for (p, n, m, b) in [(5, 2, 4, 2.0), (5, 3, 3, 2.7)]:
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p)); x = pack(U, W)
    F0, g0 = softgamma(x, m, n, p, b, 0.05)
    e = rng.normal(size=x.size); h = 1e-6
    Fp, _ = softgamma(x + h * e, m, n, p, b, 0.05); Fm, _ = softgamma(x - h * e, m, n, p, b, 0.05)
    print(f"p={p} n={n}: fd {(Fp-Fm)/(2*h):.8e} analytic {g0@e:.8e}")

print("== atoms, Lemma 6 and split")
for (p, n, m, b) in [(5, 2, 4, 2.0), (7, 2, 3, 2.5), (5, 3, 3, 2.0)]:
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p)); net = Net(U, W, b)
    at = net.atoms(); cl = net.clean(); sf = net.signflip()
    d1 = np.abs(at.sum(0) - cl).max()
    # Lemma 6: beta_{ij,T} = beta_ij prod_{t in T} xi_itj with xi = conj(uhat)/uhat
    Uh = np.fft.fft(U, axis=2); F = (p - 1) // 2; d2 = 0
    for j in range(1, F + 1):
        xi = np.conj(Uh[:, :, j]) / Uh[:, :, j]
        for T in itertools.product([0, 1], repeat=n):
            v = (at[:, j - 1] * np.prod(np.where(np.array(T)[None, :] == 1, xi, 1), axis=1)).sum()
            d2 = max(d2, abs(v - sf[(j, T)]))
    sp = split(net); d3 = max(abs(sp.signflip()[k] - sf[k]) for k in sf)
    print(f"p={p} n={n}: |sum atoms - clean| {d1:.1e}  Lemma6 dev {d2:.1e}  split coeff dev {d3:.1e}  cost {net.cost():.4f} split {sp.cost():.4f}")
