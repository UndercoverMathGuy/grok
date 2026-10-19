# (FE) bookkeeping: for balanced, output-DC-free nets sum_l nu E_l = 1; for class-C nets (frequency-pure neurons)
# H(theta) = H(nu E) + sum_l nu E_l H(theta^[l]) (chain rule).  Also: union of the dilation orbit of supports = F.
import numpy as np
from elib import Net, balance, Evec, entropy, freq_proj, Hent, programme_vertices
rng = np.random.default_rng(5); e1 = 0; e2 = 0
for p, n in [(5, 2), (7, 3), (11, 2)]:
    F = (p - 1) // 2; nu = n + 1
    for t in range(10):
        b = rng.uniform(2, nu)
        Q = np.eye(p) - freq_proj(p, 0)
        net = balance(Net(rng.normal(size=(6, n, p)), rng.normal(size=(6, p)) @ Q, b))
        e1 = max(e1, abs(nu * Evec(net).sum() - 1))
        # class C: neuron i at frequency l_i
        ls = rng.integers(1, F + 1, size=7); U = []; W = []
        for l in ls:
            Pl = freq_proj(p, l); U.append(rng.normal(size=(n, p)) @ Pl); W.append(rng.normal(size=p) @ Pl)
        net = balance(Net(np.array(U), np.array(W), b)); E = nu * Evec(net)
        Hs = sum(E[l - 1] * entropy(Net(net.U[ls == l], net.W[ls == l], b)) for l in set(ls.tolist()))
        e2 = max(e2, abs(entropy(net) - Hent(E) - Hs))
print(f"max |sum nu E - 1| (balanced, DC-free) = {e1:.1e};  class-C chain rule max error = {e2:.1e}")
for p, s in [(11, 1.5), (13, 1.5), (7, 1.8), (5, 1.9)]:
    G, maxs, _ = programme_vertices(p, s)
    U = set(); [U.update((np.nonzero(y > 1e-9)[0] + 1).tolist()) for y in maxs]
    Hz = [Hent(y ** (1 / s)) for y in maxs]
    zbar = np.mean([y ** (1 / s) / np.sum(y ** (1 / s)) for y in maxs], 0)
    print(f"p={p} s={s}: union of optimal supports = {sorted(U)}; H(z) on orbit = {np.round(Hz,6)}; "
          f"orbit average zbar has support {int((zbar>1e-12).sum())}/{(p-1)//2}, JS gap H(zbar)-H(z) = {Hent(zbar)-Hz[0]:.4f}")
