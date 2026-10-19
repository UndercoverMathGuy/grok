# v0: conventions check. Build explicit neurons for random class-C atom systems (C_joint atom_tables),
# balance them, and compare (i) the field formula of fmlib with E_width Net.fields(), (ii) the margin
# from the explicit table with fmlib's gap formula, (iii) cost: gamma(Net) = B_p h / (sum |C|^r)^s.
import numpy as np
from fmlib import *
from elib import Net, Bp
from cjlib import atom_tables


def balance(U, W):
    U = U.copy(); W = W.copy(); m, n, p = U.shape
    for i in range(m):
        norms = [np.linalg.norm(U[i, t]) for t in range(n)] + [np.linalg.norm(W[i])]
        g = np.prod(norms) ** (1.0 / (n + 1))
        for t in range(n): U[i, t] *= g / norms[t]
        W[i] *= g / norms[n]
    return U, W


rng = np.random.default_rng(1)
worst = dict(field=0, h=0, gam=0)
for (p, n) in [(5, 2), (7, 2), (5, 3), (11, 2), (7, 3)]:
    F = (p - 1) // 2
    for trial in range(6):
        S = sorted(rng.choice(np.arange(1, F + 1), size=rng.integers(1, F + 1), replace=False))
        feats = {}; Us = []; Ws = []
        for l in S:
            m = rng.integers(1, 5)
            C = rng.normal(size=m) + 1j * rng.normal(size=m)
            xi = np.exp(1j * rng.uniform(0, TAU, size=(m, n)))
            feats[l] = (C, xi)
            U, W = atom_tables(p, n, l, C, xi); Us.append(U); Ws.append(W)
        U, W = balance(np.concatenate(Us), np.concatenate(Ws))
        net = Net(U, W); cn = CNet(p, n, feats)
        # factor between explicit field and fmlib field
        fl = net.fields()                                   # (F, p,...,p) Fourier coeffs of Phi_l
        X = cn.X
        for l in S:
            Phi_net = np.fft.ifftn(fl[l - 1]) * p ** n      # back to x-space
            Phi_net = Phi_net.reshape(-1)
            ratio = Phi_net / cn.Phi[l]
            k = np.median(np.real(ratio))
            worst['field'] = max(worst['field'], np.max(np.abs(Phi_net - k * cn.Phi[l])) / np.max(np.abs(Phi_net)))
        kfac = k
        h_net = net.margin(); h_fm = cn.h() * kfac
        worst['h'] = max(worst['h'], abs(h_net - h_fm) / max(1e-12, np.abs(net.gaps()).max()))
        for b in [2.0, 2.5, n + 1.0]:
            s = (n + 1) / b; r = 1 / s
            gam_net = net.gamma(s)
            Qt = sum(cn.Q(r).values())
            gam_fm = Bp(p, n) * cn.h() / Qt ** s
            worst['gam'] = max(worst['gam'], abs(gam_net - gam_fm) / abs(gam_net))
    print(p, n, 'common field factor', kfac)
print('max rel errors:', worst)
