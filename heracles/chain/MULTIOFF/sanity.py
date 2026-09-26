# Sanity: (1) Body.rowfuncs reproduces the true junk of a pure network's gap; (2) the exposure identity
# min over rows of gap_D = C(D) - dip_D holds exactly for pure nets (so N_lam = sum lam dip_D is the exact best).
import sys, numpy as np, torch
sys.path.insert(0, '..')
from common import *
from mo import Body
rng = np.random.default_rng(1)
for p, T in ((11, [1, 3, 4]), (13, [1, 2, 5, 6])):
    net = random_net(p, 10, rng, pure=True, kset=T)
    z = table(p, net); g = gaps(z); G, J = spectrum(z)
    Jt = torch.tensor(np.array([[J['1'][k] for k in T], [J['2'][k] for k in T], [J['12'][k] for k in T]])) * 2 / p ** 3
    Ds = np.arange(1, p)
    body = Body(p, T, Ds, np.ones(p - 1) / (p - 1))
    f = body.rowfuncs(Jt).numpy()                       # -(junk) per pattern
    x = np.arange(p)
    err = 0; err2 = 0
    for i, D in enumerate(Ds):
        Cpart = 2 / p ** 3 * sum(np.real((1 - np.exp(2j * np.pi * k * D / p)) * G[k]) for k in T)
        negjunk = f[0, i][:, None] + f[1, i][None, :] + f[2, i][(x[:, None] + x[None, :]) % p]
        err = max(err, np.abs((Cpart - negjunk) - g[:, :, D]).max())
        err2 = max(err2, abs(g[:, :, D].min() - (Cpart - negjunk.max())))
    print(f"p={p} T={T}: pointwise gap reconstruction err {err:.1e}; min-gap = C - dip err {err2:.1e}")
