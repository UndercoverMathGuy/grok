# Check: for single-frequency-neuron (pure) product nets, pairing offset D at row x1-h with offset -D at row x1+h
# (h = D/2 mod p) exposes the pattern-{1} junk with the OPPOSITE sign to the same-row pairing, so
# U(y,D) + S(y,D) = 2 C(D) for every row y.  (x2 averaged out.)
import numpy as np, sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def net(p, pure=True, width=12):
    a = np.arange(p); z = np.zeros((p, p, p))
    for _ in range(width):
        k = rng.integers(1, (p - 1)//2 + 1)
        ks = (k, k, k) if pure else tuple(rng.integers(1, (p - 1)//2 + 1, 3))
        ph = rng.uniform(0, 2*np.pi, 3); amp = rng.uniform(0.2, 1.5, 3)
        e1 = amp[0]*np.cos(2*np.pi*ks[0]*a/p + ph[0]); e2 = amp[1]*np.cos(2*np.pi*ks[1]*a/p + ph[1])
        w = amp[2]*np.cos(2*np.pi*ks[2]*a/p + ph[2])
        z += np.einsum('a,b,c->abc', e1, e2, w)
    return z
def check(p, pure):
    z = net(p, pure); A, B = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    Y = (A + B) % p
    gap = np.stack([z[A, B, Y] - z[A, B, (Y + D) % p] for D in range(p)], -1)  # gap[a,b,D]
    inv2 = pow(2, -1, p); worst = {}
    for sgn in (+1, -1):
        err = 0.0
        for D in range(1, p):
            h = (sgn * D * inv2) % p
            C = 0.5*(gap[:, :, D].mean() + gap[:, :, (-D) % p].mean())
            for y in range(p):
                U = 0.5*(gap[y, :, D].mean() + gap[y, :, (-D) % p].mean())
                S = 0.5*(gap[(y - h) % p, :, D].mean() + gap[(y + h) % p, :, (-D) % p].mean())
                err = max(err, abs(U + S - 2*C))
        worst[sgn] = err
    scale = np.abs(gap).max()
    return {s: e/scale for s, e in worst.items()}
for p in (7, 11, 13):
    print(p, 'pure', check(p, True), 'impure', check(p, False))
