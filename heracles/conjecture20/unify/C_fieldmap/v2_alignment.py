# v2: Theorem R (alignment). Per-feature translations tau_a: Phi_l(x) -> Phi_l(x + a_l) are class-C,
# keep every atom modulus (xi_it -> xi_it rho^{2 l a_t}), hence every feature cost Q_l, and
#     min_a h(tau_a N) = h_prof(N) := min_d sum_l min_x Re(Phi_l(x) c_l(d)).
# Brute force over all alignments (first feature fixed by global translation invariance).
import numpy as np, itertools
from fmlib import *

rng = np.random.default_rng(3)
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

maxerr = 0; ncase = 0; costerr = 0
for (p, n, k) in [(5, 2, 2), (7, 2, 2), (5, 3, 2), (7, 2, 3), (11, 2, 2)]:
    F = (p - 1) // 2
    for trial in range(8):
        S = sorted(rng.choice(np.arange(1, F + 1), size=k, replace=False))
        feats = {}
        for l in S:
            m = rng.integers(1, 4)
            C = rng.normal(size=m) + 1j * rng.normal(size=m)
            if trial % 2 == 0:                      # near-IF features with a perturbation
                C0, xi0 = if_atoms(n, 1.0 + rng.random())
                C = np.concatenate([C0, 0.3 * C]); xi = np.concatenate([xi0, np.exp(1j * rng.uniform(0, TAU, (m, n)))])
            else:
                xi = np.exp(1j * rng.uniform(0, TAU, (m, n)))
            feats[l] = (C, xi)
        net = CNet(p, n, feats); X = net.X
        Gs = {l: net.G(l) for l in S}                 # (N, p-1)
        idx = lambda a: np.ravel_multi_index(((X + a[None, :]) % p).T, (p,) * n)
        hs = []
        shifts = list(itertools.product(range(p), repeat=n))
        for combo in itertools.product(range(len(shifts)), repeat=k - 1):
            g = Gs[S[0]].copy()
            for j, l in enumerate(S[1:]):
                a = np.array(shifts[combo[j]])
                g = g + Gs[l][idx(a)]
            hs.append(g.min())
        hs = np.array(hs)
        hp = net.hprof()
        maxerr = max(maxerr, abs(hs.min() - hp) / (abs(hp) + 1e-12)); ncase += 1
        # cost invariance: translated atoms
        l = S[-1]; C, xi = feats[l]; a = np.array(shifts[-1])
        xi2 = xi * np.exp(1j * TAU * 2 * l * a[None, :] / p)
        f2 = field(p, X, l, C, xi2); f1 = net.Phi[l][idx(a)]
        costerr = max(costerr, np.abs(f2 - f1).max())
        if trial < 2:
            say(f'p={p} n={n} S={S}: h(own)={net.h():+.4f}  h_prof={hp:+.4f}  min_a h={hs.min():+.4f}  max_a h={hs.max():+.4f}')
say(f'cases {ncase}: max |min_a h - h_prof|/|h_prof| = {maxerr:.2e};  translated-atom field error {costerr:.2e}')
open('v2_alignment.out', 'w').write('\n'.join(out) + '\n')
