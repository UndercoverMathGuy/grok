# Sanity checks of lib.py against known identities.
import numpy as np, itertools
from lib import *
rng = np.random.default_rng(0)
for (p, n, m) in [(5, 2, 4), (7, 2, 3), (5, 3, 3), (3, 2, 4)]:
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
    f = table(U, W); g = gaps(f); Phi = fields(f); co = coeffs(f)
    Fn = (p-1)//2
    # (1) E Phi_l = conj(beta_l)
    e1 = max(abs(Phi[l-1].mean() - np.conj(co[(l, (0,)*n)])) for l in range(1, Fn+1))
    # (2) g(x,delta) = sum_l Re(Phi_l(x)(1 - rho^{-l delta}))
    rec = np.zeros_like(g)
    for d in range(p):
        for l in range(1, Fn+1):
            rec[..., d] += np.real(Phi[l-1]*(1 - rho(p)**(-l*d)))
    e2 = np.abs(rec - g).max()
    # (3) sign rule: beta_{l,T} = sum_i beta_il prod_{t in T} xi_itl
    beta, xi = neuron_atoms(U, W)
    e3 = 0
    for l in range(1, Fn+1):
        for T in itertools.product([0,1], repeat=n):
            val = np.sum(beta[:, l-1]*np.prod(np.where(np.array(T)[None,:]==1, xi[:,:,l-1], 1), axis=1))
            e3 = max(e3, abs(val - co[(l, T)]))
    # (4) slack identity Phi_l(x) = (2/p)(h - sum_{delta!=0} sigma(x,delta) rho^{l delta})
    h = g[..., 1:].min(); sig = g - h; sig[..., 0] = 0
    e4 = 0
    for l in range(1, Fn+1):
        rhs = (2/p)*(h - np.sum(sig[..., 1:]*rho(p)**(l*np.arange(1, p)), axis=-1))
        e4 = max(e4, np.abs(rhs - Phi[l-1]).max())
    # (5) Lemma 2: h <= sum_l Re beta_l kappa_l(delta) for all delta
    lem2 = min(sum(np.real(co[(l,(0,)*n)])*kappa(l, d, p) for l in range(1, Fn+1)) for d in range(1, p)) - h
    print(f"p={p} n={n}: EPhi {e1:.1e}  g-recon {e2:.1e}  signrule {e3:.1e}  slack-id {e4:.1e}  Lemma2 slack {lem2:.3f}>=0")
# IF block: margin equals nu^{1-s} B_p Gamma for single feature (y=e_j): h = A kappa_min, check cost formula
for (p, n, b) in [(5, 2, 2.0), (7, 3, 2.0), (11, 2, 2.5)]:
    nu = n+1; s = nu/b
    U, W = if_block(p, n, 1, 1.0)
    f = table(U, W); h = margin(f); P = cost(U, W, b)
    gam = h/P**s
    pred = nu**(1-s)*Bp(p, nu)*kappa(1, 1, p)
    co = coeffs(f)
    interf = max(abs(co[(1,T)]) for T in itertools.product([0,1], repeat=n) if any(T))
    print(f"IF block p={p} n={n} b={b}: gamma/pred = {gam/pred:.6f}, max sign-flipped {interf:.1e}")
