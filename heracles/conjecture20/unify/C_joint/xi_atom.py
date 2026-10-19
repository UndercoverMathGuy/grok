# Field-row constant of a single atom (one neuron) and of the n-atom single-T block (torus model, Monte Carlo):
#   Xi = nu^{1-r} E[(Re prod_t (1 + e^{iU_t}))_+^r] / 1
import numpy as np
from scipy.special import gamma as G
rng = np.random.default_rng(0)
for n in [2, 3, 4, 5, 7, 10]:
    nu = n+1; U = rng.uniform(0, 2*np.pi, size=(400000, n))
    P = np.prod(1+np.exp(1j*U), axis=1)
    out = []
    for s in [1.25, 1.5, nu/2]:
        if s > nu/2: continue
        r = 1/s
        X1 = nu**(1-r)*np.mean(np.maximum(P.real, 0)**r)
        c0 = 2**r*G(r+0.5)/(np.sqrt(np.pi)*G(r+1)); XT = (nu/n)**(1-r)*c0
        out.append(f"s={s:.2f}: single atom {X1:.3f}, n-atom single-T {XT:.4f}")
    print(f"n={n}: " + " | ".join(out))
