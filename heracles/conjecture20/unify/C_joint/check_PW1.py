# Lemma (one cheap single-T feature): min_x Psi(x) <= 0, Psi = sum_l [(Re Phi_l)_+^r - nu^{r-1} Q_l], explicit networks of
# Prop. emb (IF blocks + ONE feature with a single-T block), every p tested including p=3, many n, s.
import numpy as np
rng = np.random.default_rng(41); worst = -1e9; N = 0
for trial in range(3000):
    p = int(rng.choice([3, 5, 7, 11, 13, 31, 101])); n = int(rng.integers(2, 16)); nu = n+1
    s = rng.uniform(1, nu/2); r = 1/s; F = (p-1)//2
    al = rng.exponential(1, F)*(rng.random(F) < 0.7); j = rng.integers(F)
    be = rng.exponential(1); ph = rng.uniform(-np.pi, np.pi)
    w = np.arange(p); l = j+1
    L = al[j] + be
    psi_j = np.maximum(L + be*np.cos(ph + 2*np.pi*l*w/p), 0)**r
    Qj = nu**(1-r)*al[j]**r + n**(1-r)*be**r
    others = sum(al[k]**r - nu**(r-1)*(nu**(1-r)*al[k]**r) for k in range(F) if k != j)   # IF blocks: exactly 0
    Psi = psi_j - nu**(r-1)*Qj + others
    worst = max(worst, Psi.min()/(nu**(r-1)*Qj)); N += 1
print(f"{N} networks: max_x-min of Psi / (nu^(r-1) Q_cheap) = {worst:.3e} (<= 0 expected)")
