# C_ccfactor c11: background-immune additive linear detection.  For class C, features l_1..l_k (distinct) with pairwise
# DISJOINT slot sets T_1..T_k:  D(delta) >= sum_i |beta_{l_i,T_i}| |sin(pi l_i delta/p)|  for every delta, whatever
# the other features are.  Also: the field coefficient of Phi_l at character 2l 1_T has modulus |beta_{l,T}|.
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
import ccf, cclib as C
E = C.E
rng = np.random.default_rng(31)
werr = 0; worst = -np.inf; cnt = 0; tight = []
for (p, n) in [(7, 3), (5, 3), (11, 2), (7, 2), (13, 2), (5, 4)]:
    F = (p - 1) // 2
    for t in range(12):
        y = rng.uniform(0.2, 1, F)
        net, lab = ccf.perturbed_if(p, n, y, rng, [0.2, 0.5, 1.0][t % 3], extra=int(rng.integers(1, 3)))
        fn = ccf.FeatNet(net, 1.5, lab)
        fl = net.fields()                                  # (F, p,...,p) Fourier coefficients of Phi_l
        for l in range(1, F + 1):
            for T in itertools.product([0, 1], repeat=n):
                if not any(T): continue
                idx = tuple((2 * l * Tt) % p for Tt in T)
                werr = max(werr, abs(abs(fl[l - 1][idx]) - abs(fn.sf[(l, T)])) / max(abs(fn.sf[(l, T)]), 1e-300))
        # random family: up to min(n, F) distinct features, pairwise disjoint slot sets (random partition pieces)
        k = int(rng.integers(1, min(n, F) + 1)); feats = rng.choice(np.arange(1, F + 1), size=k, replace=False)
        perm = rng.permutation(n); cuts = sorted(rng.choice(np.arange(1, n), size=k - 1, replace=False)) if k > 1 else []
        pieces = np.split(perm, cuts)
        D = fn.joint_depth()
        for d in range(p - 1):
            det = 0
            for l, piece in zip(feats, pieces):
                T = tuple(1 if t in piece else 0 for t in range(n))
                det += abs(fn.sf[(int(l), T)]) * abs(np.sin(np.pi * l * (d + 1) / p))
            worst = max(worst, det - D[d]); cnt += 1; tight.append(det / D[d] if D[d] > 0 else 0)
print(f'field coefficient modulus vs |beta_lT|: max rel err {werr:.2e}')
print(f'{cnt} (network, family, delta) cases: max (detection - D) = {worst:.3e} (<= 0 required); max detection/D = {max(tight):.4f}')
