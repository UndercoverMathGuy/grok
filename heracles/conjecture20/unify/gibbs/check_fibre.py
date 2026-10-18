# Check of the fibre (conditioning) certificate, class C, general n:
#   A  = -min_x Z_d(x),   A' = -min_x Z_{-d}(x)
#   for every admissible split (U,S) of the slot-sets with translation a (2 s_T(a) = -d on U, U disjoint from span S):
#       A A' >= sum_{T in U} V_T + sum_{T in S} C_T(e_T),   e_T = 2 s_T(a) + d
#   V_T = sum_l |a_{l,T}|^2 kappa_l(d),  C_T(e) = sum_l |a_{l,T}|^2 kappa_l(d) cos(2 pi l e/p).
# Star splits U = {T containing t} (a = -d/2 e_t) give  A A' >= E - sum_{T not containing t} E2_T,  E2_T = sum |a|^2 kappa^2.
# Also per-T Bhatia-Davis: A A' >= V_T.  We test random fields with random moduli/phases (exact minima over Z_p^n).
import numpy as np, itertools, sys
rng = np.random.default_rng(7)
def run(p, n, trials):
    F = np.arange(1, (p-1)//2 + 1); K = len(F)
    Ts = [T for T in itertools.product([0,1], repeat=n) if any(T)]
    X = np.indices((p,)*n).reshape(n, -1)
    worst = np.inf; worst_pt = np.inf
    inv2 = pow(2, -1, p)
    for tr in range(trials):
        d = int(rng.integers(1, p))
        a = (rng.exponential(size=(len(Ts), K)) * (rng.random((len(Ts), K)) < 0.6)) * np.exp(2j*np.pi*rng.random((len(Ts), K)))
        c = 1 - np.exp(-2j*np.pi*F*d/p); kap = 1 - np.cos(2*np.pi*F*d/p)
        def Z(sign):
            cc = c if sign > 0 else np.conj(c)
            out = np.zeros(X.shape[1])
            for i, T in enumerate(Ts):
                sT = (np.array(T) @ X) % p
                out += np.real((a[i]*cc)[None, :] * np.exp(2j*np.pi*np.outer(sT, 2*F)/p)).sum(1)
            return out
        A = -Z(+1).min(); Ap = -Z(-1).min()
        V = np.array([(np.abs(a[i])**2*kap).sum() for i in range(len(Ts))])
        E2 = np.array([(np.abs(a[i])**2*kap**2).sum() for i in range(len(Ts))])
        E = V.sum()
        cert = [V.max()]
        for t in range(n):   # star split
            cert.append(sum(V[i] for i, T in enumerate(Ts) if T[t]) + sum(V[i] - E2[i] for i, T in enumerate(Ts) if not T[t]))
        if n == 2:           # the third alignment: U = {1},{2}, S = {12}
            cert.append(V[Ts.index((1,0))] + V[Ts.index((0,1))] + V[Ts.index((1,1))] - E2[Ts.index((1,1))])
        best = max(cert)
        worst = min(worst, A*Ap - best)          # must be >= 0
        worst_pt = min(worst_pt, A*Ap/E)
    print(f"p={p} n={n}: min over {trials} trials of (AA' - certificate) = {worst:.3e}  (>=0 required);  min AA'/E = {worst_pt:.3f}")
for (p, n, tr) in [(5,2,3000),(7,2,3000),(11,2,2000),(13,2,1500),(5,3,1500),(7,3,800),(5,4,300)]:
    run(p, n, tr)
