# Direction (3): the delta-phases as a translation.
# For class-C fluctuations f_l(x) = sum_{T in calT} B_{l,T} rho^{2 l x_T} and c_l(d) = 1 - rho^{-l d}:
#   d_d(x) = sum_l Re(f_l(x) c_l(d)).
# Lemma: if u in Z_p^n has <1_T,u> = 1 (mod p) for every T in calT ("balanced"), then d_{-d}(x) = -d_d(x + (d/2) u),
# hence max(depth_d, depth_{-d}) >= (1/2) osc d_d.  If calT has an unbalanced relation (sum lam_T 1_T = 0, sum lam_T != 0)
# no such u exists, and the oscillation bound can fail.
import numpy as np, itertools
rng = np.random.default_rng(5)
def fluct(p, n, coeffs):
    F = (p-1)//2; X = np.indices((p,)*n).reshape(n, -1).T
    out = {}
    for d in range(1, p):
        tot = np.zeros(len(X))
        for (l, T), B in coeffs.items():
            xT = X[:, list(T)].sum(1)
            tot += np.real(B*np.exp(2j*np.pi*2*l*xT/p)*(1-np.exp(-2j*np.pi*l*d/p)))
        out[d] = tot.reshape((p,)*n)
    return out
def shift(arr, sh):
    # arr[x + sh]
    return np.roll(arr, shift=[-s for s in sh], axis=tuple(range(arr.ndim)))
inv2 = lambda p: pow(2, -1, p)
print("balanced families: max |d_{-d}(x) + d_d(x + (d/2)u)|")
for (p, n, fam, u) in [(7, 2, [(0,), (1,)], (1, 1)), (7, 2, [(0, 1)], (1, 0)), (11, 3, [(0, 1), (1, 2), (0, 2)], None),
                       (7, 3, [(0,), (1,), (2,)], (1, 1, 1)), (11, 2, [(0,), (0, 1)], (1, 0))]:
    if u is None:   # level 2 at n=3: u = (1/2,1/2,1/2)
        h = inv2(p); u = (h, h, h)
    F = (p-1)//2
    coeffs = {(l, T): rng.normal()+1j*rng.normal() for l in range(1, F+1) for T in fam}
    D = fluct(p, n, coeffs); err = 0; viol = 0
    for d in range(1, p):
        sh = tuple((d*inv2(p)*ui) % p for ui in u)
        err = max(err, np.max(np.abs(D[p-d] + shift(D[d], sh))))
        dep_p = -D[d].min(); dep_m = -D[p-d].min(); osc = D[d].max() - D[d].min()
        viol = max(viol, osc/2 - max(dep_p, dep_m))
    print(f"  p={p} n={n} T-family={fam} u={u}: identity error {err:.1e}; max[(1/2)osc - max(depth_d, depth_-d)] = {viol:.2e} (<=0 expected)")
print("unbalanced family {1},{2},{1,2} at n=2 (relation 1_1 + 1_2 - 1_12 = 0, coefficient sum 1):")
worst = -1e9
for trial in range(200):
    p = [5, 7, 11, 13][trial % 4]; F = (p-1)//2
    coeffs = {(l, T): (rng.normal()+1j*rng.normal())*rng.exponential() for l in range(1, F+1) for T in [(0,), (1,), (0, 1)]}
    D = fluct(p, 2, coeffs)
    for d in range(1, p):
        dep_p = -D[d].min(); dep_m = -D[p-d].min(); osc = D[d].max() - D[d].min()
        rel = (osc/2 - max(dep_p, dep_m))/(osc/2)
        if rel > worst: worst = rel; arg = (p, d)
print(f"  max over 200 random networks of [(1/2)osc d_d - max(depth_d, depth_-d)]/((1/2)osc) = {worst:.4f} at (p,d)={arg} (>0 means the oscillation bound fails)")
