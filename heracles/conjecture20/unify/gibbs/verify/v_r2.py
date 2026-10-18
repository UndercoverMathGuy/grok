# Referee check of R2 (reconstructed PBD): mM + m eps >= V - sqrt(V) eps ; (m+M)/2 >= c_p sqrt(V), all plane pairs.
import numpy as np, itertools
rng = np.random.default_rng(3)
def planes(p):
    X = np.array(list(itertools.product(range(p), repeat=3))); return X, X.sum(1) % p
worst1 = np.inf; worst2 = np.inf
for p in [3, 5, 7, 11]:
    X, S = planes(p); w1 = np.inf; w2 = np.inf
    eta = np.sqrt(2/(3*p)); cp = np.sqrt(1 - eta) - eta/2
    fams = []
    for t in range(1500):
        k = t % 5
        if k == 0: f = rng.normal(size=(3, p))
        elif k == 1: f = rng.integers(-2, 3, size=(3, p)).astype(float)
        elif k == 2: f = np.array([((-np.arange(p) + rng.integers(p)) % p)*1.0 for _ in range(3)]) * rng.random((3,1))
        elif k == 3: f = np.zeros((3, p)); f[np.arange(3), rng.integers(p, size=3)] = rng.normal(size=3)*5; f += 0.05*rng.normal(size=(3,p))
        else: f = rng.exponential(size=(3, p))**3
        f = f - f.mean(1, keepdims=True)
        V = (f**2).mean(1).sum()
        if V < 1e-12: continue
        osc = f.max(1) - f.min(1)
        eps = min(np.sqrt(osc[i]*osc[j]) for i, j in [(0,1),(0,2),(1,2)])/p
        F = f[0][X[:,0]] + f[1][X[:,1]] + f[2][X[:,2]]
        m = -np.array([F[S == c].min() for c in range(p)]); M = np.array([F[S == c].max() for c in range(p)])
        mm, MM = m.min(), M.min()      # worst plane pair for both statements (monotone in m, M)
        w1 = min(w1, (mm*MM + mm*eps - (V - np.sqrt(V)*eps))/V)
        w2 = min(w2, ((mm + MM)/2 - cp*np.sqrt(V))/np.sqrt(V))
    print(f"p={p}: min slack (mM + m eps - V + sqrt(V) eps)/V = {w1:.4e};  min ((m+M)/2 - c_p sqrtV)/sqrtV = {w2:.4e}  (c_p = {cp:.4f})")
