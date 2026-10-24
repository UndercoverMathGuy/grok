# Feasibility tests for a proof of the sharp L2 payment  |A| - 3^{1-s} Q^s <= ((1-3^{1-s})/sqrt3) R  (n = 2).
# Reduction (twisted matrix of Note Thm P12):  Hhat = correlation matrix, entries x = B/L, y = C/L, z = D'/L,
#   cost: 3^{1-s} Q^s >= L * M_{1/s}(nu),  M_r(nu) = (sum nu_k^r / 3)^{1/r} (power mean), nu = spec(Hhat);
#   gain: |A| - c lam3 |D| <= L sqrt(1 + c^2 lam3^2 - 2 c lam3 delta),  delta = |z|;   R >= lam . (|B|,|C|,|D|).
# So it suffices: for all correlation matrices, some unit lam >= 0 with
#   G(lam) := sqrt(1 + c^2 lam3^2 - 2 c lam3 delta) - c (lam1 b + lam2 c') <= M_{1/s}(nu).
# (T1) concavity in x = 3^{1-s} of x -> M_{1/s(x)}(nu)  (would reduce every s in [1,3/2] to s = 3/2);
# (T2) the reduced inequality at s = 3/2 with the best lam (grid), exact power mean and Jensen form 3^{-lambda/2};
#      report the minimal margin and where it occurs.
import numpy as np, sys
rng = np.random.default_rng(1)
def spec(b, c, d, q):
    P = b*b + c*c + d*d
    t = np.roots([1, 0, -P, -q]).real
    return 1 + np.sort(t)[::-1]
def M(nu, r):
    nu = np.clip(nu, 0, None); return (np.mean(nu**r))**(1/r)
def jensen(nu):
    nu = np.clip(nu, 1e-300, None); lam = np.sum(nu/3*np.log(nu))/np.log(3); return 3**(-lam/2), lam
if __name__ == '__main__':
    # (T1)
    worst = np.inf
    xs = np.linspace(1/np.sqrt(3), 1, 41)
    for t in range(3000):
        nu = rng.dirichlet(np.ones(3)*rng.choice([0.2, 1, 5]))*3
        vals = np.array([M(nu, 1/(1 - np.log(x)/np.log(3))) for x in xs])
        sec = vals[:-2] - 2*vals[1:-1] + vals[2:]
        worst = min(worst, -sec.max())
    print(f"(T1) min over spectra of -(max second difference) of x -> M_(1/s(x))(nu) = {worst:.3e}  (>= 0 means concave on the grid)")
    # (T2)
    c0 = (np.sqrt(3) - 1)/3
    lams = []
    for a in np.linspace(0, np.pi/2, 31):
        for e in np.linspace(0, np.pi/2, 31):
            lams.append((np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)))
    lams = np.array(lams)
    worst_exact = (np.inf, None); worst_jen = (np.inf, None)
    for t in range(int(sys.argv[1]) if len(sys.argv) > 1 else 20000):
        b, c, d = rng.random(3)**rng.choice([1, 0.2, 3])
        P = b*b + c*c + d*d
        qlo = max(P - 1, -2*b*c*d); qhi = 2*b*c*d
        if qlo > qhi: continue
        q = qlo if rng.random() < 0.7 else rng.uniform(qlo, qhi)
        nu = spec(b, c, d, q)
        if nu.min() < -1e-9: continue
        G = np.sqrt(1 + c0**2*lams[:, 2]**2 - 2*c0*lams[:, 2]*d) - c0*(lams[:, 0]*b + lams[:, 1]*c)
        g = G.min()
        me = M(nu, 2/3) - g; mj = jensen(nu)[0] - g
        dist = np.sqrt((1-b)**2 + (1-c)**2 + (1-d)**2)
        if me < worst_exact[0]: worst_exact = (me, (b, c, d, q, dist))
        if mj < worst_jen[0]: worst_jen = (mj, (b, c, d, q, dist))
    print(f"(T2) s=3/2 exact power mean: min margin {worst_exact[0]:.3e} at (b,c',delta,q,dist-to-corner) = {np.round(worst_exact[1], 4)}")
    print(f"(T2) s=3/2 Jensen form     : min margin {worst_jen[0]:.3e} at {np.round(worst_jen[1], 4)}")
