# Unbalanced single-set families (e.g. n=2: sets {1},{2},{1,2}; n=3: {1},{2},{1,2},{3},{1,2,3}...) with aligned atoms.
# Feature l: IF block (alpha_l) + aligned single-set block on T_l (n atoms, theta=1) of size beta_l.
#   Phi_l(x) = alpha_l + beta_l + beta_l e^{i phi_l} rho^{2 l x_{T_l}}   (exact for the J4 construction with T_l in place of [n],
#   realised by n atoms with xi = e^{2 pi i k/|T|} on T and the |T|-th roots ... we use the field model directly).
# Checks: (i) the margin bound h <= sum_l Re A_l kappa_l(d) - sigma_d at every d with E d_d^3 <= 0 and E d_{-d}^3 <= 0;
#         (ii) closure: if sum_l kappa_l(d*) min(c_n, C theta_l)^2 <= 1 then h <= sum_l at_l kappa_l(d*)  (at = IF-equiv amplitude,
#              lower bound L F_n(theta) from J2), maximised adversarially over amplitudes/phases.
import numpy as np, itertools, sys
from scipy.optimize import minimize
def Fn(th, n, s):
    r = 1/s; nu = n+1; N = n-1
    Ph = 0.5*((1+th)**r*(nu+N*th)**(1-r) + (1-th)**r*(nu-N*th)**(1-r))
    return nu**(1-s)*Ph**s
def setup(p, n, fam):
    X = np.indices((p,)*n).reshape(n, -1).T; F = (p-1)//2
    feats = [l for l, T in fam]; Ts = [T for l, T in fam]
    Z = np.array([np.exp(2j*np.pi*2*l*X[:, list(T)].sum(1)/p) for l, T in fam])      # (k, N)
    lv = np.array(feats); d = np.arange(1, p)
    KAP = 1 - np.cos(2*np.pi*np.outer(d, lv)/p); CD = 1 - np.exp(-2j*np.pi*np.outer(d, lv)/p)
    return X, Z, KAP, CD
def margin_parts(al, be, ph, Z, KAP, CD):
    f = (be*np.exp(1j*ph))[:, None]*Z                     # (k, N) fluctuations
    A = al + be
    D = np.real(CD @ f)                                    # (p-1, N): d_delta(x)
    mean = np.real(CD @ A)                                 # (p-1,)
    g = mean[:, None] + D
    return g.min(), mean, D
if __name__ == '__main__':
    rng = np.random.default_rng(1)
    viol_i = -1e9; n_i = 0; worst_ratio = -1e9; n_ii = 0
    for trial in range(400):
        p = int(rng.choice([7, 11, 13])); n = int(rng.choice([2, 3])); F = (p-1)//2; s = rng.uniform(1, (n+1)/2)
        sets = [T for k in range(1, n+1) for T in itertools.combinations(range(n), k)]
        k = int(rng.integers(2, min(F, 5)+1)); ls = rng.choice(np.arange(1, F+1), size=k, replace=False)
        fam = [(int(l), sets[rng.integers(len(sets))]) for l in ls]
        X, Z, KAP, CD = setup(p, n, fam)
        al = rng.exponential(size=k); be = rng.exponential(size=k)*rng.choice([0.05, 0.3, 1.0]); ph = rng.uniform(0, 2*np.pi, k)
        h, mean, D = margin_parts(al, be, ph, Z, KAP, CD)
        E = be**2; ReA = al + be
        for di in range(p-1):
            dm = p-2-di
            S1 = np.mean(D[di]**3); S2 = np.mean(D[dm]**3)
            if S1 <= 1e-12*np.mean(D[di]**2)**1.5 and S2 <= 1e-12*np.mean(D[dm]**2)**1.5:
                sig = np.sqrt(KAP[di] @ E)
                viol_i = max(viol_i, (h - (KAP[di] @ ReA - sig))/(KAP[di] @ ReA)); n_i += 1
        # (ii) closure test with IF-equivalent amplitudes at = (al+be) F_n(theta), theta = be/(al+be)
        r = 1/s; nu = n+1; C = 2*(s-1)/nu; cn = 1 - (n/nu)**(s-1)
        th = be/(al+be); at = (al+be)*Fn(th, n, s)
        ds = int(np.argmin(KAP @ at))
        cond = np.sum(KAP[ds]*np.minimum(cn, C*th)**2)
        skew_ok = np.mean(D[ds]**3) <= 1e-12 and np.mean(D[p-2-ds]**3) <= 1e-12
        if cond <= 1 and skew_ok:
            n_ii += 1; worst_ratio = max(worst_ratio, h/(KAP[ds] @ at))
    print(f"(i) {n_i} (network, delta) pairs with nonpositive skew at +-delta: max (h - [sum Re A kappa - sigma])/scale = {viol_i:.3e} (<= 0)")
    print(f"(ii) {n_ii} networks meeting the closure hypothesis: max h / sum_l at_l kappa_l(d*) = {worst_ratio:.4f} (<= 1)")
    # adversarial: maximise h / sum at kappa(d*) over amplitudes/phases for the unbalanced family {1},{2},{1,2} at n=2
    def adv(p, s, seeds=6):
        n = 2; fam_sets = [(0,), (1,), (0, 1)]; F = (p-1)//2; best = -1
        nu = 3; C = 2*(s-1)/nu; cn = 1 - (n/nu)**(s-1)
        for sd in range(seeds):
            rr = np.random.default_rng(100+sd); ls = rr.choice(np.arange(1, F+1), size=3, replace=False)
            fam = [(int(l), T) for l, T in zip(ls, fam_sets)]; X, Z, KAP, CD = setup(p, n, fam)
            def obj(x):
                al = np.exp(x[:3]); be = np.exp(x[3:6]); ph = x[6:]
                th = be/(al+be); at = (al+be)*Fn(th, n, s); ds = np.argmin(KAP @ at)
                h, _, _ = margin_parts(al, be, ph, Z, KAP, CD)
                return -h/(KAP[ds] @ at)
            x0 = np.concatenate([rr.normal(0, 1, 3), rr.normal(-1, 1, 3), rr.uniform(0, 2*np.pi, 3)])
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
            best = max(best, -res.fun)
        return best
    for p in [7, 11, 13]:
        for s in [1.25, 1.5]:
            print(f"   adversarial unbalanced {{1}},{{2}},{{1,2}} family, p={p}, s={s}: max h / sum at kappa(d*) = {adv(p, s):.5f}")
