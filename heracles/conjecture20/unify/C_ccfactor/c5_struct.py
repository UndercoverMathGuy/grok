# C_ccfactor c5: (a) adversarial search against the Shapley and isolated-proportional charge rules (CD kept);
# (b) single-pair copositive obstruction (2x2 algebra); (c) antipodal support features at binding offsets of
# programme maximisers; (d) compensation survival rho = D / sum_l D_l^iso versus DU tightness at binding offsets.
import sys, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
import ccf, cclib as C
import prog
E = C.E
rng = np.random.default_rng(5)


def build(theta, p, n, S, y, extra):
    """IF blocks on S with allocation y (scaled by exp(theta[:|S|])) plus `extra` free class-C neurons per feature."""
    a = np.arange(p); nu = n + 1; Us = []; Ws = []; lab = []; k = len(S); off = k
    for j, l in enumerate(S):
        lam = (2 ** n * y[j] * np.exp(3 * np.tanh(theta[j] / 3)) / nu) ** (1.0 / nu)
        U, W = E.if_block(p, n, l, lam); Us.append(U); Ws.append(W); lab += [l] * nu
        cs = np.cos(2 * np.pi * l * a / p); sn = np.sin(2 * np.pi * l * a / p)
        for e in range(extra):
            c = 3 * np.tanh(theta[off: off + 2 * (n + 1)] / 3).reshape(n + 1, 2); off += 2 * (n + 1)
            Us.append(np.array([[c[t, 0] * cs + c[t, 1] * sn for t in range(n)]]))
            Ws.append(np.array([c[n, 0] * cs + c[n, 1] * sn])); lab.append(l)
    return E.Net(np.concatenate(Us), np.concatenate(Ws)), np.array(lab)


def rule_ratios(fn, pi):
    Bl = pi * fn.B * fn.Pl ** fn.s; bind = fn.binding(); feats = [l - 1 for l in fn.feat]
    lim = Bl[feats, None] * fn.kap[feats][:, bind]
    sh = (fn.charge_shapley()[feats][:, bind] / lim).max()
    tm = (fn.charge_tm()[feats][:, bind] / lim).max()
    D = fn.joint_depth(); Di = fn.iso_depth()
    Dprop = D[None, :] * Di / np.maximum(Di.sum(0, keepdims=True), 1e-300)
    pr = ((fn.Rbar_l - Dprop)[feats][:, bind] / lim).max()
    return sh, tm, pr


print('=== (a) adversarial search: Shapley / isolated-proportional charge rules at P^s-binding offsets (n=2, pi=3^{1-s})')
for (p, n, b, S, extra, R) in [(7, 2, 2.0, [1, 2, 3], 1, 12), (7, 2, 2.0, [1, 2], 2, 10), (11, 2, 2.0, [1, 2, 4], 1, 8),
                               (5, 2, 2.0, [1, 2], 2, 10)]:
    nu = n + 1; s = nu / b; pi = 3 ** (1 - s)
    best = {'sh': (-1, None), 'pr': (-1, None)}
    for rep in range(R):
        y = rng.uniform(0.3, 1.0, len(S)); dim = len(S) + len(S) * extra * 2 * (n + 1)
        for key in ['sh', 'pr']:
            def obj(th):
                try:
                    net, lab = build(th, p, n, S, y, extra)
                    fn = ccf.FeatNet(net, s, lab)
                    if not np.isfinite(fn.h): return 1e3
                    if fn.h <= 0: return 10.0 - fn.h / abs(fn.Rbar_l.sum(0).min())
                    sh, tm, pr = rule_ratios(fn, pi)
                    v = -(sh if key == 'sh' else pr)
                    return v if np.isfinite(v) else 1e3
                except Exception:
                    return 1e3
            th0 = np.concatenate([np.zeros(len(S)), 0.4 * rng.normal(size=dim - len(S))])
            res = minimize(obj, th0, method='Powell', options=dict(maxfev=2500, xtol=1e-4, ftol=1e-6))
            net, lab = build(res.x, p, n, S, y, extra); fn = ccf.FeatNet(net, s, lab)
            if fn.h > 0:
                sh, tm, pr = rule_ratios(fn, pi); cd = fn.cd_ratio(pi)
                val = sh if key == 'sh' else pr
                if val > best[key][0]:
                    best[key] = (val, (round(sh, 4), round(tm, 4), round(pr, 4), round(cd, 4)))
                    if val > 1 and cd <= 1:
                        np.savez(f'rule_cex_{key}_p{p}_n{n}.npz', U=net.U, W=net.W, lab=lab)
    print(f'  (p,n,b)=({p},{n},{b}) S={S} extra={extra}: max Shapley charge/price = {best["sh"][0]:.4f} '
          f'[sh,tm,prop,CD]={best["sh"][1]};  max isolated-proportional charge/price = {best["pr"][0]:.4f} [sh,tm,prop,CD]={best["pr"][1]}')

print('=== (b) single offset pair: no copositive 2x2 form gives a positive weight to an antipodal feature')
cnt = 0; viol = 0; acute_ok = 0
for t in range(200000):
    al, ga = rng.exponential(size=2) * (rng.random(2) < 0.7)
    be = rng.uniform(-np.sqrt(al * ga), 3)                # copositive: al, ga >= 0, be >= -sqrt(al ga)
    kap = rng.uniform(0, 2)
    w = -kap * ((al + ga) + 2 * be * (kap - 1))           # weight of a feature with kappa_l(delta) = kap
    if kap >= 1:
        cnt += 1; viol += w > 1e-12
    elif w > 0: acute_ok += 1
print(f'  {cnt} random (copositive Q, antipodal kappa >= 1): positive weights found = {viol}; acute features with positive weight: {acute_ok}')

print('=== (c) programme maximisers: does every binding offset carry an antipodal support feature (cos(2 pi l d/p) <= 0)?')
from sympy import primerange
rows = []
for p in list(primerange(5, 62)):
    for s in [1.05, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        G, y, cov = prog.solve(p, s, restarts=4)
        S = np.nonzero(y > 1e-10)[0] + 1; F = (p - 1) // 2
        Bd = np.nonzero(cov <= G * (1 + 1e-6))[0] + 1                # binding offsets in 1..F
        allac = [d for d in Bd if all(np.cos(2 * np.pi * l * d / p) > 0 for l in S)]
        rows.append((p, s, len(S), len(Bd), len(allac)))
bad = [r for r in rows if r[2] >= 2 and r[4] > 0]
print(f'  {len(rows)} (p,s) cases, p in [5,61]; cases with K*>=2 and an all-acute binding offset: {len(bad)} {bad[:10]}')
print(f'  K*=1 cases: {[r[:2] for r in rows if r[2] == 1][:12]}')

print('=== (d) compensation survival rho(d) = D(d)/sum_l D_l^iso(d) vs DU tightness tau(d) = sum_l E_l kappa_l / sum_l D_l^iso')
import glob
for f in sorted(glob.glob('best_cd_*.npz')):
    d = np.load(f); net = E.Net(d['U'], d['W']); p, n = net.p, net.n
    b = float(f.split('_b')[1].split('_')[0]); s = (n + 1) / b; pi = (n + 1) ** (1 - s)
    fn = ccf.FeatNet(net, s, d['lab'])
    D = fn.joint_depth(); Di = fn.iso_depth(); bind = fn.binding(); Ex = fn.excess(pi)
    tau = (Ex[:, None] * fn.kap).sum(0); den = Di.sum(0)
    ifn = max(abs(fn.sf[(l, T)]) / abs(fn.beta[l - 1]) for l in fn.feat for T in itertools.product([0, 1], repeat=n) if any(T))
    print(f'  {f}: CD={fn.cd_ratio(pi):.5f} max_l,T |beta_lT|/|beta_l|={ifn:.1e}  at binding d: rho={np.round(D[bind] / np.maximum(den[bind], 1e-300), 3)}'
          f' tau={np.round(tau[bind] / np.maximum(den[bind], 1e-300), 3)}')
