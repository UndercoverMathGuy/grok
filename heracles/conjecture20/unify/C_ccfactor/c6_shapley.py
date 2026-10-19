# C_ccfactor c6: targeted search against the Shapley charge rule. Feature 1 = a genuinely cheap feature (width 2 < nu,
# free class-C weights); background = IF blocks on the other features (free scale) + free extra neurons that can
# place the joint argmin where feature 1 is high. Objective: Shapley charge of feature 1 / its price, at the P^s-binding
# offsets (rule needed for the reduction) and at all offsets (local per-feature form).  CD is recorded.
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
import ccf, cclib as C
E = C.E
rng = np.random.default_rng(17)


def build(th, p, n, S, extra, w1=2):
    a = np.arange(p); nu = n + 1; Us = []; Ws = []; lab = []; off = 0
    cs = lambda l: np.cos(2 * np.pi * l * a / p); sn = lambda l: np.sin(2 * np.pi * l * a / p)
    l1 = S[0]
    for e in range(w1):                                         # cheap feature: w1 free neurons
        c = th[off: off + 2 * nu].reshape(nu, 2); off += 2 * nu
        Us.append(np.array([[c[t, 0] * cs(l1) + c[t, 1] * sn(l1) for t in range(n)]]))
        Ws.append(np.array([c[n, 0] * cs(l1) + c[n, 1] * sn(l1)])); lab.append(l1)
    for l in S[1:]:
        lam = np.exp(2 * np.tanh(th[off] / 2)); off += 1
        U, W = E.if_block(p, n, l, lam); Us.append(U); Ws.append(W); lab += [l] * nu
        for e in range(extra):
            c = th[off: off + 2 * nu].reshape(nu, 2); off += 2 * nu
            Us.append(np.array([[c[t, 0] * cs(l) + c[t, 1] * sn(l) for t in range(n)]]))
            Ws.append(np.array([c[n, 0] * cs(l) + c[n, 1] * sn(l)])); lab.append(l)
    return E.Net(np.concatenate(Us), np.concatenate(Ws)), np.array(lab)


def evaluate(fn, pi, l1):
    Bl = pi * fn.B * fn.Pl ** fn.s; bind = fn.binding()
    Csh = fn.charge_shapley(); Ctm = fn.charge_tm()
    lim = Bl[l1 - 1] * fn.kap[l1 - 1]
    return dict(sh_bind=(Csh[l1 - 1, bind] / lim[bind]).max(), sh_all=(Csh[l1 - 1] / lim).max(),
                tm_bind=(Ctm[l1 - 1, bind] / lim[bind]).max(), cd=fn.cd_ratio(pi),
                cheap=np.real(fn.beta[l1 - 1]) / Bl[l1 - 1])


for (p, n, b, S, extra, R) in [(7, 2, 2.0, [1, 2], 1, 10), (7, 2, 2.0, [1, 2, 3], 1, 8), (5, 2, 2.0, [1, 2], 2, 10),
                               (11, 2, 2.0, [1, 2, 3], 1, 6)]:
    nu = n + 1; s = nu / b; pi = 3 ** (1 - s)
    dim = 2 * 2 * nu + (len(S) - 1) * (1 + extra * 2 * nu)
    for key in ['sh_bind', 'sh_all']:
        best = (-1, None)
        for rep in range(R):
            def obj(th):
                try:
                    net, lab = build(th, p, n, S, extra)
                    fn = ccf.FeatNet(net, s, lab)
                    if not np.isfinite(fn.h): return 1e3
                    if fn.h <= 0: return 10.0 - fn.h / abs(fn.Rbar_l.sum(0).min())
                    v = evaluate(fn, pi, S[0])[key]
                    return -v if np.isfinite(v) else 1e3
                except Exception:
                    return 1e3
            th0 = rng.normal(size=dim) * 0.6
            th0[2 * 2 * nu::1 + extra * 2 * nu] = 0.7          # background block scales
            res = minimize(obj, th0, method='Powell', options=dict(maxfev=4000, xtol=1e-5, ftol=1e-7))
            net, lab = build(res.x, p, n, S, extra); fn = ccf.FeatNet(net, s, lab)
            if fn.h > 0:
                ev = evaluate(fn, pi, S[0])
                if ev[key] > best[0]:
                    best = (ev[key], ev)
                    np.savez(f'shapley_best_{key}_p{p}_S{"-".join(map(str, S))}.npz', U=net.U, W=net.W, lab=lab)
        ev = best[1]
        print(f'(p,n,b)=({p},{n},{b}) S={S}: objective {key}: max = {best[0]:.4f};  '
              + ', '.join(f'{k}={v:.4f}' for k, v in ev.items()), flush=True)
