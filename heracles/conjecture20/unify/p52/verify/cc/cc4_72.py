"""Referee check of R14 (the (7,2) two-feature certificate, simple72_rig.py) -- independent implementation.

Bound (R12 + R13), features {1,2}, A_j = |beta_j|, M_j = max_T |beta_{j,T}|, x_j = M_j/A_j, A_1 + A_2 = 1, tau = A_2:
   ratio <= max(0, N*) / (3.5 * 3^{-s} * (sum_j (F_j A_j)^{1/s})^s),
   F_j = max(1 - c_1 x_j, 3^{1-s} max(1, x_j)),  c_1 = 1 - 3^{1-s},
   N* = min over symmetric probabilities mu of [sum_j K_j(mu) A_j - 3.5 min mu * max_j M_j].
The objective in mu is convex piecewise linear on the simplex {mu_1+mu_2+mu_3 = 1/2}; its minimum is attained at a
vertex of a linearity region, i.e. at mu = e_d/2 or at the uniform mu = 1/6 (proved in the referee notes), so
   N* = min( min_d [kappa_1(d) A_1 + kappa_2(d) A_2],  (7/6)(A_1 + A_2) - (7/12) M ).
Part A: sup over (tau, x_1, x_2) of the closed form, in mpmath (60 digits), by dense grid + Nelder-Mead polishing.
Part B: their rigorous box_ratio on tiny boxes around random points must dominate my point value (same mu).
Part C: explicit (7,2) networks with exactly the features {1,2} (random, DC, IF-like, locally optimised):
        direct ratio <= closed-form bound(own data of the network) ; R12 and R13 inequalities hold.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import mpmath as mp
from scipy.optimize import minimize
from cc_net import Net, B_p, gammaIF72, cancel_feature, pure_neuron

mp.mp.dps = 60
rng = np.random.default_rng(99)
p = 7


def kap(j, d):
    return 1 - mp.cos(2 * mp.pi * j * d / 7)


KAP = {(j, d): kap(j, d) for j in (1, 2, 3) for d in (1, 2, 3)}


def bound_mp(tau, x1, x2, s, inf1=False, inf2=False):
    tau, x1, x2, s = mp.mpf(tau), mp.mpf(x1), mp.mpf(x2), mp.mpf(s)
    A1, A2 = 1 - tau, tau
    M = max(mp.mpf(0) if inf1 else A1 * x1, mp.mpf(0) if inf2 else A2 * x2)
    Nst = min(min(KAP[(1, d)] * A1 + KAP[(2, d)] * A2 for d in (1, 2, 3)), mp.mpf(7) / 6 * (A1 + A2) - mp.mpf(7) / 12 * M)
    t3 = mp.mpf(3) ** (1 - s); c1 = 1 - t3
    F1 = mp.mpf(1) if inf1 else max(1 - c1 * x1, t3 * max(1, x1))
    F2 = mp.mpf(1) if inf2 else max(1 - c1 * x2, t3 * max(1, x2))
    den = mp.mpf(3.5) * mp.mpf(3) ** (-s) * ((F1 * A1) ** (1 / s) + (F2 * A2) ** (1 / s)) ** s
    return max(Nst, 0) / den


def lp_check():
    """brute-force check of the vertex lemma: fine grid over the mu simplex never beats the closed form"""
    worst = mp.mpf(-1)
    for _ in range(300):
        A1 = rng.uniform(); A2 = 1 - A1; M = rng.uniform(0, 2)
        cf = min(min(float(KAP[(1, d)]) * A1 + float(KAP[(2, d)]) * A2 for d in (1, 2, 3)), 7 / 6 - 7 / 12 * M)
        g = np.linspace(0, 0.5, 101)
        m1, m2 = np.meshgrid(g, g); m3 = 0.5 - m1 - m2; ok = m3 >= -1e-15; m3 = np.maximum(m3, 0)
        val = (2 * (m1 * (float(KAP[(1, 1)]) * A1 + float(KAP[(2, 1)]) * A2) + m2 * (float(KAP[(1, 2)]) * A1 + float(KAP[(2, 2)]) * A2)
                    + m3 * (float(KAP[(1, 3)]) * A1 + float(KAP[(2, 3)]) * A2)) - 3.5 * np.minimum(np.minimum(m1, m2), m3) * M)
        worst = max(worst, cf - val[ok].min())
    return worst


def sup_bound(s, ngrid=41):
    best = (-1, None)
    for inf1 in (False, True):
        for inf2 in (False, True):
            for tau in np.linspace(0, 1, ngrid):
                for x1 in ([0] if inf1 else np.linspace(0, 1.75, ngrid)):
                    for x2 in ([0] if inf2 else np.linspace(0, 1.75, ngrid)):
                        v = float(bound_mp(tau, x1, x2, s, inf1, inf2))
                        if v > best[0]:
                            best = (v, (tau, x1, x2, inf1, inf2))
    v0, (tau, x1, x2, inf1, inf2) = best

    def f(z):
        t = min(max(z[0], 0), 1); a = min(max(z[1], 0), 1.75); b = min(max(z[2], 0), 1.75)
        return -float(bound_mp(t, a, b, s, inf1, inf2))
    res = minimize(f, [tau, x1, x2], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-13, maxiter=4000))
    return -res.fun, res.x, (inf1, inf2)


# ---------------------------------------------------------------- networks at (7,2)
def stats72(net, s):
    return net.h() / net.cost(3 / s) ** s / gammaIF72(s)


def own_data(net):
    A = []; M = []
    for j in (1, 2):
        o = net.own(j)
        A.append(abs(o[0])); M.append(np.max(np.abs(o[1:])))
    return np.array(A), np.array(M)


def bound_net(net, s):
    A, M = own_data(net)
    tot = A.sum(); A_ = A / tot; x = np.where(A > 0, M / np.maximum(A, 1e-300), 0)
    return float(bound_mp(A_[1], x[0], x[1], s))


def ledger_checks(net, s):
    """R12 with every pure-vertex mu and the uniform mu, any (j,T); R13 cost ledger"""
    r = 1 / s
    h = net.h(); viol = 0
    ownd = {j: net.own(j) for j in (1, 2, 3)}
    for mu in ([0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [1 / 6] * 3, list(rng.dirichlet([1, 1, 1]) / 2)):
        K = {j: 2 * sum(mu[d - 1] * float(KAP[(j, d)]) for d in (1, 2, 3)) for j in (1, 2, 3)}
        base = sum(K[j] * ownd[j][0].real for j in (1, 2, 3))
        for j in (1, 2, 3):
            for T in range(3):
                rhs = base - 3.5 * min(mu) * abs(ownd[j][1 + T])
                if h > rhs + 1e-10:
                    viol += 1
    # R13 cost ledger with c1sharp
    A, M = own_data(net)
    t3 = 3 ** (1 - s); c1 = 1 - t3
    lhs = t3 * B_p(7) * net.cost(3 / s) ** s
    Fj = [max(1 - c1 * (M[j] / A[j]), t3 * max(1, M[j] / A[j])) if A[j] > 0 else 0 for j in (0, 1)]
    rhs = sum((Fj[j] * A[j]) ** r for j in (0, 1)) ** s
    return viol, lhs - rhs, lhs


def random72(m, dc=True):
    U1 = rng.standard_normal((m, p)); U2 = rng.standard_normal((m, p)); W = rng.standard_normal((m, p))
    if not dc:
        U1 -= U1.mean(1, keepdims=True); U2 -= U2.mean(1, keepdims=True)
    return cancel_feature(Net(p, U1, U2, W), 3)


def ifpair72(eps):
    rows = []
    for k in (1, 2):
        for i in range(3):
            ph = -np.pi * i / 3
            rows.append(pure_neuron(p, k, 1, 1, 1, ph, ph, 2 * ph))
    U1, U2, W = map(np.array, zip(*rows))
    U1 = U1 + eps * rng.standard_normal(U1.shape); U2 = U2 + eps * rng.standard_normal(U2.shape); W = W + eps * rng.standard_normal(W.shape)
    return cancel_feature(Net(p, U1, U2, W), 3)


def optimise72(s, m=6, starts=3):
    b = 3 / s; best = None; bv = -np.inf
    for st in range(starts):
        net0 = ifpair72(0.2) if st % 2 == 0 else random72(m)
        k = net0.U1.shape[0]
        z0 = np.concatenate([net0.U1.ravel(), net0.U2.ravel(), net0.W.ravel()]) / net0.cost(b) ** (1 / b)

        def unpack(z):
            return Net(p, z[:k * p].reshape(k, p), z[k * p:2 * k * p].reshape(k, p), z[2 * k * p:].reshape(k, p))

        def cin(zz):
            return unpack(zz[:-1]).gaps().ravel() - zz[-1]

        def ceq(zz):
            nt = unpack(zz[:-1]); b3 = nt.neuron_beta(3).sum()
            return np.array([nt.cost(b) - 1, 1e3 * b3.real, 1e3 * b3.imag])
        zz0 = np.concatenate([z0, [unpack(z0).h()]])
        res = minimize(lambda zz: -zz[-1], zz0, method='SLSQP', constraints=[{'type': 'ineq', 'fun': cin}, {'type': 'eq', 'fun': ceq}],
                       options=dict(maxiter=300, ftol=1e-12))
        nt = cancel_feature(unpack(res.x[:-1]), 3)
        v = stats72(nt, s)
        if v > bv:
            bv, best = v, nt
    return best


if __name__ == '__main__':
    t0 = time.time()
    print("vertex lemma (closed form minus grid min over mu; must be <= ~0):", float(lp_check()))
    for s in (1.0, 1.1, 1.15, 1.175, 1.18, 1.182, 1.19, 1.2):
        v, x, inf = sup_bound(s, ngrid=25)
        print(f"s={s}: sup of the closed-form bound = {v:.6f} at tau={x[0]:.4f} x1={x[1]:.4f} x2={x[2]:.4f} inf={inf}", flush=True)
    # Part B: their box_ratio vs my point values
    sys.argv = ['simple72_rig.py', '1.0', '1.182', '0.9995']
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
    import simple72_rig as S72
    from iv import I
    nb = 0; maxdiff = -1
    for k in range(150):
        tau = rng.uniform(); x1 = rng.uniform(0, 1.75); x2 = rng.uniform(0, 1.75); s = rng.uniform(1.0, 1.182)
        inf1 = rng.uniform() < 0.15; inf2 = rng.uniform() < 0.15
        mu = S72.choose_mu(tau, x1, x2, inf1, inf2)
        w = 1e-7
        rb = S72.box_ratio(mu, I(max(tau - w, 0), min(tau + w, 1)), I(max(x1 - w, 0), x1 + w), I(max(x2 - w, 0), x2 + w),
                           inf1, inf2, s - 1e-9, s + 1e-9)
        # my value with the SAME mu
        A1 = 1 - tau; A2 = tau
        M = max(0 if inf1 else A1 * x1, 0 if inf2 else A2 * x2)
        Z = 2 * sum(mp.mpf(float(m_)) for m_ in mu)
        num = (sum(2 * mp.mpf(float(mu[d - 1])) * (KAP[(1, d)] * A1 + KAP[(2, d)] * A2) for d in (1, 2, 3))
               - mp.mpf(3.5) * mp.mpf(float(min(mu))) * M) / Z
        t3 = mp.mpf(3) ** (1 - s); c1 = 1 - t3
        F1 = mp.mpf(1) if inf1 else max(1 - c1 * x1, t3 * max(1, x1)); F2 = mp.mpf(1) if inf2 else max(1 - c1 * x2, t3 * max(1, x2))
        den = mp.mpf(3.5) * mp.mpf(3) ** (-s) * ((F1 * A1) ** (1 / mp.mpf(s)) + (F2 * A2) ** (1 / mp.mpf(s))) ** s
        mine = float(num / den) if num > 0 else float(num)
        opt = float(bound_mp(tau, x1, x2, s, inf1, inf2))
        if rb < mine - 1e-12 or (mine > 0 and mine < opt - 1e-9):
            nb += 1; print("MISMATCH", tau, x1, x2, s, inf1, inf2, rb, mine, opt)
        maxdiff = max(maxdiff, rb - mine)
    print(f"Part B: {nb} enclosure failures over 150 random tiny boxes; max(box - point) = {maxdiff:.2e}")
    # Part C: networks
    nets = [(random72(int(rng.integers(2, 8))), 'random+DC') for _ in range(12)]
    nets += [(random72(int(rng.integers(2, 8)), dc=False), 'random') for _ in range(4)]
    nets += [(ifpair72(e), f'IFpair+{e}') for e in (0.0, 0.05, 0.3)]
    for s in (1.0, 1.1, 1.182):
        nets.append((optimise72(s), f'opt s={s}'))
    nviol = 0; worst = -1
    for net, lab in nets:
        A, M = own_data(net)
        line = f"{lab:12s} A={np.round(A,4)} x={np.round(M/np.maximum(A,1e-300),3)}"
        for s in (1.0, 1.1, 1.182):
            rt = stats72(net, s); bd = bound_net(net, s) if A.min() > 1e-12 else np.nan
            v12, slack13, _ = ledger_checks(net, s)
            ok = (np.isnan(bd) or rt <= bd + 1e-9) and v12 == 0 and slack13 >= -1e-12 and (np.isnan(bd) or bd <= 0.9995)
            nviol += (not ok); worst = max(worst, rt)
            line += f" | s={s}: ratio={rt:+.4f} bound={bd:.4f} R12viol={v12} R13slack={slack13:+.2e}"
        print(line, flush=True)
    print(f"Part C: networks={len(nets)} violations={nviol} max ratio={worst:.4f}  time={time.time()-t0:.0f}s")
