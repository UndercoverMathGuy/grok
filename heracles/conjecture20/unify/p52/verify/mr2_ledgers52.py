"""Math referee: independent checks of R2 (mode budget), R3 (Pythagorean cost), R4 (margin ledger with ARBITRARY
probabilities q, not only HS ones), R5 (R-elimination), R6 (HS family: probability + torus equivariance) at (5,2).
Also R1(ii) (relabelling a -> 3a) and the large-interference cut constants."""
import itertools
import numpy as np
from scipy.optimize import minimize
from mr_net import Net, rand_net, kill_beta, freq_vec

rng = np.random.default_rng(2026)
p, n, nu = 5, 2, 3
rho = np.exp(2j * np.pi / p)
B5 = 2 * 30.0 ** -1.5
OWN1 = [(-1, -1), (1, -1), (-1, 1), (1, 1)]        # k for (1;k) own modes: chi = k + 1 in {0, 2}^2


def nonown_mass_i(net, i):
    md = net.modes_i(i)
    tot = sum(abs(v) ** 2 for v in md.values())
    own = sum(abs(md[(1, tuple(np.array(k) % p))]) ** 2 for k in OWN1)
    return tot, own, tot - own


def test_R2():
    worst = -np.inf
    cases = 0
    for trial in range(400):
        kind = trial % 4
        if kind == 0:
            U, W = rand_net(rng, p, n, 1, dc=rng.uniform(0, 2))
        elif kind == 1:   # pure frequency 1 balanced: equality case of the budget
            U = np.stack([freq_vec(p, 1, 1, rng.uniform(0, 6)) for _ in range(2)])[None]
            W = freq_vec(p, 1, 1, rng.uniform(0, 6))[None]
        elif kind == 2:   # frequencies 1 and 2 mixed, no DC
            U = np.stack([freq_vec(p, 1, rng.normal(), rng.uniform(0, 6)) + freq_vec(p, 2, rng.normal(), rng.uniform(0, 6)) for _ in range(2)])[None]
            W = (freq_vec(p, 1, rng.normal(), rng.uniform(0, 6)) + freq_vec(p, 2, rng.normal(), rng.uniform(0, 6)))[None]
        else:             # unbalanced norms
            U, W = rand_net(rng, p, n, 1, dc=0.3)
            U[0, 0] *= 3.0; W[0] *= 0.2
        net = Net(p, n, U, W)
        e = net.e()[0]; b = net.beta_i(1)[0]
        tot, own, non = nonown_mass_i(net, 0)
        assert abs(own - 4 * abs(b) ** 2) < 1e-12 * (1 + own)
        worst = max(worst, (non - 4 * (e ** 2 - abs(b) ** 2)) / e ** 2)
        cases += 1
    # adversarial: maximise non-own mass / (4(e^2 - |beta|^2))
    def negr(v):
        U = v[:10].reshape(1, 2, 5); W = v[10:].reshape(1, 5)
        net = Net(p, n, U, W)
        e = net.e()[0]; b = net.beta_i(1)[0]
        tot, own, non = nonown_mass_i(net, 0)
        return -(non / (4 * (e ** 2 - abs(b) ** 2) + 1e-30))
    best = 0
    for st in range(6):
        r = minimize(negr, rng.normal(size=15), method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-10, fatol=1e-12))
        best = max(best, -r.fun)
    print(f"R2  max over {cases} neurons of (non-own - 4(e^2-|b|^2))/e^2 = {worst:.3e}  (<=0);  adversarial sup ratio = {best:.6f} (<=1)")


def nonown_field_norm(net):
    v = []
    for l in (1, 2):
        for chi in itertools.product(range(p), repeat=2):
            k = tuple((np.array(chi) - l) % p)
            if l == 1 and tuple(np.array(chi) % p) in [(0, 0), (2, 0), (0, 2), (2, 2)]:
                continue
            v.append(net.phi(l, chi))
    return np.linalg.norm(v)


def R_of(net):
    e = net.e(); b = net.beta_i(1)
    return 2 * np.sum(np.sqrt(np.maximum(e ** 2 - abs(b) ** 2, 0)))


def test_R3():
    worst = -np.inf; worst2 = -np.inf
    for trial in range(150):
        m = rng.integers(1, 7)
        U, W = rand_net(rng, p, n, m, dc=rng.uniform(0, 1.5))
        if trial % 3 == 0:   # IF-like block plus noise
            U = np.concatenate([U * 0.05, np.stack([np.stack([freq_vec(p, 1, 1, 2 * np.pi * q / 3), freq_vec(p, 1, 1, 2 * np.pi * q / 3)]) for q in range(3)])])
            W = np.concatenate([W * 0.05, np.stack([freq_vec(p, 1, 1, 4 * np.pi * q / 3) for q in range(3)])])
        net = Net(p, n, U, W)
        R = R_of(net)
        for s in (1.0, 1.2, 1.5):
            b = 3.0 / s; r = 1 / s
            P = net.cost(b)
            Q = np.sum(abs(net.beta_i(1)) ** r)
            P1 = 3 ** (1 - s) * Q ** s
            lhs = 3 ** (1 - s) * B5 * P ** s
            rhs = np.sqrt(P1 ** 2 + 3 ** (2 - 2 * s) * R ** 2 / 4)
            worst = max(worst, (rhs - lhs) / lhs)
        worst2 = max(worst2, nonown_field_norm(net) - R)
    print(f"R3  max rel [sqrt(P1^2 + 3^(2-2s)R^2/4) - 3^(1-s)B5 P^s] = {worst:.3e} (<=0);  max[||non-own field|| - R] = {worst2:.3e} (<=0)")


def gq_data(net):
    idx, G = net.gaps()
    return idx, G


def test_R4():
    worst_h = -np.inf; worst_b = -np.inf; worst_id = 0
    for trial in range(120):
        m = rng.integers(1, 6)
        U, W = rand_net(rng, p, n, m, dc=rng.uniform(0, 1.5))
        U, W = kill_beta(p, n, U, W, 2)
        net = Net(p, n, U, W)
        assert abs(net.beta(2)) < 1e-12
        idx, G = net.gaps()
        h = G.min()
        A = net.beta(1); bT = [net.beta(1, T) for T in [(0,), (1,), (0, 1)]]
        R = R_of(net)
        for qq in range(4):
            q = rng.exponential(size=G.shape) ** (1 + 3 * qq)
            if qq == 3:      # concentrate on few points
                q = np.zeros(G.shape); q[rng.integers(0, 25, 3), rng.integers(0, 4, 3)] = rng.random(3)
            q /= q.sum()
            sqg = np.sum(q * G)
            # C_j(x) and its Fourier coefficients
            dels = np.arange(1, p)
            Chat = {}
            for j in (1, 2):
                Cj = q @ (1 - rho ** (-j * dels))           # per x (25,)
                for psi in itertools.product(range(p), repeat=2):
                    Chat[(j, psi)] = np.mean(Cj * rho ** (-(idx @ np.array(psi))))
            a0 = 25 * Chat[(1, (0, 0))]
            aT = [25 * Chat[(1, tuple((-np.array(chi)) % p))] for chi in [(2, 0), (0, 2), (2, 2)]]
            own = (np.conj(A) * a0).real + sum((np.conj(bT[k]) * aT[k]).real for k in range(3))
            nv = []
            for j in (1, 2):
                for chi in itertools.product(range(p), repeat=2):
                    if j == 1 and chi in [(0, 0), (2, 0), (0, 2), (2, 2)]:
                        continue
                    if j == 2 and chi == (0, 0):
                        continue
                    nv.append(Chat[(j, tuple((-np.array(chi)) % p))])
            nn = 25 * np.linalg.norm(nv)
            # exact identity sum q g = 25 Re sum phi Chat(-chi)
            ident = 0
            for j in (1, 2):
                for chi in itertools.product(range(p), repeat=2):
                    ident += net.phi(j, chi) * Chat[(j, tuple((-np.array(chi)) % p))]
            ident = 25 * ident.real
            worst_id = max(worst_id, abs(ident - sqg))
            worst_h = max(worst_h, h - sqg)
            worst_b = max(worst_b, sqg - (own + R * nn))
    print(f"R4  max|sum qg - 25 Re sum phi Chat| = {worst_id:.2e};  max[h - sum qg] = {worst_h:.3e} (<=0);"
          f"  max[sum qg - (own + R n)] = {worst_b:.3e} (<=0)   [arbitrary q, beta_2 = 0 forced]")


def test_R5():
    worst = -np.inf
    for t in range(3000):
        a = rng.normal() * 2; nn = abs(rng.normal()); F = abs(rng.normal()) + 1e-3; c = abs(rng.normal()) + 1e-3
        R = np.concatenate([[0], np.geomspace(1e-6, 1e6, 4000)])
        sup = np.max((a + nn * R) / np.sqrt(F ** 2 + c * R ** 2))
        cf = np.sqrt(max(a, 0) ** 2 / F ** 2 + nn ** 2 / c)
        worst = max(worst, sup - cf)
    print(f"R5  max[grid sup_R (a+nR)/sqrt(F^2+cR^2) - sqrt(a_+^2/F^2+n^2/c)] = {worst:.3e} (<=0)")


def hs_q(Gs, t=(0.0, 0.0)):
    E = list(itertools.product(range(3), repeat=2))
    q = np.zeros((25, 4))
    xs = np.array(list(itertools.product(range(p), repeat=2)))
    for di in range(4):
        for xi, x in enumerate(xs):
            th = 2 * np.pi * 2 * x / 5 - np.array(t)
            v = np.array([np.exp(1j * (e[0] * th[0] + e[1] * th[1])) for e in E])
            q[xi, di] = (np.conj(v) @ Gs[di] @ v).real / 25
    return xs, q


def cert_numbers(xs, q, A, B, C, D):
    dels = np.arange(1, p)
    Chat = {}
    for j in (1, 2):
        Cj = q @ (1 - rho ** (-j * dels))
        for psi in itertools.product(range(p), repeat=2):
            Chat[(j, psi)] = np.mean(Cj * rho ** (-(xs @ np.array(psi))))
    a0 = 25 * Chat[(1, (0, 0))]
    aT = [25 * Chat[(1, tuple((-np.array(chi)) % p))] for chi in [(2, 0), (0, 2), (2, 2)]]
    nv = [Chat[(j, tuple((-np.array(chi)) % p))] for j in (1, 2) for chi in itertools.product(range(p), repeat=2)
          if not (j == 1 and chi in [(0, 0), (2, 0), (0, 2), (2, 2)]) and not (j == 2 and chi == (0, 0))]
    own = (np.conj(A) * a0).real + (np.conj(B) * aT[0]).real + (np.conj(C) * aT[1]).real + (np.conj(D) * aT[2]).real
    return own, 25 * np.linalg.norm(nv), q.min(), q.sum()


def test_R6():
    worst_inv = 0; minq = np.inf; worst_mass = 0
    for trial in range(30):
        Gs = []
        for di in range(4):
            M = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
            Gs.append(M @ M.conj().T)
        tr = sum(np.trace(G).real for G in Gs)
        Gs = [G / tr for G in Gs]
        A = np.exp(1j * rng.uniform(-1, 1)); B, C, D = rng.normal(size=3) + 1j * rng.normal(size=3)
        xs, q = hs_q(Gs)
        own0, n0, mq, ms = cert_numbers(xs, q, A, B, C, D)
        minq = min(minq, mq); worst_mass = max(worst_mass, abs(ms - 1))
        for k in range(3):
            t = rng.uniform(-np.pi, np.pi, 2)
            # rotated certificate D_t^* G D_t  <=>  q_t(x) = v(theta - t)^* G v(theta - t); data rotated B e^{it1}, C e^{it2}, D e^{i(t1+t2)}
            xs, qt = hs_q(Gs, t)
            own1, n1, mq, ms = cert_numbers(xs, qt, A, B * np.exp(1j * t[0]), C * np.exp(1j * t[1]), D * np.exp(1j * (t[0] + t[1])))
            minq = min(minq, mq); worst_mass = max(worst_mass, abs(ms - 1))
            worst_inv = max(worst_inv, abs(own1 - own0), abs(n1 - n0))
            # phi_pay invariance
            pp0 = np.angle(A * D * np.conj(B * C)); pp1 = np.angle(A * D * np.exp(1j * (t[0] + t[1])) * np.conj(B * np.exp(1j * t[0]) * C * np.exp(1j * t[1])))
            worst_inv = max(worst_inv, abs(np.exp(1j * pp0) - np.exp(1j * pp1)))
    print(f"R6  HS family: min q = {minq:.3e} (>=0), max|mass-1| = {worst_mass:.2e};  torus rotation: max change of"
          f" (own pairing, n, phi_pay) = {worst_inv:.2e}")


def test_R1ii():
    worst = 0
    for trial in range(30):
        U, W = rand_net(rng, p, n, rng.integers(1, 5), dc=1.0)
        net = Net(p, n, U, W)
        perm = (3 * np.arange(p)) % p
        U2 = U[:, :, perm]; W2 = W[:, perm]
        net2 = Net(p, n, U2, W2)
        worst = max(worst, abs(net2.beta(1) - net.beta(2)), abs(net2.beta(2) - np.conj(net.beta(1))),
                    abs(net2.h() - net.h()), abs(net2.cost(2.3) - net.cost(2.3)))
    print(f"R1(ii) relabelling a->3a: max |beta'_1 - beta_2|, |beta'_2 - conj beta_1|, |dh|, |dcost| = {worst:.2e}")


def test_cut():
    ka = 1 - np.cos(2 * np.pi / 5); G = 2.5 * 2 ** -1.5
    thr = ka / (0.995 * G)
    print(f"cut  kappa_a/Gamma_3/2 = {ka/G:.6f};  THR_I = {thr:.6f};  1.4/sqrt3 = {1.4/np.sqrt(3):.6f} >= THR_I: {1.4/np.sqrt(3) >= thr};"
          f"  trivial ratio at the cut <= {ka/(G*1.4/np.sqrt(3)):.4f}")


if __name__ == '__main__':
    test_R1ii()
    test_cut()
    test_R5()
    test_R2()
    test_R3()
    test_R6()
    test_R4()
