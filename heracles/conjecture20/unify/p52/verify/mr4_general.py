"""Math referee: R12 (delta-Fourier isolation, any p, n, network), R13 (cost ledger at (p,2)), c1sharp spot check,
R17 (shared neuron at b=2), R11(ii), and the p=3 counterexample test of R12."""
import itertools
from fractions import Fraction as Fr
import numpy as np
from mr_net import Net, rand_net, freq_vec

rng = np.random.default_rng(99)


def test_R12():
    worst_id = 0; worst_h = -np.inf; minq = np.inf; cnt = 0
    for p in (3, 5, 7, 11):
        for n in (2, 3):
            if p == 11 and n == 3:
                continue
            for trial in range(12 if p < 11 else 5):
                m = rng.integers(1, 5)
                U, W = rand_net(rng, p, n, m, dc=rng.uniform(0, 2))
                net = Net(p, n, U, W)
                idx, G = net.gaps()
                h = G.min()
                F = net.F
                rho = net.rho
                beta = {l: net.beta(l) for l in F}
                # random symmetric mu on Z_p^*
                half = rng.exponential(size=(p - 1) // 2) + 0.05
                mu = np.zeros(p - 1)
                for d in range(1, (p - 1) // 2 + 1):
                    mu[d - 1] = half[d - 1]; mu[p - d - 1] = half[d - 1]
                mu /= mu.sum()
                K = {l: sum(mu[d - 1] * (1 - np.cos(2 * np.pi * l * d / p)) for d in range(1, p)) for l in F}
                for j in F:
                    for k in range(1, n + 1):
                        for T in itertools.combinations(range(n), k):
                            chi = np.zeros(n, int); chi[list(T)] = (2 * j) % p
                            bT = net.beta(j, T)
                            w = rng.normal() + 1j * rng.normal()
                            dels = np.arange(1, p)
                            ph = rho ** (-(idx @ chi))              # (p^n,)
                            q = p ** (-n) * (mu[None, :] + 2 * (w * rho ** (j * dels)[None, :] * ph[:, None]).real)
                            lhs = np.sum(q * G)
                            rhs = sum(K[l] * beta[l].real for l in F) - p * (w * np.conj(bT)).real
                            worst_id = max(worst_id, abs(lhs - rhs) / (1 + abs(rhs)))
                            # aligned test
                            if abs(bT) > 0:
                                wa = 0.5 * mu.min() * bT / abs(bT)
                                qa = p ** (-n) * (mu[None, :] + 2 * (wa * rho ** (j * dels)[None, :] * ph[:, None]).real)
                                minq = min(minq, qa.min()); assert abs(qa.sum() - 1) < 1e-12
                                bound = sum(K[l] * beta[l].real for l in F) - 0.5 * p * mu.min() * abs(bT)
                                worst_h = max(worst_h, h - bound)
                            cnt += 1
    print(f"R12 identity: max rel err {worst_id:.2e} over {cnt} (p,n,net,j,T,w) cases; aligned test min q = {minq:.2e};"
          f" max[h - bound] = {worst_h:.3e} (<=0)")


def test_R12_p3():
    """p=3 counterexamples (exact_certificate.py weights, same u in every slot).  R12 at p=3: mu = (1/2,1/2),
    K_1 = 3/2, bound = (3/2) Re beta_1 - (3/4)|beta_{1,T}|."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location('ec', '/home/user/grok/heracles/conjecture20/exact_certificate.py')
    ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
    for n, net in ec.NETWORKS.items():
        p = 3; nu = n + 1
        U = np.array([[float(a) for a in row] for row in net['u']]); W = np.array([[float(a) for a in row] for row in net['w']])
        Uf = [[Fr(a) for a in row] for row in net['u']]; Wf = [[Fr(a) for a in row] for row in net['w']]
        # exact h via counts (own code)
        h = None
        for N0 in range(n + 1):
            for N1 in range(n + 1 - N0):
                N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
                z = [Uf[i][0] ** N0 * Uf[i][1] ** N1 * Uf[i][2] ** N2 for i in range(len(Uf))]
                f = [sum(Wf[i][c] * z[i] for i in range(len(Uf))) for c in range(3)]
                for c in range(3):
                    if c != y:
                        h = f[y] - f[c] if h is None else min(h, f[y] - f[c])
        h = float(h)
        Uh = np.fft.fft(U, axis=1); Wh = np.fft.fft(W, axis=1)
        b1 = 2 * 3.0 ** (-nu) * np.sum(np.conj(Wh[:, 1]) * Uh[:, 1] ** n)
        P2 = np.sum(n * np.sum(U ** 2, 1) + np.sum(W ** 2, 1))
        gam = h / P2 ** (nu / 2); s = nu / 2
        gIF = nu ** (1 - s) * 2 * (6 * nu) ** (-nu / 2) * 1.5
        worst = -np.inf
        for k in range(1, n + 1):
            bT = 2 * 3.0 ** (-nu) * np.sum(np.conj(Wh[:, 1]) * Uh[:, 1] ** (n - k) * np.conj(Uh[:, 1]) ** k)
            bound = 1.5 * b1.real - 0.75 * abs(bT)
            worst = max(worst, (h - bound) / abs(bound))
        print(f"R12 at p=3, n={n} counterexample (gamma/gamma_IF = {gam/gIF:.4f} > 1): max over |T| of (h - bound)/|bound| = {worst:.4f} (<=0)")


def c1sharp_check():
    worst = -np.inf
    for t in range(20000):
        k = rng.integers(1, 6); s = rng.choice([1.0, 1.1, 1.182, 1.3, 1.5])
        vr = rng.exponential(size=k) ** rng.uniform(0.2, 3); pp = np.exp(1j * rng.uniform(-1, 1, k) * rng.uniform(0, 3))
        xi = np.exp(1j * rng.uniform(-np.pi, np.pi, k)); eta = np.exp(1j * rng.uniform(-np.pi, np.pi, k))
        b = vr * pp
        A, B, C, D = b.sum(), (b * xi).sum(), (b * eta).sum(), (b * xi * eta).sum()
        Q = np.sum(vr ** (1 / s))
        rhs = 3 ** (1 - s) * Q ** s + (1 - 3 ** (1 - s)) * max(abs(B), abs(C), abs(D))
        worst = max(worst, (abs(A) - rhs) / vr.sum())
    print(f"c1sharp (Note, imported) random atoms: max (|A| - rhs)/L = {worst:.2e} (<=0)")


def ledger(net, s):
    p = net.p; r = 1 / s; c1 = 1 - 3 ** (1 - s)
    tot = 0
    for j in net.F:
        A = abs(net.beta(j))
        if A < 1e-14:
            continue
        x = max(abs(net.beta(j, T)) for T in [(0,), (1,), (0, 1)]) / A
        Fj = max(1 - c1 * x, 3 ** (1 - s) * max(1, x))
        tot += (Fj * A) ** r
    return tot ** s


def test_R13():
    worst = -np.inf
    for p in (5, 7, 11):
        for trial in range(60):
            m = rng.integers(1, 7)
            kind = trial % 3
            if kind == 0:
                U, W = rand_net(rng, p, 2, m, dc=rng.uniform(0, 1.5))
            elif kind == 1:   # IF blocks on two features + noise
                Us = []; Ws = []
                for j in (1, 2):
                    for qq in range(3):
                        Us.append(np.stack([freq_vec(p, j, 1, 2 * np.pi * qq / 3), freq_vec(p, j, 1, 2 * np.pi * qq / 3)]))
                        Ws.append(freq_vec(p, j, 1, 4 * np.pi * qq / 3))
                U = np.array(Us) + 0.05 * rng.normal(size=(6, 2, p)); W = np.array(Ws) + 0.05 * rng.normal(size=(6, p))
            else:             # single-frequency single atoms, balanced
                U = np.array([np.stack([freq_vec(p, j, 1, rng.uniform(0, 6)), freq_vec(p, j, 1, rng.uniform(0, 6))]) for j in (1, 2)])
                W = np.array([freq_vec(p, j, 1, rng.uniform(0, 6)) for j in (1, 2)])
            net = Net(p, 2, U, W)
            for s in (1.0, 1.1, 1.182, 1.5):
                b = 3 / s
                lhs = 3 ** (1 - s) * net.Bp() * net.cost(b) ** s
                worst = max(worst, (ledger(net, s) - lhs) / lhs)
    print(f"R13 ledger: max rel [ledger - 3^(1-s) B_p P^s] = {worst:.2e} (<=0) over p in 5,7,11, s in 1..1.5")


def test_R17():
    p = 7
    U = np.array([[freq_vec(p, 1, 1, 0.3) + freq_vec(p, 2, 1, 1.1), freq_vec(p, 1, 1, -0.4) + freq_vec(p, 2, 1, 2.0)]])
    W = np.array([freq_vec(p, 1, 1, 0.9) + freq_vec(p, 2, 1, -1.3)])
    net = Net(p, 2, U, W)
    e = net.e()[0]
    b1, b2 = net.beta(1), net.beta(2)
    md = net.modes_i(0)
    tot = sum(abs(v) ** 2 for v in md.values())
    own = 0
    for (l, k), v in md.items():
        if l in (1, 2) and all(kk in (l % p, (-l) % p) for kk in k):
            own += abs(v) ** 2
    s = 1.5
    lhs = 3 ** (1 - s) * net.Bp() * net.cost(2) ** s
    Q = [abs(b1) ** (2 / 3), abs(b2) ** (2 / 3)]
    led = (sum((3 ** (1 - s) * Qj ** s) ** (2 / 3) for Qj in Q)) ** s
    print(f"R17 shared neuron p=7: (|b1|^(2/3)+|b2|^(2/3))/e^(2/3) = {(Q[0]+Q[1])/e**(2/3):.6f}; total modes/e^2 = {tot/e**2:.6f};"
          f" own/e^2 = {own/e**2:.6f}; non-own/e^2 = {(tot-own)/e**2:.6f};  b=2: 3^(1-s)B P^s / ledger(sum_j Q_j) = {lhs/led:.6f}")
    # at b > 2 the same neuron has slack
    for b in (2.2, 2.6):
        s = 3 / b; r = 1 / s
        print(f"     b={b}: sum_j |beta_j|^r / e^r = {(abs(b1)**r+abs(b2)**r)/e**r:.4f}  (2^(1-3r/2) = {2**(1-1.5*r):.4f})")
    # F2 channel: mode (l=1; k_t=2) has x-character -(k+1) = -3 = 4 = 2*2 in the conjugate field
    print(f"     F2 channel check: -(1+2) mod 7 = {(-(1+2))%7} = 2*2 mod 7 = {4}")


def test_R11ii():
    km = 1 - np.cos(2 * np.pi / 7)
    ss = np.linspace(1, 1.5, 501)
    rat = 3 ** (2 * ss - 1) * km / 3.5
    print(f"R11(ii) ratio 3^(2s-1) kappa_min/3.5: increasing={np.all(np.diff(rat) > 0)}, max={rat.max():.6f}")
    # numerical: random one-feature networks at (7,2)
    worst = -np.inf
    for t in range(60):
        j = rng.integers(1, 4)
        m = rng.integers(1, 5)
        U = np.array([[freq_vec(7, j, rng.normal(), rng.uniform(0, 6)) + rng.normal() * 0.3 for _ in range(2)] for _ in range(m)])
        W = np.array([freq_vec(7, j, rng.normal(), rng.uniform(0, 6)) for _ in range(m)])
        net = Net(7, 2, U, W)
        for s in (1.0, 1.5):
            b = 3 / s
            gam = net.h() / net.cost(b) ** s
            gIF = 3 ** (1 - s) * net.Bp() * 3.5 * 3 ** (-s)
            worst = max(worst, gam / gIF - 3 ** (2 * s - 1) * km / 3.5)
    print(f"R11(ii) random single-feature nets: max[gamma/gamma_IF - bound] = {worst:.3e} (<=0)")


if __name__ == '__main__':
    test_R12()
    test_R12_p3()
    c1sharp_check()
    test_R13()
    test_R17()
    test_R11ii()
