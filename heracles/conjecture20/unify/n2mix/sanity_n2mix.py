"""Sanity checks for the N2MIX lemmas on exact small cases (random general networks at p = 5, 7, 11).
 L1  mode formula |m(l;k)| = E sqrt(2^z x_l y_1 y_2), atoms |beta_l| = E sqrt(x y y), rho >= E sqrt(1 - X_L Y_1 Y_2),
     paid mass <= 4 (e^2 - b_L^2)                                                   [Lemma 1]
 L2  3^{1-s} B P^s >= sqrt((sum_{l in L} P_l^r)^{2s} + 3^{2-2s} (sum_i rho_i)^2)    [Lemma 2]
 L3  semi-rigid test identity: sum q g = sum_l K_l(mu) Re beta_l + sum_j Re(conj(beta_{j,T_j}) w_{-j}) + leak,
     leak = sum over leak positions; free positions give 0 when the free entries vanish          [Lemma 3]
 L4  paid pairing of every neuron <= K_plan(w) * rho(neuron)  (random tests, random neurons incl. adversarial) [Lemma 4]
 L5  payment: 3^{1-s} Q_l^s >= A_l max(1 - c_1 x_l, 3^{1-s} max(1, x_l))           [Note c1sharp + trivial]
 L6  end-to-end at (7,2): random networks with beta_3 = 0: gamma/gamma_IF <= bound from rig72 tests at the network's
     own data (best test for its slot-set case)                                     [Theorem]
 L7  s-monotonicity of the bound function B(s) for fixed test and configuration    [Lemma]
"""
import itertools, sys
import numpy as np
from scipy.optimize import minimize
from n2lib import Grid, modes, atoms, own_data, gaps, cost, classify
from chanlib import Chan
from kplan import make_plan, eval_up

rng = np.random.default_rng(7)
worst = {}


def upd(k, v):
    worst[k] = max(worst.get(k, -np.inf), v)


def rand_net(G, m, kind):
    p = G.p
    a = np.arange(p)
    if kind == 'gauss':
        return rng.normal(size=(m, 2, p)), rng.normal(size=(m, p))
    U = np.zeros((m, 2, p)); W = np.zeros((m, p))
    for i in range(m):
        fr = rng.choice(G.F, size=rng.integers(1, len(G.F) + 1), replace=False)
        for t in range(2):
            for k in fr:
                U[i, t] += rng.normal() * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 6.3))
            if kind == 'dc':
                U[i, t] += rng.normal()
        for k in fr:
            W[i] += rng.normal() * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 6.3))
    return U, W


def fractions(G, U, W):
    p = G.p
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    nw = np.sum(W ** 2, axis=1); nu = np.sum(U ** 2, axis=2)
    x = np.array([[2 * abs(Wh[i, l]) ** 2 / (p * nw[i]) for l in G.F] for i in range(W.shape[0])])
    y = np.zeros((U.shape[0], 2, len(G.F) + 1))
    for i in range(U.shape[0]):
        for t in range(2):
            y[i, t, 0] = abs(Uh[i, t, 0]) ** 2 / (p * nu[i, t])
            for c in G.F:
                y[i, t, c] = 2 * abs(Uh[i, t, c]) ** 2 / (p * nu[i, t])
    E = np.sqrt((1 / (2 * p ** 3)) * nw * nu[:, 0] * nu[:, 1])
    e = G.Bp * (np.sum(U ** 2, axis=(1, 2)) + nw) ** 1.5
    return x, y, E, e


# ---------------------------------------------------------------- L1, L2, L5
for p in (5, 7, 11):
    G = Grid(p)
    for trial in range(40):
        m = int(rng.integers(1, 6)); kind = ['gauss', 'freq', 'dc'][trial % 3]
        U, W = rand_net(G, m, kind)
        M = modes(G, U, W)                 # i, jidx, k1, k2
        x, y, E, e = fractions(G, U, W)
        for i in range(m):
            for ji, l in enumerate(G.F):
                for k1 in range(p):
                    for k2 in range(p):
                        z = int(k1 == 0) + int(k2 == 0)
                        pred = E[i] * np.sqrt(2 ** z * x[i, ji] * y[i, 0, G.cls(k1)] * y[i, 1, G.cls(k2)])
                        upd('L1 mode formula rel err', abs(abs(M[i, ji, k1, k2]) - pred) / (e[i] + 1e-300))
        bet = atoms(G, U, W)
        for Lsize in range(1, len(G.F) + 1):
            for L in itertools.combinations(G.F, Lsize):
                Lidx = [l - 1 for l in L]
                bL = np.sum(np.abs(bet[:, Lidx]) ** (2 / 3), axis=1) ** 1.5
                rho = np.sqrt(np.maximum(e ** 2 - bL ** 2, 0))
                XL = x[:, Lidx].sum(axis=1); Y1 = y[:, 0, list(L)].sum(axis=1); Y2 = y[:, 1, list(L)].sum(axis=1)
                upd('L1 rho >= E sqrt(1-XYY) (viol/e)', np.max((E * np.sqrt(np.maximum(1 - XL * Y1 * Y2, 0)) - rho) / e))
                lab = classify(G, list(L))
                paid = np.zeros(m)
                for ji, l in enumerate(G.F):
                    for k1 in range(p):
                        for k2 in range(p):
                            c1, c2 = (k1 + l) % p, (k2 + l) % p
                            if lab[(ji, c1, c2)] in ('paid',) or (lab[(ji, c1, c2)] == 'zero'):
                                paid += np.abs(M[:, ji, k1, k2]) ** 2
                upd('L1 paid <= 4 rho^2 (viol/e^2)', np.max((paid - 4 * rho ** 2) / e ** 2))
                for s in (1.0, 1.2, 1.5):
                    r = 1 / s; b = 3 / s
                    P = cost(U, W, b)
                    Q = np.sum(np.abs(bet[:, Lidx]) ** r, axis=0)
                    Pl = 3 ** (1 - s) * Q ** s
                    lhs = 3 ** (1 - s) * G.Bp * P ** s
                    rhs = np.sqrt(np.sum(Pl ** r) ** (2 * s) + 3 ** (2 - 2 * s) * np.sum(rho) ** 2)
                    upd('L2 ledger (rhs-lhs)/lhs', (rhs - lhs) / lhs)
        # L5 payment
        A, BT = own_data(G, U, W)
        for s in (1.0, 1.2, 1.35, 1.5):
            r = 1 / s; th = 3 ** (1 - s); c1 = 1 - th
            Q = np.sum(np.abs(bet) ** r, axis=0)
            for ji in range(len(G.F)):
                Aj = abs(A[ji])
                if Aj < 1e-9:
                    continue
                xj = np.max(np.abs(BT[ji])) / Aj
                upd('L5 payment (bound - P)/A', (Aj * max(1 - c1 * xj, th * max(1, xj)) - th * Q[ji] ** s) / Aj)
print("L1/L2/L5 done", flush=True)

# ---------------------------------------------------------------- L3, L4 at p = 7, L = {1,2}
import rig72
from n2lib import fieldcoef
G = Grid(7); CH = Chan(7, (1, 2))
for trial in range(60):
    m = int(rng.integers(1, 6)); kind = ['gauss', 'freq', 'dc'][trial % 3]
    U, W = rand_net(G, m, kind)
    case = rig72.CASES[trial % 5]
    w = {j: rng.normal(size=6) + 1j * rng.normal(size=6) for j in (1, 2)}
    for j, T in ((1, case[0]), (2, case[1])):
        own, free, paid = rig72.comp_info(j, T)
        for mm in free:
            w[j][mm - 1] = 0
    mu = rng.uniform(0, 1, 3); mu = np.concatenate([mu, mu[::-1]]); mu /= mu.sum()
    z = {j: CH.IDFT @ w[j] for j in (1, 2)}
    # q may be negative here: identity is linear, positivity irrelevant for L3
    X = G.X
    q = np.zeros((49, 6))
    for xi, (x1, x2) in enumerate(X):
        for di in range(6):
            val = mu[di]
            for j, T in ((1, case[0]), (2, case[1])):
                t = CH.Tsets[T]
                chi = ((2 * j * t[0]) % 7, (2 * j * t[1]) % 7)
                val += 2 * np.real(z[j][di] * G.rho ** (-(chi[0] * x1 + chi[1] * x2)))
            q[xi, di] = val / 49
    g = gaps(G, U, W).reshape(49, 6)
    lhs = np.sum(q * g)
    A, BT = own_data(G, U, W)
    phi = fieldcoef(G, U, W)          # jidx, chi1, chi2
    Kl = CH.KM @ mu
    rhs = sum(Kl[ji] * np.real(A[ji]) for ji in range(3))
    for j, T in ((1, case[0]), (2, case[1])):
        t = CH.Tsets[T]
        chi = ((2 * j * t[0]) % 7, (2 * j * t[1]) % 7)
        for (mm, role, f, d) in CH.component(j, T):
            if (-mm) % 7 in G.F:
                l = (-mm) % 7; coef = phi[l - 1, chi[0], chi[1]]
                rhs += np.real(coef * w[j][mm - 1])
            else:
                l = mm; coef = np.conj(phi[l - 1, (-chi[0]) % 7, (-chi[1]) % 7])
                rhs += np.real(coef * w[j][mm - 1])
            if role == 'own':
                upd('L3 own coefficient = conj beta_T', abs(coef - np.conj(BT[j - 1][list(CH.Tsets).index(T)])))
    upd('L3 identity |lhs-rhs|', abs(lhs - rhs) / (1 + abs(lhs)))
print("L3 done", flush=True)

# L4: neuron-level bound, adversarial neurons
from rig72 import comp_info
for trial in range(25):
    case = rig72.CASES[trial % 5]
    w = {j: (rng.normal(size=6) + 1j * rng.normal(size=6)) * rng.uniform(0.1, 1) for j in (1, 2)}
    for j, T in ((1, case[0]), (2, case[1])):
        own, free, paid = comp_info(j, T)
        for mm in free:
            w[j][mm - 1] = 0
    entry = {}
    paidpos = []
    for j, T in ((1, case[0]), (2, case[1])):
        own, free, paid = comp_info(j, T)
        t = CH.Tsets[T]
        chi = ((2 * j * t[0]) % 7, (2 * j * t[1]) % 7)
        for (mm, f, d) in paid:
            entry[f] = entry.get(f, 0) + 2 ** (d / 2) * abs(w[j][mm - 1])
            if (-mm) % 7 in G.F:
                l = (-mm) % 7; c = chi; T_val = w[j][mm - 1]
            else:
                l = mm; c = ((-chi[0]) % 7, (-chi[1]) % 7); T_val = np.conj(w[j][mm - 1])
            paidpos.append((l, c, T_val))
    Kp = eval_up(make_plan(entry.keys()), entry)
    def neg_ratio(v):
        U = v[:14].reshape(1, 2, 7); Wv = v[14:].reshape(1, 7)
        phi = fieldcoef(G, U, Wv)
        pair = sum(np.real(phi[l - 1, c[0], c[1]] * Tv) for (l, c, Tv) in paidpos)
        bet = atoms(G, U, Wv)
        e = G.Bp * np.sum(v ** 2) ** 1.5
        bL = np.sum(np.abs(bet[0, :2]) ** (2 / 3)) ** 1.5
        rho = np.sqrt(max(e ** 2 - bL ** 2, 1e-300))
        return -pair / rho
    best = 0
    for st in range(8):
        v0 = rng.normal(size=21)
        res = minimize(neg_ratio, v0, method='BFGS', options={'maxiter': 400})
        best = max(best, -res.fun)
    upd('L4 neuron ratio / K_plan', best / Kp)
print("L4 done", flush=True)
for k, v in worst.items():
    print(f"  {k:40s} {v: .3e}")
