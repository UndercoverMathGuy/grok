"""hr1: independent checks of the ledgers R2, R3, R4, R5, R12, R13, R17 and the relabellings of R1 / R14.
Random and adversarial networks with DC, multi-frequency inputs, outputs on every frequency."""
import itertools
import numpy as np
from scipy.optimize import minimize
from hr_net import Net, B_p, kappa, rand_net, cos_vec

rng = np.random.default_rng(12345)
out = []


def rep(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


# ---------------------------------------------------------------- consistency: modes vs fields (p=5,n=2)
def check_modes_fields(net):
    p, n = net.p, net.n
    phi = net.field_coeffs()
    err = 0
    for l in net.F:
        acc = np.zeros((p,) * n, complex)
        for i in range(net.m):
            md = net.modes(i)[l]
            # mode k -> character chi = k + l 1
            for k in itertools.product(range(p), repeat=n):
                chi = tuple((np.array(k) + l) % p)
                acc[chi] += md[k]
        err = max(err, np.abs(acc - phi[l]).max())
    # E Phi_l = conj(beta_l); phi_l(chi_{l,T}) = conj(beta_{l,T})
    e2 = 0
    for l in net.F:
        e2 = max(e2, abs(phi[l][(0,) * n] - np.conj(net.beta(l))))
        for r in range(1, n + 1):
            for T in itertools.combinations(range(n), r):
                chi = tuple((2 * l if t in T else 0) % p for t in range(n))
                e2 = max(e2, abs(phi[l][chi] - np.conj(net.beta(l, T))))
    return err, e2


for (p, n) in [(5, 2), (7, 2), (5, 3), (3, 2)]:
    mx = 0
    for trial in range(5):
        net = rand_net(rng, p, n, 4)
        mx = max(mx, *check_modes_fields(net))
    rep(f"[consistency] p={p} n={n}: max |modes->field - field| and own-coefficient identities: {mx:.2e}")

# ---------------------------------------------------------------- R2 mode budget at p=5 (random + adversarial)
p, n = 5, 2
B5 = B_p(5, 3)


def r2_slack(vec):
    U = vec[:10].reshape(1, 2, 5); W = vec[10:].reshape(1, 5)
    net = Net(5, 2, U, W)
    e = B5 * net.norms2()[0] ** 1.5
    b = abs(net.beta_i(1)[0])
    md = net.modes(0)
    tot = sum(np.sum(np.abs(md[l]) ** 2) for l in (1, 2))
    own = 0
    for k in itertools.product((1, 4), repeat=2):
        own += abs(md[1][k]) ** 2
    nonown = tot - own
    return nonown - 4 * (e ** 2 - b ** 2), own - 4 * b ** 2


worst = -np.inf; worst_own = 0
for trial in range(3000):
    v = rng.normal(size=15) * rng.choice([0.1, 1, 3])
    s1, s2 = r2_slack(v)
    worst = max(worst, s1 / (B5 ** 2 * (v @ v) ** 3)); worst_own = max(worst_own, abs(s2))
# adversarial: maximise normalised slack
for trial in range(30):
    v0 = rng.normal(size=15)
    res = minimize(lambda v: -r2_slack(v)[0] / (B5 ** 2 * (v @ v) ** 3), v0, method='Nelder-Mead',
                   options=dict(maxiter=4000, xatol=1e-10, fatol=1e-14))
    worst = max(worst, -res.fun)
rep(f"[R2] max normalised [non-own - 4(e^2-|beta|^2)] / (B5^2|w|^6) = {worst:.3e} (must be <= 0); own-mass identity err {worst_own:.1e}")


# ---------------------------------------------------------------- R3 Pythagorean + non-own l2 norm, R4 ledger, R5
def own_set52():
    return {(1, (0, 0)), (1, (2, 0)), (1, (0, 2)), (1, (2, 2))}


def ledger52(net, s):
    """returns dict with h, A,B,C,D, R, P1, cost terms, nonown norm"""
    e = B5 * net.norms2() ** 1.5
    bi = net.beta_i(1)
    rho_i = np.sqrt(np.maximum(e ** 2 - np.abs(bi) ** 2, 0))
    R = 2 * rho_i.sum()
    r = 1 / s
    Q = np.sum(np.abs(bi) ** r)
    P1 = 3 ** (1 - s) * Q ** s
    P = net.cost(3 / s)
    phi = net.field_coeffs()
    own = own_set52()
    nn = 0
    for l in (1, 2):
        for chi in itertools.product(range(5), repeat=2):
            if (l, chi) in own or (l == 2 and chi == (0, 0)):
                continue
            nn += abs(phi[l][chi]) ** 2
    return dict(R=R, P1=P1, P=P, nonown=np.sqrt(nn), lhs=3 ** (1 - s) * B5 * P ** s, phi=phi)


worst3 = -np.inf; worst3b = -np.inf
for trial in range(400):
    m = rng.integers(1, 8)
    net = rand_net(rng, 5, 2, m, dc=bool(rng.integers(2)))
    for s in (1.0, 1.1, 1.25, 1.5):
        L = ledger52(net, s)
        c = 3 ** (2 - 2 * s) / 4
        worst3 = max(worst3, (np.sqrt(L['P1'] ** 2 + c * L['R'] ** 2) - L['lhs']) / L['lhs'])
    worst3b = max(worst3b, L['nonown'] - L['R'])
rep(f"[R3] max rel [sqrt(P1^2+cR^2) - 3^(1-s)B5 P^s] = {worst3:.3e} (<=0);  max [||nonown phi|| - R] = {worst3b:.3e} (<=0)")


# R4 with ARBITRARY probabilities q (not HS) and networks with beta_2 forced to 0 by a cancelling copy
def kill_beta2(net):
    """append a neuron that cancels beta_2 exactly: copy of neuron 0 with output freq-2 part rotated to cancel.
    Simplest exact route: add neuron with u = (cos 2a, cos 2a) style and output tuned; we instead solve for
    an extra neuron (u1,u2 fixed pure freq-2, w pure freq-2) whose beta_2 equals -beta_2(net)."""
    b2 = net.beta(2)
    a = np.arange(5)
    u = np.cos(2 * np.pi * 2 * a / 5)            # uhat(2) = 5/2
    # beta_2 of neuron (u,u,w) with w = c cos(2 pi 2 a/5 + psi): what(2) = (5/2) c e^{i psi}
    # beta = 2/125 conj(what(2)) (5/2)^2 = 2/125 (5/2)^3 c e^{-i psi}
    mag = abs(b2) / (2 / 125 * (5 / 2) ** 3)
    psi = -(np.angle(-b2))
    w = mag * np.cos(2 * np.pi * 2 * a / 5 + psi)
    U = np.concatenate([net.U, np.stack([u, u])[None]], axis=0)
    W = np.concatenate([net.W, w[None]], axis=0)
    return Net(5, 2, U, W)


worst4 = -np.inf; worst4id = 0; cnt = 0
cs = np.arange(1, 5)
c1 = 1 - np.exp(-2j * np.pi * cs / 5); c2 = 1 - np.exp(-2j * np.pi * 2 * cs / 5)
for trial in range(300):
    net = kill_beta2(rand_net(rng, 5, 2, rng.integers(1, 6)))
    assert abs(net.beta(2)) < 1e-12
    q = rng.random((25, 4)) ** rng.choice([1, 4, 10]); q /= q.sum()
    G = net.gaps()
    lhs = (q * G).sum()
    C1 = q @ c1; C2 = q @ c2
    Ch1 = np.fft.fftn(C1.reshape(5, 5)) / 25; Ch2 = np.fft.fftn(C2.reshape(5, 5)) / 25    # Chat(psi) = 1/25 sum C rho^{-psi x}
    neg = lambda chi: tuple((-np.array(chi)) % 5)
    a0 = 25 * Ch1[(0, 0)]
    aT = {T: 25 * Ch1[neg(chi)] for T, chi in [((0,), (2, 0)), ((1,), (0, 2)), ((0, 1), (2, 2))]}
    A = net.beta(1); bt = {T: net.beta(1, T) for T in aT}
    a = (np.conj(A) * a0).real + sum((np.conj(bt[T]) * aT[T]).real for T in aT)
    nn = 0
    for l, Ch in ((1, Ch1), (2, Ch2)):
        for chi in itertools.product(range(5), repeat=2):
            if (l, chi) in own_set52() or (l == 2 and chi == (0, 0)):
                continue
            nn += abs(25 * Ch[neg(chi)]) ** 2
    nn = np.sqrt(nn)
    L = ledger52(net, 1.5)
    # exact identity: sum q g = 25 Re sum phi_l(chi) Chat_l(-chi)
    ident = sum((25 * L['phi'][l][chi] * Ch[neg(chi)]).real for l, Ch in ((1, Ch1), (2, Ch2))
                for chi in itertools.product(range(5), repeat=2))
    worst4id = max(worst4id, abs(ident - lhs))
    worst4 = max(worst4, lhs - (a + L['R'] * nn))
    # R5 elimination with F = P1 at s=1.5
    F = L['P1']; c = 1 / 12
    if a + nn * L['R'] > 0:
        ratio_true_chain = (a + nn * L['R']) / np.sqrt(F ** 2 + c * L['R'] ** 2)
        closed = np.sqrt(max(a, 0) ** 2 / F ** 2 + nn ** 2 / c)
        assert ratio_true_chain <= closed * (1 + 1e-12)
        cnt += 1
rep(f"[R4] identity err {worst4id:.2e}; max [sum q g - (a + R n)] = {worst4:.3e} (<=0); R5 closed form held in {cnt} cases")


# ---------------------------------------------------------------- R12 isolation identity: every p, n, network
def r12_check(net, j, T, w, mu):
    p, n = net.p, net.n
    chi = np.array([(2 * j if t in T else 0) % p for t in range(n)])
    G = net.gaps()                       # N x (p-1)
    ds = np.arange(1, p)
    ph = np.exp(2j * np.pi * (net.X @ chi) / p)            # rho^{<chi,x>}
    q = p ** (-n) * (mu[None, :] + 2 * np.real(w * np.exp(2j * np.pi * j * ds / p)[None, :] / ph[:, None]))
    lhs = (q * G).sum()
    K = {l: sum(mu[d - 1] * kappa(p, l, d) for d in ds) for l in net.F}
    rhs = sum(K[l] * net.beta(l).real for l in net.F) - p * np.real(w * np.conj(net.beta(j, T)))
    return lhs, rhs, q.min(), q.sum()


worst12 = 0; worst12b = -np.inf; minq = np.inf
for (p, n) in [(3, 2), (3, 3), (5, 2), (5, 3), (7, 2), (11, 2), (7, 3), (13, 2)]:
    for trial in range(8):
        net = rand_net(rng, p, n, rng.integers(1, 6), dc=True)
        half = rng.random((p - 1) // 2); mu = np.concatenate([half, half[::-1]]); mu /= mu.sum()
        for j in net.F:
            for r in range(1, n + 1):
                for T in itertools.combinations(range(n), r):
                    w = complex(*rng.normal(size=2))
                    lhs, rhs, _, _ = r12_check(net, j, T, w, mu)
                    worst12 = max(worst12, abs(lhs - rhs) / (1 + abs(lhs)))
                    bjt = net.beta(j, T)
                    if abs(bjt) > 0:
                        wa = 0.5 * mu.min() * bjt / abs(bjt)
                        lhs, rhs, qm, qs = r12_check(net, j, T, wa, mu)
                        minq = min(minq, qm)
                        worst12b = max(worst12b, net.h() - rhs)
rep(f"[R12] identity rel err {worst12:.2e}; aligned test: min q {minq:.2e} (>=0), max [h - bound] {worst12b:.3e} (<=0)")


# ---------------------------------------------------------------- R13 cost ledger at p=5,7,11 (n=2)
def r13_gap(net, s):
    p = net.p; r = 1 / s; c1 = 1 - 3 ** (1 - s)
    lhs = 3 ** (1 - s) * B_p(p, 3) * net.cost(3 / s) ** s
    terms = []
    for j in net.F:
        A = abs(net.beta(j))
        if A < 1e-14:
            continue
        M = max(abs(net.beta(j, T)) for T in [(0,), (1,), (0, 1)])
        x = M / A
        Fj = max(1 - c1 * x, 3 ** (1 - s) * max(1, x))
        terms.append((Fj * A) ** r)
    rhs = sum(terms) ** s
    return (rhs - lhs) / lhs


worst13 = -np.inf
for p in (5, 7, 11):
    for trial in range(150):
        kind = trial % 3
        if kind == 0:
            net = rand_net(rng, p, 2, rng.integers(1, 7))
        elif kind == 1:      # single-atom-ish neurons: pure frequency j with random phases
            m = rng.integers(1, 5); U = np.zeros((m, 2, p)); W = np.zeros((m, p))
            for i in range(m):
                j = rng.integers(1, (p - 1) // 2 + 1)
                for t in range(2):
                    U[i, t] = cos_vec(p, j, rng.uniform(0, 6.3), rng.uniform(0.2, 2))
                W[i] = cos_vec(p, j, rng.uniform(0, 6.3), rng.uniform(0.2, 2))
            net = Net(p, 2, U, W)
        else:                # split neurons (equal split between two frequencies)
            m = rng.integers(1, 4); U = np.zeros((m, 2, p)); W = np.zeros((m, p))
            for i in range(m):
                j1, j2 = rng.choice(np.arange(1, (p - 1) // 2 + 1), 2, replace=(p == 5 and False))
                for t in range(2):
                    U[i, t] = cos_vec(p, j1, rng.uniform(0, 6.3)) + cos_vec(p, j2, rng.uniform(0, 6.3))
                W[i] = cos_vec(p, j1, rng.uniform(0, 6.3)) + cos_vec(p, j2, rng.uniform(0, 6.3))
            net = Net(p, 2, U, W)
        for s in (1.0, 1.1, 1.182, 1.25, 1.5):
            worst13 = max(worst13, r13_gap(net, s))
rep(f"[R13] max rel [ledger - 3^(1-s) B_p P^s] over p=5,7,11 = {worst13:.3e} (<=0)")

# ---------------------------------------------------------------- R17 split neuron at p=7
p = 7; a = np.arange(p)
U = np.zeros((1, 2, p)); W = np.zeros((1, p))
ph = rng.uniform(0, 6.3, 6)
for t in range(2):
    U[0, t] = np.cos(2 * np.pi * a / p + ph[2 * t]) + np.cos(2 * np.pi * 2 * a / p + ph[2 * t + 1])
W[0] = np.cos(2 * np.pi * a / p + ph[4]) + np.cos(2 * np.pi * 2 * a / p + ph[5])
net = Net(p, 2, U, W)
e = B_p(7, 3) * net.norms2()[0] ** 1.5
b1, b2 = abs(net.beta(1)), abs(net.beta(2))
md = net.modes(0)
tot = sum(np.sum(np.abs(md[l]) ** 2) for l in net.F)
own = sum(abs(md[l][k]) ** 2 for l in (1, 2) for k in itertools.product((l, p - l), repeat=2))
rep(f"[R17] (|b1|^(2/3)+|b2|^(2/3))/e^(2/3) = {(b1**(2/3)+b2**(2/3))/e**(2/3):.6f}; total/e^2 = {tot/e**2:.6f}; non-own/e^2 = {(tot-own)/e**2:.6f}")
# which non-own modes share a character (as functions of x, up to conjugation) with feature 2's own characters 4*1_T?
own2 = {tuple((4 if t in T else 0) % 7 for t in range(2)) for T in [(0,), (1,), (0, 1)]}
hits = []
for l in (1, 2):
    for k in itertools.product(range(7), repeat=2):
        if abs(md[l][k]) < 1e-12:
            continue
        chi = tuple((np.array(k) + l) % 7); nchi = tuple((-np.array(chi)) % 7)
        if (l == 2 and chi in own2):
            continue
        if chi in own2 or nchi in own2:
            hits.append((l, k, chi))
rep(f"[R17] non-own modes whose character is +-(feature-2 sign-flip character): {hits[:6]} ... total {len(hits)}")

# ---------------------------------------------------------------- R1 relabelling (5,2) and R14 relabelling (7,2)
for (p, mult) in [(5, 3), (7, 2)]:
    net = rand_net(rng, p, 2, 4)
    idx = (mult * np.arange(p)) % p
    net2 = Net(p, 2, net.U[:, :, idx], net.W[:, idx])
    perm = {}
    for j in net.F:
        jj = (pow(mult, -1, p) * j) % p      # beta'_j = beta_{(mult^-1) j}?  test all
        perm[j] = jj
    msg = []
    for j in net.F:
        b2 = net2.beta(j)
        cands = [(k, abs(b2 - net.beta(k)), abs(b2 - np.conj(net.beta(k)))) for k in net.F]
        best = min(cands, key=lambda c: min(c[1], c[2]))
        msg.append(f"beta'_{j} = {'beta' if best[1] < best[2] else 'conj beta'}_{best[0]} (err {min(best[1], best[2]):.1e})")
    rep(f"[relabel x->{mult}x at p={p}] margin diff {abs(net.h()-net2.h()):.1e}; " + "; ".join(msg))

open('hr1_ledgers.out', 'w').write("\n".join(out) + "\n")
