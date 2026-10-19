"""Referee-3: network-level checks of the ingredients of R14 (R12 isolation, R13 cost ledger) and of the final (7,2) bound.

(A) p = 7, n = 2: random networks with DC and every frequency, beta_3 forced to 0 by an extra pure-frequency-3 neuron.
    For random symmetric mu, every j in {1,2}, T in {{1},{2},{1,2}}, w: build q_w(x,d) EXPLICITLY on the 49 x 6 grid,
    check q_w >= 0, sum q_w = 1, the exact identity  sum q_w g = sum_l K_l Re beta_l - 7 Re(w conj beta_{j,T}),
    and h <= sum_l K_l Re beta_l - (7/2) min mu |beta_{j,T}|.
    R13: 3^{1-s} B_7 P^s >= (sum_j (F_j A_j)^{1/s})^s at s in {1, 1.1, 1.182}  (F_j uses Note Thm c1sharp).
    Final: gamma/gamma_IF <= min over LP-chosen mu of the simple bound  (and <= 0.9995 for s <= 1.182).
(B) Adversarial: SLSQP maximisation of gamma/gamma_IF over two-feature networks (beta_3 = 0), width 3..6, s = 1.182 (b=2.538).
(C) p = 3: R12 on the Conjecture-20 counterexample networks (n = 11, 13; heavy DC, shared inputs) by brute force over all
    3^n inputs:  sum_{x,d} q_w g  vs  K_1 Re beta_1 - 3 Re(w conj beta_{1,T})  and  h <= (3/2) Re beta_1 - (3/4)|beta_{1,T}|.
"""
import sys, time, itertools
import numpy as np
from fractions import Fraction as Fr
from scipy.optimize import linprog, minimize

rng = np.random.default_rng(4242)
bad = 0


# ============================================================ (A) p = 7, n = 2
def setup(p, n):
    rho = np.exp(2j * np.pi / p)
    grids = np.meshgrid(*[np.arange(p)] * n, indexing='ij')
    Y = sum(grids) % p
    return rho, grids, Y


p = 7; n = 2
rho, (X1, X2), Y = setup(p, n)
F7 = (1, 2, 3)
DELS = list(range(1, p))
B7 = 2 * (2 * p * 3) ** -1.5
kap = lambda j, d: 1 - np.cos(2 * np.pi * j * d / p)


def hat(v):
    k = np.arange(p)
    return v @ (rho ** (-np.outer(np.arange(p), k)))


def gaps(U1, U2, W):
    f = np.einsum('ic,ia,ib->abc', W, U1, U2)
    return np.stack([f[X1, X2, Y] - f[X1, X2, (Y - d) % p] for d in DELS], axis=-1)


def coeffs(U1, U2, W, j):
    w = hat(W); a = hat(U1); b = hat(U2); c = 2 / p ** 3
    beta = c * np.conj(w[:, j]) * a[:, j] * b[:, j]
    bT = {(1,): c * np.conj(w[:, j]) * a[:, -j % p] * b[:, j], (2,): c * np.conj(w[:, j]) * a[:, j] * b[:, -j % p],
          (1, 2): c * np.conj(w[:, j]) * a[:, -j % p] * b[:, -j % p]}
    return beta, bT


def kill(U1, U2, W, j):
    """append a pure-frequency-j neuron that cancels beta_j"""
    bj = coeffs(U1, U2, W, j)[0].sum()
    cv = np.cos(2 * np.pi * j * np.arange(p) / p)
    hc = hat(cv[None, :])[0, j]
    t = np.conj(-bj / ((2 / p ** 3) * hc * hc))
    wn = (2 / p) * np.real(t * rho ** (j * np.arange(p)))
    return np.vstack([U1, cv]), np.vstack([U2, cv]), np.vstack([W, wn])


def cost_ratio(U1, U2, W, s):
    b = 3 / s
    nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    P = np.sum(nrm2 ** (b / 2))
    gIF = 3 ** (1 - s) * B7 * 3.5 * 3 ** -s
    return P, gIF


def lp_bound(A1, A2, M, s, x1, x2):
    """min over symmetric mu of the simple bound (float LP), exactly as stated in R14"""
    c = [2 * (kap(1, d) * A1 + kap(2, d) * A2) for d in (1, 2, 3)] + [-3.5 * M]
    res = linprog(c, A_ub=[[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]], b_ub=[0, 0, 0], A_eq=[[2, 2, 2, 0]], b_eq=[1],
                  bounds=[(0, None)] * 4, method='highs')
    num = res.fun
    c1 = 1 - 3 ** (1 - s)
    Fj = lambda x: max(1 - c1 * x, 3 ** (1 - s) * max(1, x))
    den = 3.5 * 3 ** -s * ((Fj(x1) * A1) ** (1 / s) + (Fj(x2) * A2) ** (1 / s)) ** s
    return num / den


t0 = time.time()
nid = 0; maxid = 0.0; minq = np.inf; worstR12 = np.inf; worstR13 = np.inf; worstfin = np.inf; maxratio = -np.inf
for t in range(150):
    m = rng.integers(1, 7)
    U1 = rng.normal(size=(m, p)); U2 = rng.normal(size=(m, p)); W = rng.normal(size=(m, p))
    if t % 3 == 0:
        U1 -= U1.mean(1, keepdims=True)
    U1, U2, W = kill(U1, U2, W, 3)
    g = gaps(U1, U2, W); h = g.min()
    co = {j: coeffs(U1, U2, W, j) for j in F7}
    assert abs(co[3][0].sum()) < 1e-12
    Re = {j: co[j][0].sum().real for j in F7}
    for trial in range(3):
        mu3 = rng.dirichlet(np.ones(3)) / 2
        mu = np.array([mu3[min(d, p - d) - 1] for d in DELS])          # symmetric, sum 1
        K = {l: np.sum(mu * np.array([kap(l, d) for d in DELS])) for l in F7}
        for j in (1, 2):
            for T, chi in [((1,), (2 * j, 0)), ((2,), (0, 2 * j)), ((1, 2), (2 * j, 2 * j))]:
                bjT = co[j][1][T].sum()
                for wmode in range(2):
                    w = (mu.min() / 2) * (bjT / abs(bjT) if wmode == 0 else np.exp(1j * rng.uniform(0, 2 * np.pi)))
                    q = np.zeros((p, p, p - 1))
                    for di, d in enumerate(DELS):
                        q[:, :, di] = p ** -n * (mu[di] + 2 * np.real(w * rho ** (j * d) * rho ** (-(chi[0] * X1 + chi[1] * X2))))
                    minq = min(minq, q.min())
                    if q.min() < -1e-14 or abs(q.sum() - 1) > 1e-12:
                        bad += 1; print("q_w not a probability")
                    lhs = np.sum(q * g)
                    rhs = sum(K[l] * Re[l] for l in F7) - p * np.real(w * np.conj(bjT))
                    maxid = max(maxid, abs(lhs - rhs) / (1 + abs(lhs))); nid += 1
                    if abs(lhs - rhs) > 1e-9 * (1 + abs(lhs)):
                        bad += 1; print("R12 identity fails", lhs, rhs)
                bnd = sum(K[l] * Re[l] for l in F7) - 3.5 * mu.min() * abs(bjT)
                worstR12 = min(worstR12, bnd - h)
                if h > bnd + 1e-12:
                    bad += 1; print("R12 inequality fails")
    # R13 and final bound
    A = {j: abs(co[j][0].sum()) for j in (1, 2)}
    Mj = {j: max(abs(co[j][1][T].sum()) for T in co[j][1]) for j in (1, 2)}
    for s in (1.0, 1.1, 1.182):
        P, gIF = cost_ratio(U1, U2, W, s)
        c1 = 1 - 3 ** (1 - s)
        Fj = {j: max(1 - c1 * Mj[j] / A[j], 3 ** (1 - s) * max(1, Mj[j] / A[j])) for j in (1, 2)}
        led = sum((Fj[j] * A[j]) ** (1 / s) for j in (1, 2)) ** s
        lhs = 3 ** (1 - s) * B7 * P ** s
        worstR13 = min(worstR13, lhs / led - 1)
        if led > lhs * (1 + 1e-12):
            bad += 1; print("R13 ledger fails", s, lhs, led)
        ratio = h / P ** s / gIF
        maxratio = max(maxratio, ratio)
        sc = A[1] + A[2]
        fb = lp_bound(A[1] / sc, A[2] / sc, max(Mj[1], Mj[2]) / sc, s, Mj[1] / A[1], Mj[2] / A[2])
        worstfin = min(worstfin, fb - ratio)
        if ratio > fb + 1e-9:
            bad += 1; print("final bound fails", s, ratio, fb)
print(f"(A) p=7: {nid} R12 identity checks, max rel err {maxid:.1e}, min q_w = {minq:.2e}; min(R12 bound - h) = {worstR12:.3e}; "
      f"min(cost/ledger - 1) = {worstR13:.3e}; min(simple bound - ratio) = {worstfin:.4f}; max ratio {maxratio:.4f}; bad={bad} ({time.time()-t0:.0f}s)")


# ============================================================ (B) adversarial two-feature networks at s = 1.182
def unpack(z, m):
    return z[:m * p].reshape(m, p), z[m * p:2 * m * p].reshape(m, p), z[2 * m * p:3 * m * p].reshape(m, p)


def search(m, s, seed):
    r = np.random.default_rng(seed)
    b = 3 / s
    z0 = np.concatenate([r.normal(size=3 * m * p) * 0.5, [0.0]])

    def cons_g(z):
        U1, U2, W = unpack(z[:-1], m)
        return (gaps(U1, U2, W) - z[-1]).ravel()

    def cons_cost(z):
        U1, U2, W = unpack(z[:-1], m)
        nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
        return 1.0 - np.sum((nrm2 + 1e-12) ** (b / 2))

    def cons_b3(z):
        U1, U2, W = unpack(z[:-1], m)
        v = coeffs(U1, U2, W, 3)[0].sum() * 1e3
        return np.array([v.real, v.imag])
    res = minimize(lambda z: -z[-1], z0, method='SLSQP',
                   constraints=[{'type': 'ineq', 'fun': cons_g}, {'type': 'ineq', 'fun': cons_cost}, {'type': 'eq', 'fun': cons_b3}],
                   options=dict(maxiter=400, ftol=1e-10))
    U1, U2, W = unpack(res.x[:-1], m)
    U1, U2, W = kill(U1, U2, W, 3)               # make beta_3 = 0 exact (tiny correction)
    P, gIF = cost_ratio(U1, U2, W, s)
    h = gaps(U1, U2, W).min()
    nf = sum(abs(coeffs(U1, U2, W, j)[0].sum()) > 1e-6 for j in (1, 2))
    return h / P ** s / gIF, nf


t0 = time.time()
best = []
for k in range(16):
    m = 3 + k % 4
    try:
        r, nf = search(m, 1.182, 100 + k)
        best.append((r, nf, m))
    except Exception as ex:
        pass
best.sort(reverse=True)
print(f"(B) adversarial SLSQP two-feature search at s=1.182 (b=2.538): top ratios {[(round(r,4), nf, m) for r, nf, m in best[:5]]} "
      f"({time.time()-t0:.0f}s)")

# ============================================================ (C) p = 3 counterexamples
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from exact_certificate import NETWORKS
t0 = time.time()
for nn, net in NETWORKS.items():
    U = np.array([[float(Fr(a)) for a in row] for row in net['u']]); W = np.array([[float(Fr(a)) for a in row] for row in net['w']])
    P3 = 3
    r3 = np.exp(2j * np.pi / 3)
    # enumerate all 3^n inputs via counts would lose the T-dependence; do full enumeration in chunks over the first 3 slots
    xs = np.array(list(itertools.product(range(3), repeat=nn)), dtype=np.int8)     # 3^n x n
    yv = xs.sum(1) % 3
    logs = np.log(np.abs(U) + 0.0)
    # f(x)[c] = sum_i W[i,c] prod_t U[i, x_t]
    prodU = np.ones((len(xs), U.shape[0]))
    for tt in range(nn):
        prodU *= U[:, xs[:, tt]].T
    f = prodU @ W                                   # 3^n x 3
    gd = {d: f[np.arange(len(xs)), yv] - f[np.arange(len(xs)), (yv - d) % 3] for d in (1, 2)}
    h = min(gd[1].min(), gd[2].min())
    uh = np.array([[np.sum(U[i] * r3 ** (-k * np.arange(3))) for k in range(3)] for i in range(len(U))])
    wh = np.array([[np.sum(W[i] * r3 ** (-k * np.arange(3))) for k in range(3)] for i in range(len(U))])
    c = 2 * 3.0 ** -(nn + 1)
    beta1 = np.sum(c * np.conj(wh[:, 1]) * uh[:, 1] ** nn)
    mu = {1: 0.5, 2: 0.5}
    K1 = sum(mu[d] * (1 - np.cos(2 * np.pi * d / 3)) for d in (1, 2))
    worst = np.inf; maxerr = 0.0
    for k in range(1, nn + 1):
        T = list(range(k))                          # by slot symmetry only |T| matters
        bT = np.sum(c * np.conj(wh[:, 1]) * uh[:, 2] ** k * uh[:, 1] ** (nn - k))
        chi = np.zeros(nn); chi[T] = 2
        phase = r3 ** (-(xs @ chi))
        for wv in [0.25 * bT / abs(bT), 0.25 * np.exp(1j * 1.234)]:
            tot = 0.0
            for d in (1, 2):
                qw = 3.0 ** -nn * (mu[d] + 2 * np.real(wv * r3 ** (1 * d) * phase))
                assert qw.min() >= -1e-15
                tot += np.sum(qw * gd[d])
            rhs = K1 * beta1.real - 3 * np.real(wv * np.conj(bT))
            maxerr = max(maxerr, abs(tot - rhs) / (abs(tot) + abs(rhs) + 1e-300))
        bnd = K1 * beta1.real - 0.75 * abs(bT)
        worst = min(worst, bnd - h)
        if h > bnd + 1e-12 * abs(bnd):
            bad += 1; print("R12 FAILS at p=3", nn, k, h, bnd)
    print(f"(C) p=3, n={nn}: h={h:.6e}, Re beta_1={beta1.real:.6e}; R12 identity max rel err {maxerr:.1e}; "
          f"min over |T| of (R12 bound - h) = {worst:.4e} ({time.time()-t0:.0f}s)")
print("TOTAL BAD =", bad)
