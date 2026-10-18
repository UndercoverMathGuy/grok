"""Numerical sanity checks of every lemma used in the (5,2) proof, on random and structured networks.
  L_B  mode budget per neuron:  sum_{non-own} |m|^2 <= 4 (e^2 - |beta_1|^2)
  L_C  Minkowski cost:          3^{1-s} B_5 P^s >= sqrt(P_1^2 + 3^{2-2s} (sum rho_i)^2)       (s = 1.5, 1.25, 1.1)
  L_D  certificate bound:       h <= Re(conj A a0) + sum Re(conj beta_T a_T) + R n     (random HS certificates, any phases)
  L_F  twisted payments:        P_1 >= L (sum nu^r / 3)^s for the three chain matrices; |X' - alpha X| <= sqrt(2L(L-|A|))
  L_S  spectral monotonicity:   S(P,q) decreasing in P, increasing in q on the feasible set
  L_A  dilation a -> 3a swaps features 1 and 2 and preserves the margin
"""
import numpy as np
from p52lib import *

rng = np.random.default_rng(12345)
s0 = 1.5


def kill_beta2(U, W):
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    v = Uh[:, 0, 2] * Uh[:, 1, 2]
    cc = np.conj(np.sum(np.conj(Wh[:, 2]) * v) / np.sum(np.abs(v) ** 2))
    Wh[:, 2] = Wh[:, 2] - cc * v
    Wh[:, 3] = np.conj(Wh[:, 2])
    return np.real(np.fft.ifft(Wh, axis=1))


def modes(u1, u2, w):
    """m(j,k1,k2) = (2/125) what(j) uhat1(k1) uhat2(k2), j in {1,2}"""
    a = np.fft.fft(u1); b = np.fft.fft(u2); c = np.fft.fft(w)
    return {j: (2 / 125) * c[j] * np.outer(a, b) for j in (1, 2)}


def own_data(U, W):
    Phi = fields(U, W)
    A = np.conj(fcoef(Phi[1], (0, 0)))
    bT = [np.conj(fcoef(Phi[1], chi)) for chi in OWN1]
    return A, bT, Phi


def make_net(kind, m=5):
    if kind == 'random':
        U = rng.normal(size=(m, 2, 5)); W = rng.normal(size=(m, 5))
    elif kind == 'IFpert':
        a = np.arange(5)
        U = []; W = []
        for l in range(3):
            al = np.pi * l / 3
            U.append([np.cos(2 * np.pi * a / 5 + al), np.cos(2 * np.pi * a / 5 + al)])
            W.append(np.cos(2 * np.pi * a / 5 + 2 * al))
        U = np.array(U) + 0.15 * rng.normal(size=(3, 2, 5)); W = np.array(W) + 0.15 * rng.normal(size=(3, 5))
    else:   # single-atom-like plus compensators
        a = np.arange(5)
        U = np.array([[np.cos(2 * np.pi * a / 5 + 0.3), np.cos(2 * np.pi * a / 5 - 0.7)]]) + 0.05 * rng.normal(size=(1, 2, 5))
        W = np.array([np.cos(2 * np.pi * a / 5 + 0.1)]) + 0.3 * rng.normal(size=(1, 5))
        U = np.concatenate([U, 0.3 * rng.normal(size=(3, 2, 5))]); W = np.concatenate([W, 0.3 * rng.normal(size=(3, 5))])
    return U, kill_beta2(U, W)


def check_LB_LC(U, W, s):
    r = 1 / s
    worstB = -np.inf; P = 0; Q = 0; Rsum = 0
    for i in range(U.shape[0]):
        mm = modes(U[i, 0], U[i, 1], W[i])
        nrm = np.sqrt(np.sum(U[i] ** 2) + np.sum(W[i] ** 2))
        e = B5 * nrm ** 3
        beta = abs(mm[1][1, 1])                 # |beta_i1| = |m(1; 1,1)|
        own = sum(abs(mm[1][k1, k2]) ** 2 for k1 in (1, 4) for k2 in (1, 4))
        tot = sum(np.sum(np.abs(mm[j]) ** 2) for j in (1, 2))
        worstB = max(worstB, (tot - own) - 4 * (e ** 2 - beta ** 2))
        P += nrm ** (3 * r); Q += beta ** r; Rsum += np.sqrt(max(e ** 2 - beta ** 2, 0))
    lhs = 3 ** (1 - s) * B5 * P ** s
    rhs = np.sqrt((3 ** (1 - s) * Q ** s) ** 2 + 3 ** (2 - 2 * s) * Rsum ** 2)
    return worstB, rhs - lhs, 2 * Rsum, 3 ** (1 - s) * Q ** s


def rand_cert():
    Gs = []
    for d in range(4):
        X = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
        Gs.append(X @ X.conj().T)
    tot = sum(np.trace(G).real for G in Gs)
    return [G / tot for G in Gs]


E2 = [(a, b) for a in range(3) for b in range(3)]


def cert_q(Gs):
    """grid density q(x,delta) of an HS certificate"""
    q = np.zeros((4, 5, 5))
    for di in range(4):
        for x1 in range(5):
            for x2 in range(5):
                th = (2 * np.pi * 2 * x1 / 5, 2 * np.pi * 2 * x2 / 5)
                v = np.array([np.exp(1j * (e[0] * th[0] + e[1] * th[1])) for e in E2])
                q[di, x1, x2] = np.real(v.conj() @ Gs[di] @ v) / 25
    return q


def bound_from_cert(Gs, A, bT, R):
    from cert_hs import cert_data
    cd = cert_data(Gs)
    a0 = complex(cd['a0'][0].lo, cd['a0'][1].lo)
    aT = [complex(x[0].lo, x[1].lo) for x in cd['aT']]
    n = float(cd['n'].hi)
    return np.real(np.conj(A) * a0) + sum(np.real(np.conj(b) * a) for b, a in zip(bT, aT)) + R * n


if __name__ == '__main__':
    out = []
    worst = dict(LB=-np.inf, LC=-np.inf, LD=-np.inf, LDdirect=-np.inf)
    for t in range(60):
        kind = ['random', 'IFpert', 'atom'][t % 3]
        U, W = make_net(kind)
        for s in (1.5, 1.25, 1.1):
            wb, wc, R, P1 = check_LB_LC(U, W, s)
            worst['LB'] = max(worst['LB'], wb); worst['LC'] = max(worst['LC'], wc)
        _, _, R, P1 = check_LB_LC(U, W, 1.5)
        G = gaps(U, W); h = G.min()
        A, bT, Phi = own_data(U, W)
        # direct check: h <= sum q g for the grid density
        Gs = rand_cert()
        q = cert_q(Gs)
        sq = sum(np.sum(q[di] * G[:, :, di]) for di in range(4))
        worst['LDdirect'] = max(worst['LDdirect'], h - sq)
        bnd = bound_from_cert(Gs, A, bT, R)
        worst['LD'] = max(worst['LD'], sq - bnd)       # sum q g <= bound (Cauchy-Schwarz part)
    print("L_B  max[(non-own mass) - 4(e^2-|beta|^2)]   =", worst['LB'], "(must be <= 0)")
    print("L_C  max[sqrt(P1^2+cR^2) - 3^{1-s}B5 P^s]    =", worst['LC'], "(must be <= 0)")
    print("L_D  max[h - sum q g]                        =", worst['LDdirect'], "(must be <= 0)")
    print("L_D  max[sum q g - certificate bound]        =", worst['LD'], "(must be <= 0)")
    # L_F: twisted payments on random atom systems
    from tw2 import S
    wF = -np.inf; wdisk = -np.inf
    for t in range(3000):
        k = rng.integers(1, 7)
        rho_ = rng.exponential(size=k); p_ = np.exp(1j * rng.uniform(0, 2 * np.pi, k))
        xi = np.exp(1j * rng.uniform(0, 2 * np.pi, k)); eta = np.exp(1j * rng.uniform(0, 2 * np.pi, k))
        if t % 4 == 0:
            p_ = np.exp(1j * rng.normal(0, 0.2, k))
        beta = rho_ * p_
        A = beta.sum(); B = (beta * xi).sum(); C = (beta * eta).sum(); D = (beta * xi * eta).sum()
        L = rho_.sum(); al = A / abs(A)
        for s in (1.5, 1.2):
            r = 1 / s
            P1 = 3 ** (1 - s) * np.sum(rho_ ** r) ** s
            for u in [np.stack([np.ones(k), p_ * xi, p_ ** 2 * xi * eta], 1),
                      np.stack([np.ones(k), p_ * xi, p_ * xi * eta], 1),
                      np.stack([np.ones(k), p_ * eta, p_ * xi * eta], 1)]:
                H = (u.T * rho_) @ u.conj()
                nu = np.clip(np.linalg.eigvalsh(H / L), 0, None)
                wF = max(wF, L * (np.sum(nu ** r) / 3) ** s - P1)
        Dp = np.sum(rho_ * p_ ** 2 * xi * eta); Cp = np.sum(rho_ * eta); Bp = np.sum(rho_ * xi)
        rad = np.sqrt(2 * L * (L - abs(A)))
        wdisk = max(wdisk, abs(Dp - al * D) - rad, abs(Cp - np.conj(al) * C) - rad, abs(Bp - np.conj(al) * B) - rad)
    print("L_F  max[twisted bound - P_1]                =", wF, "(must be <= 0)")
    print("L_F  max[|X' - alpha^{+-1} X| - sqrt(2L(L-|A|))] =", wdisk, "(must be <= 0)")
    # L_S monotonicity of S(P,q) on the feasible set (finite differences)
    bad = 0; tot = 0
    for t in range(4000):
        P = rng.uniform(0.01, 3); dh = 2 * (P / 3) ** 1.5; lo = max(P - 1, -dh)
        if lo >= dh:
            continue
        q = rng.uniform(lo, dh)
        for r in (2 / 3, 0.8):
            h_ = 1e-6
            if q + h_ <= dh:
                tot += 1; bad += S(P, q + h_, r) < S(P, q, r) - 1e-12
            if P + h_ <= 3 and q >= (P + h_) - 1 and q >= -2 * ((P + h_) / 3) ** 1.5 and q <= 2 * ((P + h_) / 3) ** 1.5:
                tot += 1; bad += S(P + h_, q, r) > S(P, q, r) + 1e-12
    print(f"L_S  monotonicity violations: {bad} / {tot}")
    # L_A dilation
    U, W = make_net('random')
    Ud = U[:, :, (3 * np.arange(5)) % 5]; Wd = W[:, (3 * np.arange(5)) % 5]
    b = betas(U, W); bd = betas(Ud, Wd)
    print("L_A  |beta'_1 - beta_2| =", abs(np.sum(bd[1]) - np.sum(b[2])), " |beta'_2| - |beta_1| =",
          abs(np.sum(bd[2])) - abs(np.sum(b[1])), " margin diff =", gaps(Ud, Wd).min() - gaps(U, W).min())
