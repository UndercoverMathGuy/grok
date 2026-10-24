"""hr6: spot-check of the (5,2) certificate machinery with INDEPENDENT evaluation.
1. Generate an HS certificate with the authors' SDP (hs2.elim_ratio) at a box centre near the worst region.
2. Re-evaluate it with my own code: q on the 25x4 grid (q>=0, sum 1), C_j, Fourier transforms, a_0, a_T, n, and the
   bound sqrt(a_+^2/F^2 + 12 n^2)/Gamma; compare with cert_hs.cert_data / eval_cells.
3. Rotate the certificate (torus action) and test it on actual NETWORKS whose own data are rotations of the box point:
   h <= a + R n, and gamma/gamma_IF <= the certified number.
4. Compare the rigorous box payment F (pay_adapt) with my own grid estimate of the circle payment (must be F <= mine)."""
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52')
from hs2 import elim_ratio
from cert_hs import cert_data, eval_cells
from pay_adapt import payment
from hr_net import Net, B_p

rho = np.exp(2j * np.pi / 5)
B5 = B_p(5, 3)
Gam = 2.5 * 2 ** -1.5
X = np.array(list(itertools.product(range(5), repeat=2)))
E = [(a, b) for a in range(3) for b in range(3)]
rng = np.random.default_rng(99)


# ---- helpers copied from hr1_ledgers / hr2_payment (to avoid re-running their tests on import)
def S_of(P, q, r):
    roots = np.roots([1, 0, -P, -q]).real
    return np.sum(np.maximum(1 + roots, 0) ** r)


def lowK(P):
    return max(P - 1, -2 * (P / 3) ** 1.5)


def upK(P):
    return 2 * (P / 3) ** 1.5


def Smin_closed(Pbar, qlow, r):
    Pbar = min(Pbar, 3.0)
    if qlow > upK(Pbar) + 1e-15:
        return np.inf
    return S_of(Pbar, max(qlow, lowK(Pbar)), r)


def P1fun(beta, s):
    return 3 ** (1 - s) * np.sum(np.abs(beta) ** (1 / s)) ** s


def circle_payment(Am, mB, mC, mD, phipay, s, nL=160, npsi=181):
    r = 1 / s
    Lmin = max(Am, mB, mC, mD)
    best = -np.inf
    for (mX, mY, mZ) in [(mD, mB, mC), (mC, mB, mD), (mB, mC, mD)]:
        cur = 3 ** (1 - s) * 2.5 * Lmin
        for t in np.concatenate([np.linspace(0, 0.05, 40), np.linspace(0.05, 1.5, nL)]):
            L = Lmin * (1 + t); rh = np.sqrt(max(2 * L * (L - Am), 0))
            for psi in np.linspace(-np.pi, np.pi, npsi):
                Pb = (mY ** 2 + mZ ** 2 + mX ** 2 + rh ** 2 + 2 * mX * rh * np.cos(psi - phipay)) / L ** 2
                q = 2 * mY * mZ * (mX * np.cos(phipay) + rh * np.cos(psi)) / L ** 3
                S = Smin_closed(Pb, q, r)
                if np.isfinite(S):
                    cur = min(cur, L * (S / 3) ** s)
        best = max(best, cur)
    return best


def kill_beta2(net):
    b2 = net.beta(2)
    a = np.arange(5)
    u = np.cos(2 * np.pi * 2 * a / 5)
    mag = abs(b2) / (2 / 125 * (5 / 2) ** 3)
    psi = -(np.angle(-b2))
    w = mag * np.cos(2 * np.pi * 2 * a / 5 + psi)
    U = np.concatenate([net.U, np.stack([u, u])[None]], axis=0)
    W = np.concatenate([net.W, w[None]], axis=0)
    return Net(5, 2, U, W)


def ledger52(net, s):
    e = B5 * net.norms2() ** 1.5
    bi = net.beta_i(1)
    rho_i = np.sqrt(np.maximum(e ** 2 - np.abs(bi) ** 2, 0))
    return dict(R=2 * rho_i.sum())


def my_eval(Gs, A, B, C, D, F):
    th = 2 * np.pi * 2 * X / 5                                   # theta_t = 2 pi 2 x_t /5
    V = np.exp(1j * th @ np.array(E).T)                          # 25 x 9: v_e(theta(x)) = e^{i<e,theta>}
    q = np.stack([np.real(np.einsum('xe,ef,xf->x', V.conj(), G, V)) for G in Gs], 1)   # 25 x 4
    mass = q.sum(); q = q / mass
    ds = np.arange(1, 5)
    c1 = 1 - rho ** (-ds); c2 = 1 - rho ** (-2 * ds)
    C1 = q @ c1; C2 = q @ c2
    Ch = lambda Cv, psi: np.sum(Cv * rho ** (-(X @ np.array(psi)))) / 25
    a0 = 25 * Ch(C1, (0, 0))
    aT = [25 * Ch(C1, ((-2) % 5, 0)), 25 * Ch(C1, (0, (-2) % 5)), 25 * Ch(C1, ((-2) % 5, (-2) % 5))]
    own1 = {(0, 0), (2, 0), (0, 2), (2, 2)}
    n2 = 0
    for chi in itertools.product(range(5), repeat=2):
        mchi = tuple((-np.array(chi)) % 5)
        if chi not in own1:
            n2 += abs(25 * Ch(C1, mchi)) ** 2
        if chi != (0, 0):
            n2 += abs(25 * Ch(C2, mchi)) ** 2
    nn = np.sqrt(n2)
    a = (np.conj(A) * a0).real + (np.conj(B) * aT[0]).real + (np.conj(C) * aT[1]).real + (np.conj(D) * aT[2]).real
    bound = np.sqrt(max(a, 0) ** 2 / F ** 2 + 12 * nn ** 2) / Gam
    return dict(q=q, qmin=q.min(), a0=a0, aT=aT, n=nn, a=a, bound=bound)


def net_with_data(alpha, m1, m2, m12, phi, extra):
    """build a NETWORK with beta_2 = 0 whose own data are (A,B,C,D) up to scale: use the identity-based construction:
    IF-style atoms are hard to target exactly, so we instead take random networks and only use their own data."""
    raise NotImplementedError


# ---- box near the worst Region-II boxes reported in the logs
boxes = [((0.2, 0.2, 0.8), (0.3, 0.4, 1.0), (-1.1781, -0.7854)), ((0.6, 0.6, 0.6), (0.7, 0.7, 0.8), (0.0, 0.3927)),
         ((0.9, 0.9, 0.9), (1.0, 1.0, 1.0), (-0.3927, 0.0))]
for (mlo, mhi, (plo, phi_)) in boxes:
    mlo = np.array(mlo); mhi = np.array(mhi)
    F = payment(mlo, mhi, plo, phi_)
    mc = 0.5 * (mlo + mhi); pc = 0.5 * (plo + phi_)
    mine = circle_payment(1.0, mc[0], mc[1], mc[2], pc, 1.5, nL=80, npsi=121)
    print(f"box {mlo}-{mhi} phi_pay [{plo},{phi_}]: rigorous box F = {F:.4f};  my grid circle payment at centre = {mine:.4f} (need F <= mine)")
    for al in (0.0, 0.3, 0.6):
        val, Gs = elim_ratio(al, mc[0], mc[1], mc[2], pc - al, F, deg=2, ret=True)
        cd = cert_data(Gs)
        A = np.exp(1j * al); B, C, D = mc[0], mc[1], mc[2] * np.exp(1j * (pc - al))
        me = my_eval(Gs, A, B, C, D, F)
        theirs = eval_cells(cd, np.array([al]), np.array([al]), mc[None, :], mc[None, :], np.array([pc - al]),
                            np.array([pc - al]), np.array([F]))[0]
        print(f"   alpha={al}: SDP {val:.5f}; my eval bound {me['bound']:.6f} (qmin {me['qmin']:.2e}); cert_hs rigorous {theirs:.6f};"
              f" |a0 diff| {abs(me['a0'] - complex(cd['a0'][0].lo, cd['a0'][1].lo)):.1e}; n mine {me['n']:.6f} vs [{float(cd['n'].lo):.6f},{float(cd['n'].hi):.6f}]")

# ---- end-to-end on networks: random networks with beta_2 killed; normalise own data; certificate at its own data
worst = -np.inf; cnt = 0
for trial in range(25):
    net = kill_beta2(Net(5, 2, rng.normal(size=(3, 2, 5)), rng.normal(size=(3, 5))))
    A = net.beta(1); B = net.beta(1, (0,)); C = net.beta(1, (1,)); D = net.beta(1, (0, 1))
    if A.real <= 0:
        continue
    sc = abs(A)
    al = np.angle(A); Bn, Cn, Dn = B / sc, C / sc, D / sc
    # torus-normalise: B, C real >= 0
    t1, t2 = -np.angle(Bn), -np.angle(Cn)
    m1, m2, m12 = abs(Bn), abs(Cn), abs(Dn); ph = np.angle(Dn * np.exp(1j * (t1 + t2)))
    if max(m1, m2, m12) > 1.4:
        continue
    beta_atoms = net.beta_i(1)
    P1 = P1fun(beta_atoms / sc, 1.5)
    F = min(P1, circle_payment(1.0, m1, m2, m12, ph + al, 1.5, nL=40, npsi=61))   # a valid lower bound (<= P1)
    val, Gs = elim_ratio(al, m1, m2, m12, ph, F, deg=2, ret=True)
    # rotate certificate back: G -> D_t^* G D_t with D_t = diag(e^{-i<e,t>}) for t with B e^{it1} real etc.
    tt = -np.array([t1, t2])          # certificate rotation angle = +arg B, +arg C (see derivation)
    Dt = np.diag(np.exp(-1j * (np.array(E) @ tt)))
    Gr = [Dt.conj().T @ G @ Dt for G in Gs]
    # my evaluation at the ORIGINAL (unrotated) data with the rotated certificate
    me_rot = my_eval(Gr, np.exp(1j * al), Bn, Cn, Dn, F)
    me_nrm = my_eval(Gs, np.exp(1j * al), m1, m2, m12 * np.exp(1j * ph), F)
    # actual network test: h <= sum q g <= a + R n  (unnormalised data)
    L = ledger52(net, 1.5)
    G = net.gaps()
    sqg_rot = (me_rot['q'] * G).sum()
    a_un = me_rot['a'] * sc
    worst = max(worst, sqg_rot - (a_un + L['R'] * me_rot['n']))
    gam = net.h() / net.cost(2) ** 1.5; gIF = 3 ** -0.5 * B5 * Gam
    cnt += 1
    print(f"net {trial}: rotated bound {me_rot['bound']:.5f} vs normal-form bound {me_nrm['bound']:.5f}; gamma/gIF {gam/gIF:.4f};"
          f" sum q g - (a+Rn) = {sqg_rot - (a_un + L['R']*me_rot['n']):.3e}")
print(f"end-to-end: {cnt} networks, max [sum q g - (a + R n)] = {worst:.3e} (<=0)")
