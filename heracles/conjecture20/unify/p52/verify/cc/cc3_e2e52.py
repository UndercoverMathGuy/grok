"""Referee end-to-end check of the (5,2) single-feature certificate (R9) on explicit networks.

For each network theta at (5,2) with beta_2 = 0 (random with DC and multi-frequency content, IF blocks, perturbations,
single atom + compensators, and locally optimised adversaries):
  1. direct gamma/gamma_IF at s = 3/2 (b = 2) and s = 5/4 (b = 2.4);
  2. own data (A, B, C, D) from the TABLE (cc_net), mode mass R = 2 sum rho_i, check ||phi_nonown||_2 <= R (R2-R3);
  3. normal form (scale, conjugation, slot swap, torus angles) -> their rigorous payment F (pay_adapt) at the point,
     check P_1(theta) >= F;
  4. their SDP certificate (hs2.elim_ratio) + their rigorisation (cert_hs.cert_data) + their bound (eval_cells);
  5. MY reconstruction of the probability q(x, delta) from the rigorised Gram matrices, rotated by the torus angles and
     mapped back through swap / conjugation to the ORIGINAL network; I check q >= 0, sum q = 1, h <= sum q g,
     the Fourier identity sum q g = Re sum_{j,chi} phi_j(chi) Cq_j(chi), the ledger sum q g <= a_own + R n, and that
     my a_own, n agree with their interval enclosures;
  6. ratio <= min(trivial, certificate bound) <= 0.995.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import minimize
from cc_net import Net, B_p, gammaIF52, cancel_feature, pure_neuron

p = 5; rho = np.exp(2j * np.pi / p)
KA = 1 - np.cos(2 * np.pi / 5)
GAM = 2.5 * 2 ** -1.5
B5 = B_p(5)
rng = np.random.default_rng(2026)


def stats(net, s):
    b = 3 / s
    return net.h() / net.cost(b) ** s / gammaIF52(s)


def make_random(m, dc=True):
    U1 = rng.standard_normal((m, p)); U2 = rng.standard_normal((m, p)); W = rng.standard_normal((m, p))
    if not dc:
        U1 -= U1.mean(1, keepdims=True); U2 -= U2.mean(1, keepdims=True)
    return cancel_feature(Net(p, U1, U2, W), 2)


def if_block(scale=1.0):
    rows = []
    for i in range(3):
        ph1 = -np.pi * i / 3; ph2 = -np.pi * i / 3
        rows.append(pure_neuron(p, 1, scale, scale, scale, ph1, ph2, ph1 + ph2))
    U1, U2, W = map(np.array, zip(*rows))
    return Net(p, U1, U2, W)


def add(net, other):
    return Net(p, np.vstack([net.U1, other.U1]), np.vstack([net.U2, other.U2]), np.vstack([net.W, other.W]))


def perturbed_if(eps):
    base = if_block()
    U1 = base.U1 + eps * rng.standard_normal(base.U1.shape); U2 = base.U2 + eps * rng.standard_normal(base.U2.shape)
    W = base.W + eps * rng.standard_normal(base.W.shape)
    net = Net(p, U1, U2, W)
    extra = Net(p, eps * rng.standard_normal((2, p)), eps * rng.standard_normal((2, p)), eps * rng.standard_normal((2, p)))
    return cancel_feature(add(net, extra), 2)


def single_atom_plus(eps):
    u1, u2, w = pure_neuron(p, 1, 1, 1, 1, 0.3, -0.2, 0.1)
    net = Net(p, [u1], [u2], [w])
    extra = Net(p, eps * rng.standard_normal((3, p)), eps * rng.standard_normal((3, p)), eps * rng.standard_normal((3, p)))
    return cancel_feature(add(net, extra), 2)


def optimise(m, s, starts=4):
    """locally maximise h / P^s with beta_2 = 0 (SLSQP; variables neurons + t, cost normalised to 1)"""
    b = 3 / s; best = None; bestv = -np.inf
    for st in range(starts):
        if st % 2 == 0:
            net0 = perturbed_if(0.3)
        else:
            net0 = make_random(m)
        k = net0.U1.shape[0]
        z0 = np.concatenate([net0.U1.ravel(), net0.U2.ravel(), net0.W.ravel()])
        z0 = z0 / net0.cost(b) ** (1 / b)

        def unpack(z):
            return Net(p, z[:k * p].reshape(k, p), z[k * p:2 * k * p].reshape(k, p), z[2 * k * p:].reshape(k, p))

        def cons_g(zz):
            return unpack(zz[:-1]).gaps().ravel() - zz[-1]

        def cons_eq(zz):
            nt = unpack(zz[:-1]); b2 = nt.neuron_beta(2).sum()
            return np.array([nt.cost(b) - 1.0, b2.real * 1e3, b2.imag * 1e3])
        zz0 = np.concatenate([z0, [unpack(z0).h()]])
        res = minimize(lambda zz: -zz[-1], zz0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': cons_g}, {'type': 'eq', 'fun': cons_eq}],
                       options=dict(maxiter=400, ftol=1e-12))
        nt = cancel_feature(unpack(res.x[:-1]), 2)
        v = stats(nt, s)
        if v > bestv:
            bestv, best = v, nt
    return best


# ---------------------------------------------------------------- certificate machinery (theirs, under test)
from pay_adapt import payment
from hs2 import elim_ratio
from cert_hs import cert_data, eval_cells
from iv import dn


def my_q_from_G(Gs, t, mass_only=False):
    """MY reconstruction: q(x, d) = v(theta(x) - t)^* G~_d v(theta(x) - t) / (25 * mass), G~ = U diag(lam_+) U^*"""
    E = [(a, b) for a in range(3) for b in range(3)]
    Gt = []
    for G in Gs:
        G = (G + G.conj().T) / 2
        lam, U = np.linalg.eigh(G)
        Gt.append((U * np.maximum(lam, 0)) @ U.conj().T)
    mass = sum(np.trace(G).real for G in Gt)
    q = np.zeros((p, p, 4))
    for x1 in range(p):
        for x2 in range(p):
            th = 2 * np.pi * 2 * np.array([x1, x2]) / p - np.asarray(t)
            v = np.array([np.exp(1j * (e[0] * th[0] + e[1] * th[1])) for e in E])
            for d in range(4):
                q[x1, x2, d] = (v.conj() @ Gt[d] @ v).real
    return q / (25 * mass)


def ledger_from_q(q, net):
    """direct Fourier pairing of q with the network fields"""
    c = {j: np.array([1 - rho ** (-j * d) for d in range(1, 5)]) for j in (1, 2)}
    phi = net.phi()
    # Cq_j(chi) = sum_{x,d} q(x,d) c_j(d) rho^{<chi,x>}
    Cq = {}
    for j in (1, 2):
        Cx = q @ c[j]           # p x p
        Cq[j] = np.fft.ifft2(Cx) * p * p     # sum_x Cx(x) rho^{+<chi,x>}
    S_id = sum((phi[j] * Cq[j]).sum() for j in (1, 2)).real
    own_idx = [(0, 0), (2, 0), (0, 2), (2, 2)]
    A, B, C, D = net.own(1)
    a0 = Cq[1][0, 0]; aT = [Cq[1][2, 0], Cq[1][0, 2], Cq[1][2, 2]]
    a_own = (np.conj(A) * a0 + np.conj(B) * aT[0] + np.conj(C) * aT[1] + np.conj(D) * aT[2]).real
    mask1 = np.ones((p, p), bool)
    for (i, j) in own_idx:
        mask1[i, j] = False
    mask2 = np.ones((p, p), bool); mask2[0, 0] = False
    nn = np.sqrt(np.sum(np.abs(Cq[1][mask1]) ** 2) + np.sum(np.abs(Cq[2][mask2]) ** 2))
    phin = np.sqrt(np.sum(np.abs(phi[1][mask1]) ** 2) + np.sum(np.abs(phi[2][mask2]) ** 2))
    return S_id, a_own, nn, phin, a0, aT


def check(net, label, s_list=(1.5, 1.25)):
    A, B, C, D = net.own(1)
    b2 = net.own(2)[0]
    assert abs(b2) < 1e-10 * max(1, abs(A)), b2
    bi = net.neuron_beta(1)
    assert abs(bi.sum() - A) < 1e-9 * max(1, abs(A))
    e = B5 * net.norms2() ** 1.5
    rho_i = np.sqrt(np.maximum(e ** 2 - np.abs(bi) ** 2, 0)); R = 2 * rho_i.sum()
    assert np.all(np.abs(bi) <= e * (1 + 1e-12))
    out = dict(label=label)
    for s in s_list:
        out[f'ratio{s}'] = stats(net, s)
    ratio = out['ratio1.5']
    # normal form
    lam = 1 / abs(A)
    A_, B_, C_, D_ = A * lam, B * lam, C * lam, D * lam
    conj = np.angle(A_) < 0
    if conj:
        A_, B_, C_, D_ = map(np.conj, (A_, B_, C_, D_))
    swap = abs(B_) < abs(C_)
    if swap:
        B_, C_ = C_, B_
    alpha = np.angle(A_)
    m = np.array([abs(B_), abs(C_), abs(D_)])
    out['m'] = m; out['alpha'] = alpha
    if alpha > np.pi / 2:
        out['case'] = 'ReA<=0'; out['ok'] = net.h() <= 1e-12; return out
    t = (np.angle(B_), np.angle(C_))
    phi = np.angle(D_) - t[0] - t[1]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    php = (alpha + phi + np.pi) % (2 * np.pi) - np.pi
    P1 = 3 ** -0.5 * np.sum(np.abs(bi * lam) ** (2 / 3)) ** 1.5
    if m.max() >= 1.4:
        F = m.max() / np.sqrt(3)
        out['case'] = 'large'; out['F'] = F; out['bound'] = KA * np.cos(alpha) / (GAM * F)
        out['ok'] = (P1 >= F * (1 - 1e-12)) and ratio <= out['bound'] + 1e-12 and out['bound'] <= 0.995
        return out
    F = payment(m, m, php, php)
    out['F'] = F; out['P1'] = P1
    triv = KA * np.cos(alpha) / (GAM * F)
    val, Gs = elim_ratio(alpha, m[0], m[1], m[2], phi, F, deg=2, ret=True)
    cd = cert_data(Gs)
    cert = eval_cells(cd, np.array([alpha]), np.array([alpha]), m[None, :], m[None, :], np.array([phi]),
                      np.array([phi]), np.array([F]))[0]
    out['triv'] = triv; out['cert'] = cert; out['bound'] = min(triv, cert)
    # my reconstruction of q in the ORIGINAL frame
    qn = my_q_from_G(Gs, t)                  # certificate for the conj/swapped network with original torus angles
    q = qn.copy()
    if swap:
        q = np.transpose(q, (1, 0, 2))
    if conj:
        idx = (-np.arange(p)) % p
        q = q[idx][:, idx][:, :, ::-1]       # q(x, d) = q'(-x, -d); d index 0..3 <-> d = 1..4, -d <-> 3-index
    g = net.gaps()
    S = float((q * g).sum())
    S_id, a_own, nn, phin, a0, aT = ledger_from_q(q, net)
    # compare with their enclosures (rotate their a_T by the torus angles; undo conj/swap on the data side)
    B0 = (np.cos(alpha) * cd['a0'][0] + np.sin(alpha) * cd['a0'][1] + m[0] * cd['aT'][0][0] + m[1] * cd['aT'][1][0]
          + m[2] * (np.cos(phi) * cd['aT'][2][0] + np.sin(phi) * cd['aT'][2][1]))
    out.update(qmin=q.min(), qsum=q.sum(), S=S, h=net.h(), S_id_err=abs(S - S_id), a_own=a_own * lam,
               B0lo=float(B0.lo), B0hi=float(B0.hi), n_mine=nn, n_lo=float(cd['n'].lo), n_hi=float(cd['n'].hi),
               R=R, phin=phin)
    out['ok'] = bool(q.min() >= -1e-14 and abs(q.sum() - 1) < 1e-12 and net.h() <= S + 1e-12
                     and abs(S - S_id) < 1e-9 and S <= a_own + R * nn + 1e-9 and phin <= R * (1 + 1e-9) + 1e-12
                     and P1 >= F * (1 - 1e-12) and ratio <= out['bound'] + 1e-9 and out['bound'] <= 0.995
                     and B0.lo - 1e-9 <= a_own * lam <= B0.hi + 1e-9 and cd['n'].lo - 1e-9 <= nn <= cd['n'].hi + 1e-9)
    return out


if __name__ == '__main__':
    t0 = time.time()
    nets = []
    for k in range(10):
        nets.append((make_random(int(rng.integers(1, 7)), dc=True), 'random+DC'))
    for k in range(4):
        nets.append((make_random(int(rng.integers(1, 7)), dc=False), 'random'))
    nets.append((if_block(), 'IF block'))
    for eps in (0.02, 0.1, 0.3):
        nets.append((perturbed_if(eps), f'IF+pert{eps}'))
    for eps in (0.05, 0.3, 1.0):
        nets.append((single_atom_plus(eps), f'atom+comp{eps}'))
    for (m, s) in [(4, 1.5), (6, 1.5), (5, 1.25), (6, 1.0)]:
        nets.append((optimise(m, s), f'opt m={m} s={s}'))
    nbad = 0; worst = 0
    for net, lab in nets:
        o = check(net, lab)
        worst = max(worst, o['ratio1.5'])
        if not o['ok']:
            nbad += 1
        extra = ''
        if 'cert' in o:
            extra = (f" F={o['F']:.4f} P1={o['P1']:.4f} triv={o['triv']:.4f} cert={o['cert']:.4f} qmin={o['qmin']:.1e}"
                     f" |S-id|={o['S_id_err']:.1e} h={o['h']:.4f}<=S={o['S']:.4f} a_own={o['a_own']:.5f} in"
                     f" [{o['B0lo']:.5f},{o['B0hi']:.5f}] n={o['n_mine']:.5f} in [{o['n_lo']:.5f},{o['n_hi']:.5f}]"
                     f" phin={o['phin']:.4f}<=R={o['R']:.4f}")
        elif 'bound' in o:
            extra = f" case={o['case']} bound={o['bound']:.4f}"
        else:
            extra = f" case={o.get('case')}"
        print(f"{lab:18s} ratio(b=2)={o['ratio1.5']:+.4f} ratio(b=2.4)={o['ratio1.25']:+.4f} m={np.round(o['m'],3)} "
              f"alpha={o['alpha']:.3f} ok={o['ok']}{extra}", flush=True)
    print(f"networks={len(nets)} failures={nbad} max ratio(b=2)={worst:.4f} time={time.time()-t0:.0f}s")
