"""End-to-end sanity checks at (7,2) for the certified bound (rig72) and the s-monotonicity lemma.
 E1  random two-feature networks (beta_3 = 0 enforced by a cancelling frequency-3 neuron), with DC, frequency-3
     inputs and shared neurons: gamma/gamma_IF <= B(tau, x1, x2; case of the network) at s in {1.1, 1.25}.
 E2  locally optimised two-feature networks (outputs without frequency 3; inputs unrestricted) at s = 1.25 and 1.5:
     report gamma/gamma_IF and the bound.
 E3  s-monotonicity: for fixed tests, B(s) is nondecreasing in s (random configurations).
"""
import sys
import numpy as np
from scipy.optimize import minimize
from n2lib import Grid, gaps, cost, atoms, own_data
import rig72

G = Grid(7)
rng = np.random.default_rng(3)
TS = ['1', '2', '12']


def gamma_ratio(U, W, s):
    b = 3 / s
    h = gaps(G, U, W).min()
    P = cost(U, W, b)
    gam = h / P ** s
    gIF = 3 ** (1 - s) * G.Bp * 3.5 * 3.0 ** (-s)
    return gam / gIF


def net_config(U, W):
    A, BT = own_data(G, U, W)
    A1, A2 = abs(A[0]), abs(A[1])
    x1 = np.max(np.abs(BT[0])) / A1; x2 = np.max(np.abs(BT[1])) / A2
    T1 = TS[int(np.argmax(np.abs(BT[0])))]; T2 = TS[int(np.argmax(np.abs(BT[1])))]
    return A1, A2, x1, x2, T1, T2, abs(A[2])


MIRROR = {'1': '2', '2': '1', '12': '12'}


def bound_at(s, A1, A2, x1, x2, T1, T2, ledger='pat'):
    case = (T1, T2)
    if case not in rig72.CASES:
        case = (MIRROR[T1], MIRROR[T2])
    m = rig72.Model(case, ledger)
    tau = A2 / (A1 + A2)
    i1 = x1 > rig72.XMAX; i2 = x2 > rig72.XMAX
    mu3, w, v = m.solve(s, tau, min(x1, rig72.XMAX), min(x2, rig72.XMAX), i1, i2)
    cert = rig72.certify_test(case, mu3, w, ledger)
    return rig72.box_bound(cert, tau, tau, min(x1, rig72.XMAX), min(x1, rig72.XMAX), min(x2, rig72.XMAX),
                           min(x2, rig72.XMAX), i1, i2, s)


def cancel3(U, W):
    """append a frequency-3 neuron whose atom cancels beta_3"""
    a = np.arange(7)
    bet = atoms(G, U, W)[:, 2].sum()
    u = np.cos(2 * np.pi * 3 * a / 7)
    # atom of (u, u, w=c*cos(2pi 3a/7 + phi)): (2/343) conj(what(3)) uhat(3)^2 ; uhat(3) = 7/2
    # what(3) for w = c cos(.. + phi) = c (7/2) e^{i phi}; atom = (2/343) c (7/2)^3 e^{-i phi} = c e^{-i phi}/4
    c = 4 * abs(bet); phi = np.angle(-bet) * -1
    wv = c * np.cos(2 * np.pi * 3 * a / 7 + phi)
    U2 = np.concatenate([U, np.stack([u, u])[None]], axis=0); W2 = np.concatenate([W, wv[None]], axis=0)
    return U2, W2


worst = {}
if __name__ == '__main__':
    worst = {}
    for trial in range(24):
        m = int(rng.integers(2, 6))
        a = np.arange(7)
        U = np.zeros((m, 2, 7)); W = np.zeros((m, 7))
        for i in range(m):
            for t in range(2):
                for k in (1, 2, 3):
                    U[i, t] += rng.normal() * np.cos(2 * np.pi * k * a / 7 + rng.uniform(0, 6.3))
                U[i, t] += 0.5 * rng.normal()
            for k in (1, 2, 3):
                W[i] += rng.normal() * np.cos(2 * np.pi * k * a / 7 + rng.uniform(0, 6.3))
        U, W = cancel3(U, W)
        A1, A2, x1, x2, T1, T2, A3 = net_config(U, W)
        assert A3 < 1e-9, A3
        for s in (1.1, 1.25):
            g = gamma_ratio(U, W, s)
            B = bound_at(s, A1, A2, x1, x2, T1, T2)
            worst['E1 gamma/gIF - bound (must be <= 0)'] = max(worst.get('E1 gamma/gIF - bound (must be <= 0)', -9), g - B)
    print("E1 done:", worst, flush=True)

    # E2: local optimisation of two-feature networks (no output frequency 3)
    def unpack(v, m):
        U = v[:m * 14].reshape(m, 2, 7); W = v[m * 14:].reshape(m, 7)
        Wh = np.fft.fft(W, axis=1); Wh[:, 3] = 0; Wh[:, 4] = 0
        return U, np.real(np.fft.ifft(Wh, axis=1))


    for s in (1.25, 1.5):
        best = (-9, None)
        for st in range(6):
            m = 6
            v0 = rng.normal(size=m * 21)
            def f(v):
                U, W = unpack(v, m)
                g = gaps(G, U, W)
                # soft-min margin
                t = 60.0
                sm = -np.log(np.sum(np.exp(-t * (g - g.min())))) / t + g.min()
                return -sm / cost(U, W, 3 / s) ** s
            res = minimize(f, v0, method='L-BFGS-B', options={'maxiter': 3000})
            U, W = unpack(res.x, m)
            g = gamma_ratio(U, W, s)
            if g > best[0]:
                best = (g, (U, W))
        U, W = best[1]
        A1, A2, x1, x2, T1, T2, A3 = net_config(U, W)
        B = bound_at(s, A1, A2, x1, x2, T1, T2) if (A1 > 1e-6 and A2 > 1e-6) else np.nan
        print(f"E2 s={s}: best gamma/gIF = {best[0]:.5f} (features |A|=({A1:.3g},{A2:.3g},{A3:.1e}), x=({x1:.3f},{x2:.3f}))  bound there {B:.5f}", flush=True)

    # E3: s-monotonicity of the bound function for fixed tests
    viol = -9
    for trial in range(30):
        case = rig72.CASES[trial % 5]
        tau = rng.uniform(0.05, 0.95); x1 = rng.uniform(0, 1.6); x2 = rng.uniform(0, 1.6)
        m = rig72.Model(case, 'pat')
        mu3, w, _ = m.solve(1.3, tau, x1, x2, False, False)
        cert = rig72.certify_test(case, mu3, w, 'pat')
        vals = [rig72.box_bound(cert, tau, tau, x1, x1, x2, x2, False, False, s) for s in np.linspace(1.0, 1.5, 26)]
        viol = max(viol, max(vals[i] - vals[i + 1] for i in range(len(vals) - 1)))
    print(f"E3 max decrease of B(s) along s-grid (must be <= ~0): {viol:.3e}")
