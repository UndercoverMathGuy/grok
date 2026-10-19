"""E2 (improved): locally optimised two-feature networks at (7,2), started from the two-feature IF network (blocks for
frequencies 1 and 2 at the programme-optimal amplitude ratio) plus noise (DC, frequency-3 inputs, shared content).
Outputs are kept free of frequency 3 so beta_3 = 0 exactly.  Reports gamma/gamma_IF before/after and the certified bound
at the optimised network's own data (s <= 1.2725 only; above that the bound is not claimed < 1).
Usage: python sanity_opt72.py s nstarts"""
import sys
import numpy as np
from scipy.optimize import minimize
from n2lib import Grid, gaps, cost, own_data
import rig72
from sanity_end2end import gamma_ratio, net_config, bound_at

G = Grid(7); p = 7; a = np.arange(7)
s = float(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rng = np.random.default_rng(5)


def if_block(j, amp):
    U = np.zeros((3, 2, 7)); W = np.zeros((3, 7))
    for i in range(3):
        ph = -np.pi * i / 3
        U[i, 0] = np.cos(2 * np.pi * j * a / 7 + ph); U[i, 1] = np.cos(2 * np.pi * j * a / 7 + ph)
        W[i] = np.cos(2 * np.pi * j * a / 7 + 2 * ph)
    return U * amp ** (1 / 3), W * amp ** (1 / 3)


def project(W):
    Wh = np.fft.fft(W, axis=1); Wh[:, 3] = 0; Wh[:, 4] = 0
    return np.real(np.fft.ifft(Wh, axis=1))


def softgamma(v, m, t):
    U = v[:m * 14].reshape(m, 2, 7); W = project(v[m * 14:].reshape(m, 7))
    g = gaps(G, U, W)
    gm = g.min()
    sm = gm - np.log(np.sum(np.exp(-t * (g - gm)))) / t
    return sm / cost(U, W, 3 / s) ** s


best = (-9, None)
for st in range(NS):
    ratio = 2 * np.cos(np.pi / 7) * rng.uniform(0.8, 1.25)
    U1, W1 = if_block(1, 1.0); U2, W2 = if_block(2, ratio)
    U = np.concatenate([U1, U2]); W = np.concatenate([W1, W2])
    m = 6 + 2
    U = np.concatenate([U, 0.3 * rng.normal(size=(2, 2, 7))]); W = np.concatenate([W, 0.3 * rng.normal(size=(2, 7))])
    U = U + 0.05 * rng.normal(size=U.shape); W = W + 0.05 * rng.normal(size=W.shape)
    v = np.concatenate([U.ravel(), W.ravel()])
    g0 = gamma_ratio(U, project(W), s)
    for t in (30.0, 100.0, 300.0, 1000.0):
        res = minimize(lambda vv: -softgamma(vv, m, t), v, method='L-BFGS-B', options={'maxiter': 1500})
        v = res.x
    U = v[:m * 14].reshape(m, 2, 7); W = project(v[m * 14:].reshape(m, 7))
    g1 = gamma_ratio(U, W, s)
    print(f"start {st}: gamma/gIF IF-start {g0:.5f} -> optimised {g1:.5f}", flush=True)
    if g1 > best[0]:
        best = (g1, (U, W))
U, W = best[1]
A1, A2, x1, x2, T1, T2, A3 = net_config(U, W)
B = bound_at(min(s, 1.5), A1, A2, x1, x2, T1, T2) if (A1 > 1e-6 and A2 > 1e-6) else float('nan')
print(f"s={s}: best optimised two-feature gamma/gIF = {best[0]:.5f}; own data |A|=({A1:.4g},{A2:.4g}), x=({x1:.3f},{x2:.3f}), "
      f"beta_3={A3:.1e}; certified-type bound at this data = {B:.5f}", flush=True)
