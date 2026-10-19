"""Referee: adversarial networks at (5,2) whose own data is pinned inside the HARDEST Region-II boxes of the logs.

maximise h subject to: cost sum ||omega||^2 = 1 (b = 2), beta_2 = 0, |B| = m1 |A|, |C| = m2 |A|, |D| = m12 |A|,
Im(e^{-i ph} A D conj(B C)) = 0  (phi_pay = ph), starting from random / IF-perturbed / atom-plus-compensator networks.
Then run the end-to-end check (cc3.check): direct ratio <= min(trivial, certificate) at the network's own data <= 0.995,
and compare with the logged box bound.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import minimize
from cc_net import Net, cancel_feature
import cc3_e2e52 as E

p = 5
rng = np.random.default_rng(31)
TARGETS = [((0.5, 0.1, 1.1), -2.55), ((0.9, 0.95, 0.7), -2.8), ((0.3, 0.1, 1.1), -2.6), ((0.9, 0.1, 0.9), -2.0),
           ((0.7, 0.5, 0.9), 2.0), ((1.1, 0.3, 0.1), -1.2), ((0.35, 0.35, 0.9), 1.0), ((0.6, 0.6, 0.6), 0.0)]


def own_t(nt):
    A = nt.neuron_beta(1).sum()
    uh1 = np.fft.fft(nt.U1, axis=1); uh2 = np.fft.fft(nt.U2, axis=1); wh = np.fft.fft(nt.W, axis=1)
    c = 2 * p ** -3.0 * np.conj(wh[:, 1])
    B = np.sum(c * uh1[:, -1] * uh2[:, 1]); C = np.sum(c * uh1[:, 1] * uh2[:, -1]); D = np.sum(c * uh1[:, -1] * uh2[:, -1])
    b2 = np.sum(2 * p ** -3.0 * np.conj(wh[:, 2]) * uh1[:, 2] * uh2[:, 2])
    return A, B, C, D, b2


def run(m, ph, k=5, starts=4):
    best = None; bv = -np.inf
    for st in range(starts):
        net0 = [E.perturbed_if(0.4), E.single_atom_plus(0.5), E.make_random(k), E.make_random(k)][st % 4]
        kk = net0.U1.shape[0]
        z0 = np.concatenate([net0.U1.ravel(), net0.U2.ravel(), net0.W.ravel()]) / np.sqrt(net0.cost(2))

        def unpack(z):
            return Net(p, z[:kk * p].reshape(kk, p), z[kk * p:2 * kk * p].reshape(kk, p), z[2 * kk * p:].reshape(kk, p))

        def ceq(zz):
            nt = unpack(zz[:-1]); A, B, C, D, b2 = own_t(nt)
            a2 = abs(A) ** 2
            sc = 1e3
            return np.array([nt.cost(2) - 1, sc * b2.real, sc * b2.imag, sc * (abs(B) ** 2 - m[0] ** 2 * a2),
                             sc * (abs(C) ** 2 - m[1] ** 2 * a2), sc * (abs(D) ** 2 - m[2] ** 2 * a2),
                             sc * np.imag(np.exp(-1j * ph) * A * D * np.conj(B * C))])

        def cin(zz):
            nt = unpack(zz[:-1]); A, B, C, D, b2 = own_t(nt)
            return np.concatenate([nt.gaps().ravel() - zz[-1], [1e3 * np.real(np.exp(-1j * ph) * A * D * np.conj(B * C))]])
        zz0 = np.concatenate([z0, [unpack(z0).h()]])
        res = minimize(lambda zz: -zz[-1], zz0, method='SLSQP', constraints=[{'type': 'eq', 'fun': ceq}, {'type': 'ineq', 'fun': cin}],
                       options=dict(maxiter=500, ftol=1e-12))
        if np.max(np.abs(ceq(res.x))) > 1e-5:
            continue
        nt = cancel_feature(unpack(res.x[:-1]), 2)
        v = E.stats(nt, 1.5)
        if v > bv:
            bv, best = v, nt
    return best, bv


if __name__ == '__main__':
    t0 = time.time(); nbad = 0
    for (m, ph) in TARGETS:
        nt, v = run(m, ph)
        if nt is None:
            print(f"target m={m} ph={ph}: no feasible optimum found"); continue
        o = E.check(nt, 'pinned')
        A, B, C, D, _ = own_t(nt)
        mm = np.array([abs(B), abs(C), abs(D)]) / abs(A)
        print(f"target m={m} ph={ph:+.2f}: achieved m={np.round(mm,3)} ratio(b=2)={o['ratio1.5']:+.4f} "
              f"bound={o.get('bound', float('nan')):.4f} F={o.get('F', float('nan')):.4f} P1={o.get('P1', float('nan')):.4f} ok={o['ok']}", flush=True)
        nbad += (not o['ok'])
    print(f"failures={nbad}; time {time.time()-t0:.0f}s")
