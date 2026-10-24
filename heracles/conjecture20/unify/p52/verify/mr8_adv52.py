"""Math referee: adversarial sanity search for R9/R10 at (5,2): maximise gamma/gamma_IF over networks with beta_2 = 0.
Class searched: width m in {3,4,6}, inputs arbitrary (DC + both frequencies), outputs on frequencies {0, 1} plus a
frequency-2 output part that is projected so that beta_2 = 0 exactly (beta_2 is linear in the w-frequency-2 data).
Objective: exact margin h (min over 100 gaps) / P^s, maximised by L-BFGS on a soft-min surrogate then polished."""
import numpy as np
from scipy.optimize import minimize
from mr_net import Net

p, n = 5, 2
rng = np.random.default_rng(31)
B5 = 2 * 30.0 ** -1.5
a = np.arange(p)


def build(v, m):
    U = v[:m * n * p].reshape(m, n, p)
    W0 = v[m * n * p:m * n * p + m * 3].reshape(m, 3)        # DC, cos1, sin1
    W2 = v[m * n * p + m * 3:].reshape(m, 2)                  # cos2, sin2 amplitudes
    W = W0[:, :1] + W0[:, 1:2] * np.cos(2 * np.pi * a / p) + W0[:, 2:3] * np.sin(2 * np.pi * a / p)
    # project the frequency-2 output data so that beta_2 = sum_i conj(what_i(2)) uh1_i(2) uh2_i(2) = 0
    Uh = np.fft.fft(U, axis=2)
    g = Uh[:, 0, 2] * Uh[:, 1, 2]                              # beta_2 = (2/125) sum conj(wh_i(2)) g_i
    z = (p / 2) * (W2[:, 0] - 1j * W2[:, 1])                  # wh(2) for w = c cos + d sin : (p/2)(c - i d)
    # remove the component of conj(z) along conj(g)... want sum conj(z_i) g_i = 0: z <- z - (sum conj(z) g / |g|^2) conj(conj g)?
    gg = np.vdot(g, g).real
    if gg > 1e-300:
        lam = np.sum(np.conj(z) * g) / gg
        z = z - np.conj(lam) * g                               # then sum conj(z) g = lam*gg - lam*gg = 0
    c = (2 / p) * z.real; d = -(2 / p) * z.imag
    W = W + c[:, None] * np.cos(4 * np.pi * a / p) + d[:, None] * np.sin(4 * np.pi * a / p)
    return U, W


def ratio(v, m, s, soft=None):
    U, W = build(v, m)
    net = Net(p, n, U, W)
    G = net.gaps()[1]
    if soft is None:
        h = G.min()
    else:
        h = -soft * np.log(np.mean(np.exp(-(G - G.min()) / soft))) + G.min()
    P = net.cost(3 / s)
    gIF = 3 ** (1 - s) * B5 * 2.5 * 2 ** (-s)
    return h / P ** s / gIF, net


if __name__ == '__main__':
    best = -np.inf
    for s in (1.5, 1.25):
        for m in (3, 4, 6):
            for st in range(3):
                nv = m * n * p + m * 5
                v0 = rng.normal(size=nv)
                for soft in (0.05, 0.01, 0.002):
                    r = minimize(lambda v: -ratio(v, m, s, soft)[0], v0, method='L-BFGS-B', options=dict(maxiter=400))
                    v0 = r.x
                r = minimize(lambda v: -ratio(v, m, s)[0], v0, method='Nelder-Mead', options=dict(maxiter=3000))
                val, net = ratio(r.x, m, s)
                assert abs(net.beta(2)) < 1e-10 * (1 + abs(net.beta(1)))
                best = max(best, val)
                print(f"s={s} m={m} start={st}: gamma/gamma_IF = {val:.4f}  |beta_1|={abs(net.beta(1)):.3e} |beta_2|={abs(net.beta(2)):.1e}", flush=True)
    print(f"best one-feature gamma/gamma_IF found = {best:.4f}  (R9 claims <= 0.995; IF block value kappa_a/Gamma_s)")
