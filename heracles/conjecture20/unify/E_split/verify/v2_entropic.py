"""Referee check of E2 (entropic Lemma 7, aligned) and E2b (non-aligned counterexample).
Adversarial: minimise H(esc_r(w)) over aligned configurations subject to the full IF condition (penalty + exact polish).
Also checks the frame identity sum_l w_l m_l m_l^* = I (aligned + interval IF) => max w <= 1/nu."""
import itertools, math
import numpy as np
from scipy.optimize import minimize, least_squares

rng = np.random.default_rng(7)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def Ts(n):
    return [T for r in range(1, n + 1) for T in itertools.combinations(range(n), r)]


def resid(w, phi, n):
    xi = np.exp(1j * phi)
    R = []
    for T in Ts(n):
        v = (w * np.prod(xi[:, list(T)], axis=1)).sum()
        R += [v.real, v.imag]
    return np.array(R)


def Hesc(w, r):
    e = w ** r; e = e / e.sum()
    e = e[e > 0]
    return float(-(e * np.log(e)).sum())


def unpack(x, m, n):
    a = x[:m]; phi = x[m:].reshape(m, n)
    w = np.exp(a - a.max()); w = w / w.sum()
    return w, phi


worst = {}
for n in [2, 3, 4]:
    nu = n + 1
    for m in [nu, nu + 1, nu + 2, nu + 4]:
        for r in [1.0, 0.6, 2.0 / nu]:
            best = np.inf
            for trial in range(12):
                x0 = np.concatenate([rng.standard_normal(m) * 1.5, rng.uniform(0, 2 * np.pi, m * n)])
                for mu in [10.0, 100.0, 1e3, 1e4]:
                    f = lambda x: Hesc(unpack(x, m, n)[0], r) + mu * (resid(*unpack(x, m, n), n) ** 2).sum()
                    x0 = minimize(f, x0, method='L-BFGS-B').x
                # exact polish: stay close, enforce IF
                def g(x):
                    return np.concatenate([1e3 * resid(*unpack(x, m, n), n), 1e-3 * (x - x0)])
                x1 = least_squares(g, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15).x
                w, phi = unpack(x1, m, n)
                res = np.abs(resid(w, phi, n)).max()
                if res > 1e-10:
                    continue
                Hv = Hesc(w, r)
                best = min(best, Hv - math.log(nu))
                # frame identity and max-weight bound
                xi = np.exp(1j * phi)
                M = np.zeros((nu, nu), complex)
                for l in range(m):
                    ch = np.concatenate([[1], np.cumprod(xi[l])])
                    M += w[l] * np.outer(ch, ch.conj())
                assert np.abs(M - np.eye(nu)).max() < 1e-8, np.abs(M - np.eye(nu)).max()
                assert w.max() <= 1 / nu + 1e-9
            worst[(n, m, r)] = best
            say(f"n={n} m={m} r={r:.3f}: min over exact IF optima of H(esc_r)-log nu = {best:+.3e}")
say(f"OVERALL min = {min(worst.values()):+.3e}  (E2 predicts >= 0)")

say("== E2b non-aligned example ==")
mod = np.array([0.49144832, 0.10252906, 0.40602261]); arg = np.array([4.59829238, 3.03313222, 1.07566332])
xa = np.array([[6.24048247, 2.48936183], [5.18952748, 1.43840683], [0.21132144, 2.7433861]])
beta = mod * np.exp(1j * arg); xi = np.exp(1j * xa)
for T in Ts(2):
    say(f"  T={T}: |sum beta_l prod xi| = {abs((beta * np.prod(xi[:, list(T)], axis=1)).sum()):.2e}")
w = mod / mod.sum()
say(f"  H(|beta|) = {-(w*np.log(w)).sum():.6f} vs log 3 = {math.log(3):.6f};  |beta| = {abs(beta.sum()):.5f}")
for r in [2/3, 0.8, 1.0]:
    say(f"  r={r:.3f}: Lemma 7 sum|b|^r = {(mod**r).sum():.4f} >= 3^(1-r)|beta|^r = {3**(1-r)*abs(beta.sum())**r:.4f}")
# Realise as actual neurons at p = 5, n = 2 and recompute atoms/IF sums from Fourier coefficients
p, n = 5, 2; nu = 3; j = 1
a = np.arange(p)
U = np.zeros((3, n, p)); W = np.zeros((3, p))
for l in range(3):
    # xi = conj(uhat)/uhat = e^{-2 i arg uhat}  => arg uhat = -xa/2
    for t in range(n):
        U[l, t] = np.cos(2 * np.pi * j * a / p - xa[l, t] / 2)
    # beta_l = 2 p^-nu conj(what) prod uhat ; choose what phase/modulus
    prod_uhat = np.prod([(p / 2) * np.exp(-1j * xa[l, t] / 2) for t in range(n)])
    target = beta[l]
    conj_what = target / (2 * p ** (-nu) * prod_uhat)
    what = np.conj(conj_what)
    W[l] = (2 / p) * np.abs(what) * np.cos(2 * np.pi * j * a / p + np.angle(what))
Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1)
at = 2 * p ** (-nu) * np.conj(Wh[:, j]) * np.prod(Uh[:, :, j], axis=1)
say(f"  realised atoms match: {np.abs(at - beta).max():.2e}")
for T in Ts(2):
    pr = np.conj(Wh[:, j]).copy()
    for t in range(n):
        pr = pr * (Uh[:, t, (-j) % p] if t in T else Uh[:, t, j])
    say(f"  realised beta_(1,{T}) = {abs(2*p**(-nu)*pr.sum()):.2e}")
open(__file__.replace('.py', '.out'), 'w').write("\n".join(out) + "\n")
