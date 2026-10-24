# v5: centred directional prices of single class-C features (counterexample search, NUMERICAL).
#   pi^c_k = sup_{atoms, eta} min_x [Re Phi(x) - cot(pi k/p) |Im Phi(x) + eta|] / Q^s,  Q = sum |C_i|^r.
# Theorem S: one coupled feature against an IF background beats the per-feature decoupled bound iff
# pi^c_k > pi for some k.  Compare with the IF price nu^{1-s} (= pi at n = 2).  Usage:
#   python v5_directional.py p n b [starts] [seed]
import numpy as np, sys, time
from scipy.optimize import minimize, minimize_scalar
from fmlib import grid, TAU

def make(p, n):
    X = grid(p, n); Z = np.exp(1j * TAU * 2 * X / p)            # (N, n), frequency-1 feature
    return Z

def unpack(x, m, n):
    a = x[:m]; b = x[m:2 * m]; th = x[2 * m:2 * m + m * n].reshape(m, n); eta = x[-1]
    return a + 1j * b, np.exp(1j * th), eta

def evalf(x, Z, m, n, cotk, s, tau, eps=1e-9):
    r = 1.0 / s
    C, xi, eta = unpack(x, m, n)
    fac = 1 + xi[None, :, :] * Z[:, None, :]                    # (N, m, n)
    P = np.prod(fac, axis=2)                                     # (N, m)
    Phi = P @ C + 1j * eta
    im = np.imag(Phi); sq = np.sqrt(im ** 2 + eps ** 2)
    cm = np.real(Phi) - cotk * sq
    mn = cm.min(); e = np.exp(-(cm - mn) / tau); Zs = e.sum(); q = e / Zs
    sm = mn - tau * np.log(Zs)
    Q = np.sum((np.abs(C) ** 2 + eps ** 2) ** (r / 2))
    val = sm / Q ** s
    # gradients
    w = 1 - 1j * cotk * im / sq                                  # d cm = Re(conj(w) dPhi)
    qw = q * np.conj(w)                                          # (N,)
    gC = qw @ P                                                  # sum_x q conj(w) P_i  (complex)
    ga = np.real(gC); gb = np.real(1j * gC)
    # dPhi/dth_it = C_i * prod_{t' != t} fac * i xi_it z_t
    gth = np.zeros((m, n))
    for t in range(n):
        others = np.prod(np.delete(fac, t, axis=2), axis=2) if n > 1 else np.ones_like(P)
        dP = others * (1j * xi[None, :, t] * Z[:, t:t + 1])       # (N, m)
        gth[:, t] = np.real((qw @ dP) * C)
    geta = np.real(np.sum(qw * 1j))
    dQa = r * (np.abs(C) ** 2 + eps ** 2) ** (r / 2 - 1) * np.real(C)
    dQb = r * (np.abs(C) ** 2 + eps ** 2) ** (r / 2 - 1) * np.imag(C)
    gval = np.concatenate([ga / Q ** s - s * val / Q * dQa, gb / Q ** s - s * val / Q * dQb,
                           gth.ravel() / Q ** s, [geta / Q ** s]])
    return val, gval, Phi, Q

def exact_ratio(x, Z, m, n, cotk, s):
    C, xi, _ = unpack(x, m, n)
    Phi = np.prod(1 + xi[None, :, :] * Z[:, None, :], axis=2) @ C
    Q = np.sum(np.abs(C) ** (1.0 / s))
    f = lambda eta: -np.min(np.real(Phi) - cotk * np.abs(np.imag(Phi) + eta))
    lo, hi = -np.abs(Phi).max() - 1, np.abs(Phi).max() + 1
    res = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
    return -res.fun / Q ** s, -f(0.0) / Q ** s

def search(p, n, b, k, starts=12, seed=0, centred=True):
    s = (n + 1) / b; nu = n + 1; Z = make(p, n); cotk = 1 / np.tan(np.pi * k / p)
    rng = np.random.default_rng(seed); best = (-np.inf, None)
    for st in range(starts):
        m = [n, nu, nu + 1, 2 * nu][st % 4]
        if st % 3 == 0 and m >= nu:      # perturbed KM2 block (+ extra atoms)
            a = np.zeros(m); a[:nu] = 1.0 / nu; b_ = np.zeros(m)
            th = np.zeros((m, n)); th[:nu] = (TAU * np.arange(nu) / nu)[:, None] * np.ones(n)
            a += 0.15 * rng.normal(size=m) / nu; b_ += 0.15 * rng.normal(size=m) / nu
            th += 0.4 * rng.normal(size=th.shape)
        else:
            a = rng.normal(size=m); b_ = rng.normal(size=m); th = rng.uniform(0, TAU, (m, n))
        x = np.concatenate([a, b_, th.ravel(), [0.0]])
        for tau in [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
            def f(x):
                xx = x.copy()
                if not centred: xx[-1] = 0.0
                v, g, _, Q = evalf(xx, Z, m, n, cotk, s, tau * (np.abs(xx[:2 * m]).sum() + 1e-9))
                if not centred: g[-1] = 0.0
                pen = 1e-2 * np.log(Q) ** 2
                return -(v - pen), -g   # pen gradient omitted (only fixes scale drift; tiny)
            res = minimize(f, x, jac=True, method='L-BFGS-B', options={'maxiter': 400})
            x = res.x
        rc, r0 = exact_ratio(x, Z, m, n, cotk, s)
        val = rc if centred else r0
        if val > best[0]: best = (val, (m, x.copy()))
    return best

if __name__ == '__main__':
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    starts = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    s = (n + 1) / b; nu = n + 1; F = (p - 1) // 2
    t0 = time.time()
    for k in range(1, F + 1):
        val, (m, x) = search(p, n, b, k, starts, seed)
        print(f'p={p} n={n} b={b} k={k} (half-angle {180*k/p:.1f} deg): max ratio pi^c_k/nu^(1-s) = '
              f'{val/nu**(1-s):.6f}  (best width {m})  [{time.time()-t0:.0f}s]', flush=True)
        np.save(f'v5_best_p{p}_n{n}_b{b}_k{k}.npy', np.concatenate([[m], x]))
