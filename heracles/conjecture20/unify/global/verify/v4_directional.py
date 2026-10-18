# R9 (directional price at n=2) and R8 (profile programme) independent checks.
#  Single feature = outputs in span{1, cos, sin} at frequency 1 (DC cancels in gaps), inputs arbitrary.
#  m(l) = min_x g(x,l) (unnormalised), M = max_T |beta_{1,T}|, T nonempty subset of {1,2}.
#  (i)   (m(l)+m(-l))/2 <= kappa_1(l) (Re beta_1 - |cos(pi l/p)| M)              arbitrary inputs
#  (ii)  (m(l)+m(-l))/2 <= kappa_1(l) Re beta_1 - |sin(pi l/p)| M                 single-frequency inputs
#  ThmA  |beta_1| - R/(2 sqrt 3) <= 3^{1-s} Q^s,  R^2 = sum_T |beta_{1,T}|^2       (cited, Part I)
#  (iii) m(l)+m(-l) <= 2 3^{1-s} kappa_1(l) B_p P^s   (all l single-freq; |cos|>=1/2 arbitrary)
import math, sys, numpy as np
from scipy.optimize import minimize
from vnet import *

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 11)


def single_feature(p, n, m, kind):
    a = np.arange(p)
    if kind == 'gauss':
        U = rng.normal(size=(m, n, p))
    elif kind == 'partner':        # strong frequency-3 (own partner) + frequency-1 content
        B = np.vstack([freq_basis(p, 1), freq_basis(p, 3 % p if 3 % p else 1)])
        U = rng.normal(size=(m, n, 4)) @ B
    elif kind == 'dc':             # DC + frequency 1 (single-frequency class)
        B = np.vstack([freq_basis(p, 0), freq_basis(p, 1)])
        U = rng.normal(size=(m, n, 3)) @ B * np.array([3.0, 1, 1])[None, None, :].mean()
        U = rng.normal(size=(m, n, 3)) @ (np.diag([rng.uniform(0, 3), 1, 1]) @ B)
    elif kind == 'sign':           # sign vectors (p=3 mechanism shape)
        U = np.sign(rng.normal(size=(m, n, p))) * np.exp(rng.normal(size=(m, n, 1)) * np.arange(p)[None, None, :] / p)
    elif kind == 'ifpert':
        U0, W0 = if_block(p, n, 1, 1.0, rng.uniform(0, 1)); k = min(m, n + 1)
        U = 10 ** rng.uniform(-3, 0.3) * rng.normal(size=(m, n, p)); U[:k] += U0[:k]
        W = 10 ** rng.uniform(-3, 0.3) * rng.normal(size=(m, 2)) @ freq_basis(p, 1); W[:k] += W0[:k]
        return U, W
    W = rng.normal(size=(m, 3)) @ np.vstack([freq_basis(p, 0), freq_basis(p, 1)])
    return U, W


def is_single_freq(U, p):
    allowed = {0, 1, p - 1}
    Uh = np.abs(np.fft.fft(U, axis=2))
    return all(Uh[..., k].max() < 1e-9 * max(1, Uh.max()) for k in range(p) if k not in allowed)


def check(U, W, b):
    m, n, p = U.shape; nu = n + 1; s = nu / b; r = 1 / s
    f = table(U, W); G = gaps(f).reshape(-1, p - 1); mp_ = G.min(0)
    beta = clean(f)[0]; sf = signflip(f, 1)
    Ts = [T for T in sf if any(T)]
    M = max(abs(sf[T]) for T in Ts); R = math.sqrt(sum(abs(sf[T]) ** 2 for T in Ts))
    Q = (np.abs(atoms(U, W, 1)) ** r).sum(); P = cost(U, W, b)
    scale = abs(beta) + M + 1e-300
    out = {}
    out['ThmA'] = (abs(beta) - R / (2 * math.sqrt(3)) - 3 ** (1 - s) * Q ** s) / scale
    out['ThmA_half'] = (beta.real - M / 2 - 3 ** (1 - s) * Q ** s) / scale
    sfq = is_single_freq(U, p)
    for l in range(1, (p - 1) // 2 + 1):
        ka = kappa(p, 1, l); lhs = 0.5 * (mp_[l - 1] + mp_[p - l - 1])
        out.setdefault('i', -np.inf)
        out['i'] = max(out['i'], (lhs - ka * (beta.real - abs(math.cos(math.pi * l / p)) * M)) / (ka * scale))
        if sfq:
            out.setdefault('ii', -np.inf)
            out['ii'] = max(out['ii'], (lhs - (ka * beta.real - abs(math.sin(math.pi * l / p)) * M)) / (ka * scale))
            out.setdefault('iii_sf', -np.inf)
            out['iii_sf'] = max(out['iii_sf'], (2 * lhs - 2 * 3 ** (1 - s) * ka * B_p(p, n) * P ** s) / (ka * B_p(p, n) * P ** s))
        if abs(math.cos(math.pi * l / p)) >= 0.5:
            out.setdefault('iii_arb', -np.inf)
            out['iii_arb'] = max(out['iii_arb'], (2 * lhs - 2 * 3 ** (1 - s) * ka * B_p(p, n) * P ** s) / (ka * B_p(p, n) * P ** s))
    # R8 cone lemma: m(l) >= e kappa_1(l) where e*B_p P^s = min_x cm(Phi_1(x))
    Ph = Phi(f, 1).ravel(); cm = (Ph.real - abs(Ph.imag) / math.tan(math.pi / p)).min()
    out['cone'] = max((cm * kappa(p, 1, l) - mp_[l - 1]) / scale for l in range(1, p))
    return out, sfq


worst = {}
count = {}
for trial in range(1200):
    p = int(rng.choice([5, 7, 11, 13])); n = 2; m = int(rng.integers(1, 7)); b = rng.uniform(2, 3)
    kind = rng.choice(['gauss', 'partner', 'dc', 'sign', 'ifpert'])
    U, W = single_feature(p, n, m, kind)
    o, sfq = check(U, W, b)
    for k, v in o.items():
        worst[k] = max(worst.get(k, -np.inf), v); count[k] = count.get(k, 0) + 1
print("random single features (n=2; p in 5,7,11,13; b in [2,3]); max normalised violation (must be <= 0):")
for k in sorted(worst):
    print(f"   {k:9s}: {worst[k]: .3e}   ({count[k]} nets)")

# adversarial: maximise violation of (i), (ii) and ThmA by Powell from many starts (small widths)
def adv(key, p, m, b, single_freq, starts=6, iters=3000):
    n = 2; best = -np.inf
    a = np.arange(p)
    Bin = np.vstack([freq_basis(p, 0), freq_basis(p, 1)]) if single_freq else np.eye(p)
    Bout = np.vstack([freq_basis(p, 0), freq_basis(p, 1)])
    di = Bin.shape[0]
    def unpack(z):
        U = z[:m * n * di].reshape(m, n, di) @ Bin; W = z[m * n * di:].reshape(m, 3) @ Bout; return U, W
    def obj(z):
        U, W = unpack(z); o, _ = check(U, W, b); return -o[key]
    for st in range(starts):
        if st % 2 == 0:
            U0, W0 = if_block(p, n, 1, 1.0, rng.uniform(0, 1)); k = min(m, 3)
            z0 = rng.normal(size=m * n * di + m * 3) * 0.3
            # project IF block on basis (least squares)
            zU = np.linalg.lstsq(Bin.T, U0.reshape(-1, p).T, rcond=None)[0].T.reshape(3, n, di)
            zW = np.linalg.lstsq(Bout.T, W0.T, rcond=None)[0].T
            z0[:k * n * di] += zU[:k].ravel(); z0[m * n * di:m * n * di + 3 * k] += zW[:k].ravel()
        else:
            z0 = rng.normal(size=m * n * di + m * 3)
        res = minimize(obj, z0, method='Powell', options={'maxiter': iters, 'xtol': 1e-10, 'ftol': 1e-13})
        best = max(best, -res.fun)
    return best

print("adversarial (Powell) maxima of normalised violation (must be <= 0):")
for (p, m, b) in [(5, 2, 2.0), (5, 3, 2.5), (7, 2, 2.0), (7, 3, 3.0), (11, 2, 2.2)]:
    r_i = adv('i', p, m, b, False); r_ii = adv('ii', p, m, b, True); r_A = adv('ThmA', p, m, b, False)
    print(f"   p={p} m={m} b={b}: (i) {r_i: .3e}   (ii, single-freq) {r_ii: .3e}   ThmA {r_A: .3e}", flush=True)
