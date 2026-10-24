# Referee checks: R6 identity (class C), R3(c) equality outside class C, R7(a) at b=2 with mixed budget-tight
# neurons, R5(i) bound on no-cheap class-C networks (incl. b=2 and non-feature interference), budget-equality lemma,
# R7(b)/no-cheap chain on the p=3 counterexample (n=11, exact_certificate weights).
import numpy as np, itertools, sys
from vlib import *
from v_r3 import if_union

rng = np.random.default_rng(7)

def freq_neuron(p, n, j, amps, phases):
    xs = np.arange(p)
    U = np.array([amps[t] * np.cos(2 * np.pi * j * xs / p + phases[t]) for t in range(n)])
    W = amps[n] * np.cos(2 * np.pi * j * xs / p + phases[n])
    return U, W

# ---- R6 identity on random class-C networks
worst = 0
for trial in range(40):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); F = (p - 1) // 2; m = rng.integers(2, 7)
    X, Y = grid(p, n)
    Us, Ws = [], []
    for i in range(m):
        u, w = freq_neuron(p, n, rng.integers(1, F + 1), rng.uniform(0.3, 1.5, n + 1), rng.uniform(0, 2 * np.pi, n + 1))
        Us.append(u); Ws.append(w)
    U, W = np.array(Us), np.array(Ws)
    G = gaps(U, W, p, n, X, Y)
    for d in range(1, p):
        gh = np.fft.fftn(G[:, d - 1].reshape((p,) * n)) / p ** n          # E_x g rho^{-<chi,x>}
        for l in range(1, F + 1):
            for k in range(1, n + 1):
                for T in itertools.combinations(range(n), k):
                    chi = tuple((2 * l) % p if t in T else 0 for t in range(n))
                    pred = 0.5 * np.conj(signflip(U, W, p, l, T)) * (1 - np.exp(-2j * np.pi * l * d / p))
                    worst = max(worst, abs(gh[chi] - pred) / np.abs(G).max())
print(f'R6 identity: max rel error over 40 random class-C networks = {worst:.1e}')

# ---- R3(c): an 'IF' (all beta_{l,T}=0) network outside class C whose gap depends on x
p, n = 7, 2; X, Y = grid(p, n); xs = np.arange(p)
U = np.array([[np.cos(2 * np.pi * 1 * xs / p), np.cos(2 * np.pi * 2 * xs / p)]]); W = np.array([np.cos(2 * np.pi * 3 * xs / p)])
G = gaps(U, W, p, n, X, Y)
allbT = max(abs(signflip(U, W, p, l, T)) for l in range(1, 4) for k in (1, 2) for T in itertools.combinations(range(n), k))
print(f'R3(c) test: mixed neuron (1,2;3) at p=7: max|beta_l| = {np.abs(atoms(U, W, p)).max():.1e}, max|beta_lT| = {allbT:.1e}, '
      f'but g(.,delta) ranges over [{G.min():.3f},{G.max():.3f}] -> E_q g != sum Re(conj(beta)K~) = 0 for point-mass q '
      f'(E_q g = {G[np.argmax(G[:,0]),0]:.3f})')

# ---- budget-equality lemma: b>2 forces purity; b=2 allows equal-profile mixtures
def budget_ratio(U, W, p, b):
    nu = U.shape[0] + 1
    om2 = (U ** 2).sum() + (W ** 2).sum(); r = b / nu; Cb = Bp(p, nu - 1) ** r
    at = atoms(U[None], W[None], p)[0]
    return np.sum(np.abs(at) ** r) / (Cb * om2 ** (b / 2)), at

p, n = 7, 2; nu = 3; xs = np.arange(p)
c1, c2 = np.sqrt(0.6), np.sqrt(0.4)
def mixed(ph):
    f = lambda a: np.sqrt(2 / p) * (c1 * np.cos(2 * np.pi * xs / p + a) + c2 * np.cos(2 * np.pi * 2 * xs / p + 2 * a))
    return np.array([f(ph), f(ph)]), f(-2 * ph) * 1.0
U, W = mixed(0.3)
for b in [2.0, 2.2, 3.0]:
    print(f'budget ratio of an equal-profile 2-frequency neuron (p=7,n=2) at b={b}: {budget_ratio(U, W, p, b)[0]:.6f}')

# ---- R7(a) at b = 2: features carried by budget-tight MIXED neurons with different profiles, aligned atoms
b = 2.0; s = nu / b; r = 1 / s
def mixed_c(c1, scale):
    c2 = np.sqrt(1 - c1 ** 2)
    f = lambda: np.sqrt(2 / p) * (c1 * np.cos(2 * np.pi * xs / p) + c2 * np.cos(2 * np.pi * 2 * xs / p))
    return scale * np.array([f(), f()]), scale * f()
Us, Ws = [], []
for (c, sc) in [(0.9, 1.0), (0.5, 0.8), (0.3, 1.3)]:
    u, w = mixed_c(c, sc); Us.append(u); Ws.append(w)
U, W = np.array(Us), np.array(Ws)
at = atoms(U, W, p); sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
Cb = Bp(p, n) ** r
print('  budget ratios of the three neurons at b=2:', [round(float(np.sum(np.abs(at[i]) ** r) / (Cb * sq[i] ** (b / 2))), 10) for i in range(3)])
a_i = sq ** (nu / 2)
for l in (1, 2):
    Ql = np.sum(np.abs(at[:, l - 1]) ** r); bl = np.abs(at[:, l - 1].sum()); aligned = np.isclose(bl, np.abs(at[:, l - 1]).sum())
    ratio = bl / (nu ** (1 - s) * Ql ** s)
    wl = sq ** (b / 2) / np.sum(sq ** (b / 2))           # cost shares of the neurons carrying l
    Hs = np.log(np.sum(wl ** s)) / (1 - s)
    print(f'  R7(a) b=2 feature {l}: aligned={aligned}; cheapness ratio {ratio:.6f} vs exp((s-1)(log nu - H_s(w))) = '
          f'{np.exp((s-1)*(np.log(nu)-Hs)):.6f};  |beta_il| = B_p a_i ? {np.allclose(np.abs(at[:, l-1]), Bp(p,n)*a_i)}')

# ---- R5(i): no-cheap class-C networks near IF optima, b in {2, 2.4, nu}, plus a non-feature interfering pair
def chk_R5(p, n, b, S, trials=60, eps=0.15, nonfeat=None):
    nu = n + 1; s = nu / b; r = 1 / s; X, Y = grid(p, n); F = (p - 1) // 2
    Gam, y = programme(p, s, S, starts=80)
    U0, W0, lab = if_union(p, n, b, y)
    worst = -np.inf; cnt = 0; act = []
    for tr in range(trials):
        U = U0 * np.exp(eps * rng.standard_normal((len(lab), n, 1))); W = W0 * np.exp(eps * rng.standard_normal((len(lab), 1)))
        # random in-plane phase rotations (stay frequency-pure)
        for i in range(len(lab)):
            j = lab[i]; xs = np.arange(p)
            for t in range(n + 1):
                v = U[i, t] if t < n else W[i]
                vh = np.fft.fft(v); ph = np.exp(1j * eps * rng.standard_normal())
                vh[j] *= ph; vh[p - j] *= np.conj(ph)
                v2 = np.real(np.fft.ifft(vh))
                if t < n: U[i, t] = v2
                else: W[i] = v2
        if nonfeat is not None:
            l2, amp = nonfeat
            u1, w1 = freq_neuron(p, n, l2, [amp] * (n + 1), rng.uniform(0, 2 * np.pi, n + 1))
            u2, w2 = u1.copy(), -w1.copy()                       # opposite atom: beta_l2 cancels, beta_{l2,T} need not
            u2[0] = freq_neuron(p, n, l2, [amp] * (n + 1), rng.uniform(0, 2 * np.pi, n + 1))[0][0]
            # re-make atom of neuron 2 the negative of neuron 1's by solving the output phase
            at1 = atoms(u1[None], w1[None], p)[0, l2 - 1]
            at2 = atoms(u2[None], w1[None], p)[0, l2 - 1]
            # choose w2 = rotated w1 so that atom2 = -at1
            rot = -at1 / at2
            wh = np.fft.fft(w1); wh[l2] *= np.conj(rot); wh[p - l2] *= rot; w2 = np.real(np.fft.ifft(wh))
            U = np.concatenate([U, u1[None], u2[None]]); W = np.concatenate([W, w1[None], w2[None]])
        at = atoms(U, W, p)
        P, sq = cost(U, W, b)
        Q = np.sum(np.abs(at) ** r, 0); bl = np.abs(at.sum(0))
        feats = [l for l in range(1, F + 1) if bl[l - 1] > 1e-12 * bl.max()]
        if not set(feats) <= set(S): continue
        if np.any(bl > nu ** (1 - s) * Q ** s * (1 + 1e-12)): continue
        G = gaps(U, W, p, n, X, Y); h = G.min()
        dvals = [sum(Q[l - 1] ** s * kap(p, l, d) for l in S) for d in range(1, p)]
        d0 = int(np.argmin(dvals)) + 1
        det = max(abs(signflip(U, W, p, l, T)) * np.sqrt(kap(p, l, d0) / 2)
                  for l in range(1, F + 1) for k in range(1, n + 1) for T in itertools.combinations(range(n), k))
        bound = nu ** (1 - s) * Bp(p, n) * Gam - det / P ** s
        worst = max(worst, (h / P ** s - bound) / (nu ** (1 - s) * Bp(p, n) * Gam)); cnt += 1; act.append(det / P ** s / (nu ** (1 - s) * Bp(p, n) * Gam))
    print(f'  R5(i) (p,n,b)=({p},{n},{b}) S={S} nonfeat={nonfeat}: {cnt} no-cheap samples; max (gamma - bound)/gIF = {worst:.2e} (<=0); '
          f'det/gIF in [{min(act) if act else 0:.1e},{max(act) if act else 0:.1e}]')
    sys.stdout.flush()

chk_R5(5, 2, 2.0, [1, 2]); chk_R5(5, 2, 2.6, [1, 2]); chk_R5(7, 2, 2.0, [1, 2, 3]); chk_R5(5, 3, 2.0, [1]);
chk_R5(5, 3, 2.0, [1], nonfeat=(2, 0.05)); chk_R5(7, 2, 2.4, [1, 3], nonfeat=(2, 0.05)); chk_R5(7, 3, 4.0, [1, 2, 3])
