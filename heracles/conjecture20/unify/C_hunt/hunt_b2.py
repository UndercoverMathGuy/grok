# HUNT (b2): can foreign zero-mean fields LIFT a cheap single feature?  Usage: python hunt_b2.py p n b nrest seed
# Own feature 1: class-C neurons u_t = A_i cos(2 pi a/p + phi_it), w = A_i cos(2 pi c/p + sum_t phi_it) with FROZEN
#   amplitudes A_i (mass profile) and aligned atoms (beta_i real > 0): the feature's gain G = |A|/(nu^{1-s} Q^s) is fixed.
#   Phases phi_it free.
# Extra neurons: pure interference (every j in F, the own frequency included, missing from some slot), seeded as
#   channel neurons (inputs |2-j| on T, j off T, output j), partner neurons (inputs 3 on T, 1 off T, output 1), DC
#   neurons, random neurons.
# For each own profile we compare: own-only optimum (phases only) vs. own + foreign optimum.
# Ratio: gamma / (nu^{1-s} B_p kappa_min) (= W/D at n = 2).
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
import hlib as H, glib as Gb
GL = H.GL

p = int(sys.argv[1]); n = int(sys.argv[2]); bexp = float(sys.argv[3]); nrest = int(sys.argv[4]); seed = int(sys.argv[5])
nu = n + 1; s = nu / bexp; r = 1 / s; F = (p - 1) // 2
rng = np.random.default_rng(seed)
kmin = 1 - np.cos(2 * np.pi / p)
D = nu ** (1 - s) * H.Bp(p, n) * kmin
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
foreign = [j for j in range(2, F + 1)]
ALL = set(range(0, F + 1))
a_ = np.arange(p)


def red(k):
    k %= p
    return min(k, p - k)


def struct_allowed(base):
    # extra neurons are PURE INTERFERENCE: for every j in F (including the own frequency 1) some slot misses j,
    # so beta_ij = 0 for all j: the clean coefficients come from the frozen own block alone.
    al = [set(A) for A in base]
    for j in range(1, F + 1):
        if any(j not in A for A in al): continue
        al[int(rng.integers(0, n + 1))].discard(j)
    return al


class Combo:
    def __init__(self, Aown, galw):
        self.Aown = np.asarray(Aown, float); self.mo = len(Aown)
        self.g = Gb.GNet(p, n, bexp, galw) if galw else None
        self.nf = self.g.nx if self.g else 0

    def UW(self, x):
        ph = x[:self.mo * n].reshape(self.mo, n)
        Uo = self.Aown[:, None, None] * np.cos(2 * np.pi * a_[None, None, :] / p + ph[:, :, None])
        psi = ph.sum(1)
        Wo = self.Aown[:, None] * np.cos(2 * np.pi * a_[None, :] / p + psi[:, None])
        if self.g:
            Uf, Wf = self.g.to_UW(x[self.mo * n:])
            return np.concatenate([Uo, Uf]), np.concatenate([Wo, Wf]), ph, psi
        return Uo, Wo, ph, psi

    def gamma(self, x):
        U, W, _, _ = self.UW(x); nt = GL.Net(U, W, bexp); return nt.margin() / nt.cost() ** s

    def soft(self, x, tau):
        U, W, ph, psi = self.UW(x); nt = GL.Net(U, W, bexp)
        c = nt.cost(); g = nt.gaps() * c ** (-s)
        gmin = g.min(); z = np.exp(-(g - gmin) / tau); Z = z.sum(); lam = z / Z
        Fv = gmin - tau * np.log(Z)
        gU, gW = nt.grad_lin(lam); cU, cW = nt.grad_cost(); Eg = (lam * g).sum()
        dU = c ** (-s) * gU - s * Eg / c * cU; dW = c ** (-s) * gW - s * Eg / c * cW
        mo = self.mo
        sU = -self.Aown[:, None, None] * np.sin(2 * np.pi * a_[None, None, :] / p + ph[:, :, None])
        sW = -self.Aown[:, None] * np.sin(2 * np.pi * a_[None, :] / p + psi[:, None])
        dph = (dU[:mo] * sU).sum(2) + (dW[:mo] * sW).sum(1)[:, None]
        gr = dph.ravel()
        if self.g: gr = np.concatenate([gr, self.g.back(dU[mo:], dW[mo:])])
        return Fv, gr

    def opt(self, x, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5), iters=1500):
        sc = H.Bp(p, n) * nu ** (1 - s)
        for tau in taus:
            res = minimize(lambda z: tuple(-v for v in self.soft(z, tau * sc)), x, jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-13})
            x = res.x
        return x


def gain_of(w):
    w = np.asarray(w, float); w = w / w.sum(); return 1.0 / (nu ** (1 - s) * np.sum(w ** r) ** s)


def vec(fs, amp):
    v = np.zeros(p)
    for k in fs:
        v += amp * (rng.standard_normal() if k == 0 else np.cos(2 * np.pi * k * a_ / p + rng.uniform(0, 2 * np.pi)))
    return v


profiles = [('w2', [1.0, 1.0])] + [(f'c{c}', [1.0] * (nu - 1) + [c]) for c in (0.3, 0.6, 0.85)] + \
           [('w%d' % (nu - 1), [1.0] * (nu - 1))] * (1 if nu - 1 > 2 else 0)
t0 = time.time()
print(f"# b2 p={p} n={n} b={bexp}: D = nu^(1-s) B_p kappa_min", flush=True)
for name, prof in profiles:
    mass = np.array(prof) / sum(prof)                           # |b_i| shares
    Aown = (2 ** n * mass) ** (1.0 / nu)                         # |b| = A^nu 2^{-n}
    G = gain_of(mass)
    best_own = -np.inf; best_fr = -np.inf; best_rec = None
    for rr in range(nrest):
        mo = len(prof)
        ph0 = (np.pi * np.arange(mo)[:, None] / mo + 0.6 * rng.standard_normal((mo, n)) * (rr % 3 != 0)) * np.ones((1, n))
        # own only
        co = Combo(Aown, None)
        xo = co.opt(ph0.ravel()); vo = co.gamma(xo) / D
        best_own = max(best_own, vo)
        # with foreign neurons
        nf = int(rng.integers(1, 2 * nu + 2)); amp = float(rng.choice([0.05, 0.2, 0.5])) * Aown.mean()
        al = []; Us = []; Ws = []; kinds = []
        for q in range(nf):
            u = rng.random(); T = [t for t in range(n) if rng.random() < 0.5] or [int(rng.integers(0, n))]
            if u < 0.5 and foreign:
                j = int(rng.choice(foreign)); fs = [{red(2 - j)} if t in T else {j} for t in range(n)]; out = {j}
                kinds.append(f'chan{j}T{T}')
            elif u < 0.7:
                fs = [{red(3)} if t in T else {1} for t in range(n)]; out = {1}; kinds.append(f'part{T}')
            elif u < 0.85:
                fs = [{0} | set(rng.choice(sorted(ALL), size=1).tolist()) for _ in range(n)]
                out = set(rng.choice(sorted(ALL), size=2).tolist()); kinds.append('dc')
            else:
                fs = [ALL] * n; out = ALL; kinds.append('rand')
            Us.append(np.array([vec(f, amp) for f in fs])); Ws.append(vec(out, amp))
            al.append(struct_allowed([ALL] * (n + 1)))
        cf = Combo(Aown, al)
        x = np.concatenate([xo, cf.g.from_UW(np.array(Us), np.array(Ws))])
        x = cf.opt(x); v = cf.gamma(x) / D
        U, W, _, _ = cf.UW(x); nt = GL.Net(U, W, bexp)
        fe = sum(float(np.sum((nt.W @ GL.freq_proj(p, k)) ** 2)) for k in foreign) / float(np.sum(nt.W ** 2))
        if v > best_fr:
            best_fr = v; best_rec = dict(kinds=kinds, foreign_out_energy=fe,
                                         rel_beta=float(np.abs(nt.clean()[1:]).max() / abs(nt.clean()[0])) if F > 1 else 0.)
        if v > 1 + 1e-9:
            np.savez(os.path.join(OUT, f"VIOL_b2_p{p}_n{n}_b{bexp}_{name}_{rr}.npz"), U=U, W=W, ratio=v, p=p, n=n, b=bexp)
    print(json.dumps(dict(profile=name, gain=G, own_only=best_own, with_foreign=best_fr, lift=best_fr - best_own,
                          **(best_rec or {})), default=float), flush=True)
print(f"# done {time.time() - t0:.0f}s", flush=True)
