"""p3a_prof.py -- the PROFILE parametrisation of a pure gated-chain network, and the
two per-frequency cost functions that price a profile.

WHY A NEW PARAMETRISATION.  Agents 1-4 worked with ATOMS (one complex c_l plus n unit
phases v_lj per neuron).  For the junk question at K >= 2 frequencies the natural
coordinates are instead the FOURIER PROFILE of each frequency:

    C_T^{(k)},  T subset of {1..n},     C_empty^{(k)} = x_k >= 0  (the CLEAN amplitude),
                                        C_T^{(k)}, T != empty      (the JUNK).

Two facts make this exact (both inherited, both re-verified here):

 (a) MARGIN depends on the network only through the profile.  For pure neurons
     (GLOBAL_OPT sec 2, extended to several frequencies),

        2^n * margin = Lambda := min over a in Z_p^n and D != 0 of
                       sum_k Re[ (1 - e^{2 pi i k D / p}) h_k(a) ],
        h_k(a) = sum_T C_T^{(k)} omega^{-2 k sigma_T(a)},   sigma_T(a) = sum_{j in T} a_j,
        omega = e^{2 pi i / p}.

     Plain reading: neuron l of frequency k emits the clean cosine plus 2^n - 1 "junk"
     cosines, one per sign pattern T; C_T^{(k)} is the total amplitude the whole
     population puts on pattern T, and nothing else about the population is visible in
     the logit table.

 (b) COST depends on the network only through the MODULI |C_T^{(k)}|.  Define

        Cprof_r(m) := min { sum_l |c_l|^r : C_empty = 1, |C_T| = m_T for all T != empty },
        Cl2_r(eta) := min { sum_l |c_l|^r : C_empty = 1, sum_{T != empty} |C_T|^2 <= eta^2 }

     (any number of atoms, all unit phases free; r = q/nu).  The per-slot gauge
     v_lj -> e^{i psi_j} v_lj leaves every |c_l| fixed and rotates C_T by
     exp(i sum_{j in T} psi_j), so the cost of a profile really is a function of the
     moduli alone (GLOBAL_OPT4 sec 5.2).  Cl2_r(eta) = min over profiles with l_2 junk
     norm eta of Cprof_r, so Cl2_r <= Cprof_r: the eta-only cost is a RELAXATION.

Consequently, for pure networks, with  P_q = ((n+1)p/2)^{q/2} sum_{k,l}|c_lk|^r  and
B = 2(2p(n+1))^{-nu/2},

    gamma_q / B  =  Lambda / ( sum_{k,l} |c_lk|^r )^{1/r}
                 =  Lambda / ( sum_k Cprof_r(m^{(k)}) x_k^r )^{1/r}      (EXACT)
                <=  Lambda / ( sum_k Cl2_r(eta_k)   x_k^r )^{1/r}       (the eta-only relaxation)

and the target (T) is  gamma_q/B <= (n+1)^{-(1-r)/r} Gamma_{nu/q}(p).

This module provides: the profile -> Lambda machinery, Cl2_r on a grid, and Cprof_r.

All float64, local CPU, no pods.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "FINAL_RUNS" / "P3A"
EPS2 = 1e-30


# --------------------------------------------------------------------- subsets
def subsets(n, nonempty=False):
    out = []
    for k in range(1 if nonempty else 0, n + 1):
        for T in itertools.combinations(range(n), k):
            out.append(T)
    return out


# --------------------------------------------------------------------- the programme
def gmat(p):
    """g[k-1, D-1] = 1 - cos(2 pi k D / p), k = 1..(p-1)/2, D = 1..p-1."""
    k = np.arange(1, (p - 1) // 2 + 1)
    d = np.arange(1, p)
    return 1.0 - np.cos(2 * np.pi * np.outer(k, d) / p)


def gamma_s(p, r, starts=400, seed=0):
    """Gamma_{1/r}(p) = max { min_D sum_k x_k g_k(D) : x >= 0, sum_k x_k^r <= 1 }."""
    g = gmat(p)
    nf = g.shape[0]
    rng = np.random.default_rng(seed)

    def negval(z):
        x = z ** 2
        s = np.sum(x ** r)
        if s <= 0:
            return 0.0
        return -float((g.T @ (x / s ** (1.0 / r))).min())

    best, bestx = 0.0, None
    for s in range(starts):
        if s < nf:
            z0 = np.zeros(nf)
            z0[s] = 1.0
        else:
            K = rng.integers(1, nf + 1)
            z0 = np.zeros(nf)
            z0[rng.choice(nf, K, replace=False)] = 1.0
            z0 = z0 * rng.uniform(0.5, 1.5, nf)
        res = minimize(negval, z0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
        res = minimize(negval, res.x, method='Powell',
                       options={'maxiter': 20000, 'xtol': 1e-13, 'ftol': 1e-15})
        if -res.fun > best:
            x = res.x ** 2
            best, bestx = -res.fun, x / np.sum(x ** r) ** (1.0 / r)
    return best, bestx


# --------------------------------------------------------------------- profile -> Lambda
class Profile:
    """Geometry for one (p, n, list of frequencies).  Holds

        PH[k, t, a] = omega^{-2 k sigma_{T_t}(a)}      (a runs over all p^n rows)
        W[k, D]     = 1 - e^{2 pi i k D / p}

    so that, with a profile  C[k, t]  (t = 0 is T = empty, i.e. the clean amplitude),

        H[k, a]   = sum_t C[k, t] PH[k, t, a]
        M[a, D]   = sum_k Re( W[k, D] H[k, a] )
        Lambda    = min over (a, D) of M.
    """

    def __init__(self, p, n, ks):
        self.p, self.n = p, n
        self.ks = np.asarray(ks, dtype=np.int64)
        self.K = len(self.ks)
        self.Ts = subsets(n)
        self.nT = len(self.Ts)
        r = np.arange(p ** n)
        ops = np.stack([(r // p ** j) % p for j in range(n)], 1)       # (p^n, n)
        self.ops = ops
        sig = np.stack([ops[:, list(T)].sum(1) % p if T else np.zeros(p ** n, np.int64)
                        for T in self.Ts], 0)                          # (nT, p^n)
        ang = -2.0 * 2 * np.pi * np.einsum('k,ta->kta', self.ks, sig) / p
        self.PH = np.exp(1j * ang)                                     # (K, nT, p^n)
        D = np.arange(1, p)
        self.W = 1.0 - np.exp(1j * 2 * np.pi * np.outer(self.ks, D) / p)  # (K, p-1)
        self.g = 1.0 - np.cos(2 * np.pi * np.outer(self.ks, D) / p)       # (K, p-1)

    def H(self, C):
        return np.einsum('kt,kta->ka', C, self.PH)

    def M(self, C):
        return np.einsum('kd,ka->ad', self.W, self.H(C)).real

    def Lam(self, C):
        return float(self.M(C).min())


# --------------------------------------------------------------------- Cl2_r and Cprof_r
def _unpack(z, N, n):
    return z[:N], z[N:2 * N], z[2 * N:].reshape(N, n)


def _moments(z, N, n, Ts):
    cr, ci, phi = _unpack(z, N, n)
    c = cr + 1j * ci
    S = np.zeros((N, len(Ts)))
    for a, T in enumerate(Ts):
        if T:
            S[:, a] = phi[:, list(T)].sum(axis=1)
    E = np.exp(1j * S)
    return c, E, (c[:, None] * E).sum(axis=0)


def _cost(z, N, n, r):
    cr, ci, _ = _unpack(z, N, n)
    return float(np.sum((cr ** 2 + ci ** 2 + EPS2) ** (r / 2.0)))


def _cost_grad(z, N, n, r):
    cr, ci, _ = _unpack(z, N, n)
    d = r * (cr ** 2 + ci ** 2 + EPS2) ** (r / 2.0 - 1.0)
    g = np.zeros_like(z)
    g[:N] = d * cr
    g[N:2 * N] = d * ci
    return g


def _dabs2(z, N, n, Ts, a):
    c, E, C = _moments(z, N, n, Ts)
    g = np.zeros(z.size)
    e, CT = E[:, a], C[a]
    g[:N] = 2.0 * (np.conj(CT) * e).real
    g[N:2 * N] = 2.0 * (np.conj(CT) * 1j * e).real
    for j in Ts[a]:
        g[2 * N + np.arange(N) * n + j] = 2.0 * (np.conj(CT) * (1j * c * e)).real
    return g


def _start(N, n, rng, kind, scale=0.0):
    nb = n + 1
    cr, ci = np.zeros(N), np.zeros(N)
    phi = rng.uniform(0, 2 * np.pi, size=(N, n))
    if kind == 'block':
        cr[:nb] = 1.0 / nb
        for l in range(nb):
            phi[l, :] = -2.0 * np.pi * l / nb
    else:
        cr[:] = rng.standard_normal(N) / N
        ci[:] = rng.standard_normal(N) / N
    z = np.concatenate([cr, ci, phi.ravel()])
    return z + scale * rng.standard_normal(z.size) if scale else z


def cl2(n, r, eta, Ns=None, starts=24, rng=None):
    """Cl2_r(eta): cheapest feature with l_2 junk norm <= eta and clean amplitude 1."""
    Ts = subsets(n)
    rng = rng or np.random.default_rng(0)
    Ns = Ns or (n + 1, n + 2)
    best = np.inf
    for N in Ns:
        cons = [{'type': 'eq',
                 'fun': lambda z, N=N: np.array([_moments(z, N, n, Ts)[2][0].real - 1.0,
                                                 _moments(z, N, n, Ts)[2][0].imag])},
                {'type': 'ineq',
                 'fun': lambda z, N=N: np.array(
                     [eta ** 2 - np.sum(np.abs(_moments(z, N, n, Ts)[2][1:]) ** 2)]),
                 'jac': lambda z, N=N: np.array(
                     [-sum(_dabs2(z, N, n, Ts, a) for a in range(1, len(Ts)))])}]
        for s in range(starts):
            z0 = (_start(N, n, rng, 'block') if s == 0 else
                  _start(N, n, rng, 'block', 10.0 ** (-s)) if s < 6 else
                  _start(N, n, rng, 'rand'))
            try:
                res = minimize(lambda z: _cost(z, N, n, r), z0,
                               jac=lambda z: _cost_grad(z, N, n, r),
                               constraints=cons, method='SLSQP',
                               options={'maxiter': 800, 'ftol': 1e-16})
            except Exception:
                continue
            if not res.success:
                continue
            C = _moments(res.x, N, n, Ts)[2]
            if abs(C[0] - 1.0) > 1e-9 or np.sum(np.abs(C[1:]) ** 2) > eta ** 2 + 1e-10:
                continue
            best = min(best, _cost(res.x, N, n, r))
    return best


def cprof(n, r, m, Ns=None, starts=16, rng=None):
    """Cprof_r(m): cheapest feature with clean amplitude 1 and |C_T| = m_T exactly.

    m is indexed by subsets(n, nonempty=True) in that order.  Returns +inf if the
    profile is not realisable (it always is, for enough atoms).
    """
    Ts = subsets(n)
    rng = rng or np.random.default_rng(0)
    Ns = Ns or (n + 1, n + 2, n + 3)
    m = np.asarray(m, dtype=float)
    best = np.inf
    for N in Ns:
        def eqf(z, N=N):
            C = _moments(z, N, n, Ts)[2]
            return np.concatenate([[C[0].real - 1.0, C[0].imag],
                                   np.abs(C[1:]) ** 2 - m ** 2])

        def eqj(z, N=N):
            _, E, _ = _moments(z, N, n, Ts)
            J = np.zeros((2 + len(m), z.size))
            e = E[:, 0]
            J[0, :N] = e.real
            J[0, N:2 * N] = -e.imag
            J[1, :N] = e.imag
            J[1, N:2 * N] = e.real
            for a in range(1, len(Ts)):
                J[1 + a] = _dabs2(z, N, n, Ts, a)
            return J

        cons = [{'type': 'eq', 'fun': eqf, 'jac': eqj}]
        for s in range(starts):
            z0 = (_start(N, n, rng, 'block', 0.3) if s < 4 else
                  _start(N, n, rng, 'rand'))
            try:
                res = minimize(lambda z: _cost(z, N, n, r), z0,
                               jac=lambda z: _cost_grad(z, N, n, r),
                               constraints=cons, method='SLSQP',
                               options={'maxiter': 900, 'ftol': 1e-16})
            except Exception:
                continue
            if np.max(np.abs(eqf(res.x))) > 1e-9:
                continue
            best = min(best, _cost(res.x, N, n, r))
    return best


def cl2_grid(n, q, npts=41, starts=24, seed=1, cache=True):
    """Cl2_r on a grid of eta in [0, sqrt(2^n-1)], monotone-enforced."""
    r = q / (n + 1)
    emax = float(np.sqrt(2 ** n - 1))
    fn = OUT / f"cl2_n{n}_q{q}_{npts}.json"
    if cache and fn.exists():
        d = json.loads(fn.read_text())
        return np.array(d['eta']), np.array(d['C'])
    etas = np.linspace(0.0, emax, npts)
    rng = np.random.default_rng(seed)
    vals = [(n + 1) ** (1 - r)]
    for e in etas[1:]:
        vals.append(cl2(n, r, float(e), starts=starts, rng=rng))
    vals = np.minimum.accumulate(np.array(vals))
    vals = np.maximum(vals, 1.0)          # Cl2_r >= |C_empty|^r = 1 always
    OUT.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps({'n': n, 'q': q, 'r': r,
                              'eta': etas.tolist(), 'C': vals.tolist()}, indent=1))
    return etas, vals
