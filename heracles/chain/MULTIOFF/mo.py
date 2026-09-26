"""Multi-offset aiming for single-frequency networks (n = 2, q = 2).

Conventions (theory_modadd / chain/common.py): candidate s - D; the junk part of the gap on row (a, b) at offset D is
    junk_D(a,b) = Re sum_{k in T} (1 - w^{kD}) [J1_k w^{-2ka} + J2_k w^{-2kb} + J12_k w^{-2k(a+b)}]      (x 2/p^3, dropped)
(VERIFY V3a.2).  dip_D(J) = max over rows of (-junk_D).  For an offset distribution lam (symmetric in D <-> -D) with
INDEPENDENT row choices per offset, the best test distribution gives exactly
    N_lam(J) = sum_D lam(D) dip_D(J)                          (exposure)
while the clean part averages to sum_k W_k Re G_k,  W_k = sum_D lam(D) (1 - cos(2 pi k D / p)).
Multi-offset depth:  A(T, lam) = min_J N_lam(J) / sum_k W_k |J_k|_2   (eta-form units; |J_k|_2 = l2 over the 3 patterns).
Search = UPPER estimate of A (explicit bad J).
"""
import itertools, math, numpy as np, torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def cov(p, k, D):
    return 1 - np.cos(2 * np.pi * np.outer(k, D) / p)   # (K, nD)


def weights(p, T, Ds, lam):
    return cov(p, np.array(T), np.array(Ds)) @ np.array(lam)   # W_k


class Body:
    """Holds (p, T, offsets Ds, weights lam, mode) and evaluates exposure / depth ratio for junk J."""

    def __init__(self, p, T, Ds, lam, mode='3'):
        self.p, self.T, self.Ds, self.lam, self.mode = p, np.array(T), np.array(Ds), np.array(lam, float), mode
        K = len(T); x = np.arange(p)
        self.chi = torch.tensor(np.exp(-2j * np.pi * 2 * np.outer(T, x) / p))          # (K, p): w^{-2kx}
        self.tw = torch.tensor(1 - np.exp(2j * np.pi * np.outer(T, Ds) / p))           # (K, nD): 1 - w^{kD}
        self.W = torch.tensor(weights(p, T, Ds, lam))
        self.lamt = torch.tensor(self.lam)
        self.ab = torch.tensor((x[:, None] + x[None, :]) % p)

    def rowfuncs(self, J):
        # J: complex tensor (3, K) [pattern 1, 2, 12]; returns f_T(D, x) = Re sum_k -(1 - w^{kD}) J_Tk w^{-2kx}
        c = -(J[:, :, None] * self.tw[None])                  # (3, K, nD)
        return torch.real(torch.einsum('tkd,kx->tdx', c, self.chi))   # (3, nD, p)

    def dips(self, J, beta=None):
        f = self.rowfuncs(J)
        if self.mode == '1':            # single-pattern junk (pattern 1 only): rows are 1-D
            F = f[0]                    # (nD, p)
            return F.max(1).values if beta is None else torch.logsumexp(beta * F, 1) / beta
        F = f[0][:, :, None] + f[1][:, None, :] + f[2][:, self.ab]    # (nD, p, p)
        F = F.reshape(F.shape[0], -1)
        return F.max(1).values if beta is None else torch.logsumexp(beta * F, 1) / beta

    def denom(self, J):
        return (self.W * torch.sqrt((torch.abs(J) ** 2).sum(0))).sum()

    def ratio(self, J, beta=None):
        return (self.lamt * self.dips(J, beta)).sum() / self.denom(J)


def to_J(v, K, mode):
    v = torch.as_tensor(v)
    if mode == '1':
        z = torch.complex(v[:K], v[K:])
        return torch.stack([z, torch.zeros_like(z), torch.zeros_like(z)])
    if mode == 'lone':   # lone-atom shape: J_T = c_k * prod_{j in T} v_jk, |v| = 1
        c = torch.complex(v[:K], v[K:2 * K]); v1 = torch.exp(1j * v[2 * K:3 * K]); v2 = torch.exp(1j * v[3 * K:4 * K])
        return torch.stack([c * v1, c * v2, c * v1 * v2])
    z = torch.complex(v[:3 * K], v[3 * K:]).reshape(3, K)
    return z


def nparams(K, mode):
    return {'1': 2 * K, 'lone': 4 * K, '3': 6 * K}[mode]


def structured_starts(p, T, mode, rng):
    K = len(T); T = np.array(T)
    leg = np.array([1.0 if pow(int(k), (p - 1) // 2, p) == 1 else -1.0 for k in T])
    phs = {'ones': np.zeros(K), 'alt': np.pi * (np.arange(K) % 2), 'chirp': np.pi * np.arange(K) ** 2 / max(K, 1),
           'leg': np.where(leg > 0, 0.0, np.pi), 'legi': np.where(leg > 0, 0.5, 1.5) * np.pi,
           'rand': rng.uniform(0, 2 * np.pi, K)}
    out = []
    for name, ph in phs.items():
        z = np.exp(1j * ph)
        if mode == '1':
            out.append(np.r_[z.real, z.imag])
        elif mode == 'lone':
            out.append(np.r_[z.real, z.imag, rng.uniform(0, 2 * np.pi, 2 * K)])
            out.append(np.r_[z.real, z.imag, np.zeros(2 * K)])
        else:
            for t in range(3):
                Z = np.zeros((3, K), complex); Z[t] = z
                out.append(np.r_[Z.real.ravel(), Z.imag.ravel()])
    return out


def depth_search(body, mode, rng, starts=None, extra=6, steps=400, betas=(20., 80., 400.)):
    """Upper estimate of min_J ratio.  Adam on a softmax-smoothed exposure with annealed temperature, exact re-evaluation."""
    K = len(body.T)
    if starts is None:
        starts = structured_starts(body.p, body.T, mode, rng)
    starts = list(starts) + [rng.normal(size=nparams(K, mode)) for _ in range(extra)]
    best, bestv = np.inf, None
    for s in starts:
        v = torch.tensor(np.array(s, float), requires_grad=True)
        opt = torch.optim.Adam([v], lr=0.03)
        for beta in betas:
            for _ in range(steps):
                opt.zero_grad()
                J = to_J(v, K, mode)
                scale = torch.sqrt((torch.abs(J) ** 2).sum()) + 1e-12
                loss = body.ratio(J / scale, beta * float(K))
                loss.backward(); opt.step()
        with torch.no_grad():
            r = float(body.ratio(to_J(v, K, mode)))
        if r < best:
            best, bestv = r, v.detach().numpy().copy()
    return best, bestv


# ---------- the adversary's spectrum on T: floored share programme, binding offsets, dual weights ----------
from scipy.optimize import minimize, nnls


def floored_spectrum(p, T, eps, rng, starts=6, pool=None):
    """Adversary's effective spectrum: junk on T (every k in T carries share >= eps), clean helpers allowed on `pool`.
    max_b min_D sum_k cov_k(D) b_k^{3/2}, sum b = 1.  Returns (value, b over T+pool, binding Ds, coverage, support list)."""
    h = (p - 1) // 2
    S = sorted(set(T) | set(pool or []))
    Dall = np.arange(1, h + 1)
    M = cov(p, np.array(S), Dall).T            # (h, |S|)
    n = len(S); isT = np.array([k in set(T) for k in S])
    lo = np.where(isT, eps, 0.0)
    best = (-1, None)
    for s in range(starts):
        b0 = np.ones(n) / n if s == 0 else rng.dirichlet(np.ones(n) * 0.5)
        b0 = np.maximum(b0, lo); b0 /= b0.sum()
        cons = [{'type': 'eq', 'fun': lambda v: v[:n].sum() - 1},
                {'type': 'ineq', 'fun': lambda v: M @ np.clip(v[:n], 0, None) ** 1.5 - v[n]}]
        r = minimize(lambda v: -v[n], np.r_[b0, (M @ b0 ** 1.5).min()], constraints=cons,
                     bounds=[(l, 1) for l in lo] + [(None, None)], method='SLSQP', options=dict(maxiter=1500, ftol=1e-14))
        b = np.maximum(np.clip(r.x[:n], 0, None), lo); b /= b.sum(); v = (M @ b ** 1.5).min()
        if v > best[0]:
            best = (v, b)
    v, b = best
    C = M @ b ** 1.5
    bind = Dall[C <= v * (1 + 1e-5)]
    return v, b, bind, C, S


def dual_weights(p, T, b, bind, eps):
    """KKT dual on the binding offsets: sum_D lam_D cov_k(D) (3/2) b_k^{1/2} = mu for interior k (b_k > eps)."""
    K = len(T); inter = b > eps * (1 + 1e-6)
    A = cov(p, np.array(T), bind) * (1.5 * np.sqrt(b))[:, None]        # (K, nb)
    if inter.sum() == 0:
        return np.ones(len(bind)) / len(bind)
    # find lam >= 0, sum lam = 1 with A[inter] lam = const: minimise ||A[inter] lam - mu 1|| over lam, mu
    Ai = A[inter]
    best = None
    for mu in np.linspace(0.2, 3, 57):
        lam, res = nnls(np.vstack([Ai, 50 * np.ones(len(bind))]), np.r_[mu * np.ones(Ai.shape[0]), 50.0])
        if best is None or res < best[0]:
            best = (res, lam)
    lam = best[1]; lam /= lam.sum()
    return lam


def sym(Ds, lam, p):
    """symmetrise offsets D -> {D, p - D} with half weight each."""
    D2 = np.r_[Ds, p - np.array(Ds)]; l2 = np.r_[lam, lam] / 2
    return D2, l2
