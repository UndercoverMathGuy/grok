"""VERIFY links 1, 3, 5: the network-level chain, re-derived and re-checked from scratch.

Nothing is imported from the authors' scripts; every quantity is recomputed by brute force.

SETTING (every symbol defined).
  p prime, n operands, nu = n+1.  Gated chain:
      z(a_1..a_n, c) = sum_i W[c,i] * prod_j E[j,i,a_j],     E: (n,m,p),  W: (p,m)
  margin  = min over all p^n rows a and all wrong classes c != s(a) of  z(a,s(a)) - z(a,c),
            s(a) = (a_1+...+a_n) mod p.
  ||theta_i||^2 = sum_j ||E[j,i,:]||^2 + ||W[:,i]||^2 ;  P_q = sum_i ||theta_i||^q.
  gamma_q = margin / P_q^{nu/q}.
  ehat_ij(k) = sum_a E[j,i,a] e^{-2 pi i k a/p}   (unnormalised DFT), same for what_i(k).
  c_ik = prod_j ehat_ij(k) * conj(what_i(k))        (neuron i's ATOM at frequency k)
  v_ijk = conj(ehat_ij(k)) / ehat_ij(k)             (its phase datum, unit modulus)
  G_k   = sum_i c_ik                                (clean Fourier coefficient)
  J_{k,T} = sum_i c_ik prod_{j in T} v_ijk          (same-|k| junk, T nonempty subset of {1..n})
  H = {1..(p-1)/2}  positive frequencies;  g_k(D) = 1 - cos(2 pi k D / p)
  B = 2 (2 p nu)^{-nu/2};   m_ik = sum_j |ehat_ij(k)|^2 + |what_i(k)|^2
"""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260921)


# --------------------------------------------------------------- brute force
def logits(E, W, p, n):
    m = E.shape[1]
    idx = np.array(list(itertools.product(range(p), repeat=n)))
    h = np.ones((idx.shape[0], m))
    for j in range(n):
        h *= E[j][:, idx[:, j]].T
    return h @ W.T, idx


def margin_of(E, W, p, n):
    z, idx = logits(E, W, p, n)
    s = idx.sum(1) % p
    R = np.arange(z.shape[0])
    zc = z[R, s]
    zz = z.copy()
    zz[R, s] = -np.inf                      # exclude the correct class
    return float(np.min(zc - zz.max(1)))


def Pq(E, W, q):
    nrm2 = (E ** 2).sum(axis=(0, 2)) + (W ** 2).sum(axis=0)
    return float(np.sum(nrm2 ** (q / 2.0)))


def rebalance(E, W, n):
    """Rescale each neuron's n+1 factors at fixed product (a free operation: the logit function is
       unchanged) so that all n+1 norms are equal -- this minimises ||theta_i||^2."""
    E = E.copy(); W = W.copy()
    m = E.shape[1]
    for i in range(m):
        norms = [np.linalg.norm(E[j, i]) for j in range(n)] + [np.linalg.norm(W[:, i])]
        if min(norms) < 1e-300:
            continue
        g = float(np.prod(norms) ** (1.0 / (n + 1)))
        for j in range(n):
            E[j, i] *= g / norms[j]
        W[:, i] *= g / norms[n]
    return E, W


def fourier(E, W, p, n):
    w = np.exp(-2j * np.pi * np.outer(np.arange(p), np.arange(p)) / p)
    Eh = np.einsum('ka,jia->jik', w, E)
    Wh = np.einsum('ka,ai->ik', w, W)
    return Eh, Wh


def clean_and_junk(Eh, Wh, p, n):
    nf = (p - 1) // 2
    G = np.zeros(nf + 1, dtype=complex)
    J = {}
    for k in range(1, nf + 1):
        prod = np.prod(Eh[:, :, k], axis=0)
        c = prod * np.conj(Wh[:, k])
        den = Eh[:, :, k]
        v = np.conj(den) / np.where(np.abs(den) > 1e-300, den, 1.0)
        G[k] = c.sum()
        for size in range(1, n + 1):
            for T in itertools.combinations(range(n), size):
                J[(k, T)] = complex(np.sum(c * np.prod(v[list(T)], axis=0)))
    return G, J


def clean_g(E, W, p, n):
    z, idx = logits(E, W, p, n)
    s = idx.sum(1) % p
    return np.array([z[np.arange(z.shape[0]), (s - D) % p].mean() for D in range(p)])


# --------------------------------------------------------------- Gamma_s(p)
def Gamma_s(p, s, restarts=200, seed=0):
    """max over x >= 0 with sum_k x_k^{1/s} <= 1 of min_{D != 0} sum_k x_k g_k(D).
       Substituting y_k = x_k^{1/s} (so y on the simplex) this is
          max_y  min_D  sum_k y_k^s g_k(D),   y >= 0, sum y <= 1,
       solved as the smooth NLP  max t  s.t.  sum_k y_k^s g_k(D) >= t for every D != 0."""
    nf = (p - 1) // 2
    ks = np.arange(1, nf + 1)
    Dm = np.arange(1, p)
    Gm = 1 - np.cos(2 * np.pi * np.outer(ks, Dm) / p)        # (nf, p-1)
    rr = np.random.default_rng(seed)

    best, bestx = -1.0, None
    # exhaustive equal-amplitude ladder (exact feasible floor)
    for K in range(1, nf + 1):
        for S in itertools.combinations(range(nf), K):
            val = Gm[list(S)].sum(0).min() * (K ** (-s))
            if val > best:
                best, bestx = val, (np.eye(nf)[list(S)].sum(0) / K)
    # free-amplitude NLP
    def obj(z):
        return -z[-1]
    def cons_f(z):
        y = z[:-1]
        return np.concatenate([(np.abs(y) ** s) @ Gm - z[-1], [1.0 - np.abs(y).sum()]])
    for t in range(restarts):
        if t == 0:
            y0 = bestx.copy()
        else:
            K = int(rr.integers(1, nf + 1))
            S = rr.choice(nf, size=K, replace=False)
            y0 = np.zeros(nf); y0[S] = rr.random(K); y0 /= y0.sum()
        z0 = np.concatenate([y0, [float((np.abs(y0) ** s) @ Gm).min() if False else float(((np.abs(y0) ** s) @ Gm).min())]])
        res = minimize(obj, z0, method="SLSQP",
                       constraints=[dict(type="ineq", fun=cons_f)],
                       bounds=[(0, 1)] * nf + [(None, None)],
                       options=dict(maxiter=500, ftol=1e-14))
        y = np.abs(res.x[:-1])
        ssum = y.sum()
        if ssum > 1:
            y = y / ssum
        v = float(((y ** s) @ Gm).min())
        if v > best:
            best, bestx = v, y
    return best, bestx


# --------------------------------------------------------------- symmetrisation
def gauge_symmetrise(E, W, p, n):
    """Average the network over the n+1 gauge phases phi_l = pi l/(n+1).  Rotating every
       positive-frequency Fourier coefficient of each embedding by e^{i phi} and of the readout by
       e^{i n phi} is an orthogonal map on each real vector, so it preserves P_q and the clean
       part; the average over l kills every same-|k| junk coefficient."""
    nu = n + 1
    m = E.shape[1]
    nf = (p - 1) // 2
    En = np.zeros((n, nu * m, p)); Wn = np.zeros((p, nu * m))
    Eh, Wh = fourier(E, W, p, n)
    for l in range(nu):
        phi = np.pi * l / nu
        Ehl = Eh.copy(); Whl = Wh.copy()
        for k in range(1, nf + 1):
            Ehl[:, :, k] *= np.exp(1j * phi); Ehl[:, :, p - k] *= np.exp(-1j * phi)
            Whl[:, k] *= np.exp(1j * n * phi); Whl[:, p - k] *= np.exp(-1j * n * phi)
        Er = np.real(np.fft.ifft(Ehl, axis=2)) * p / p
        Wr = np.real(np.fft.ifft(Whl, axis=1)).T
        En[:, l * m:(l + 1) * m, :] = Er
        Wn[:, l * m:(l + 1) * m] = Wr / nu
    return En, Wn


# --------------------------------------------------------------- checks
def run():
    rep = {}
    print("=" * 104)
    print("V4  the B identity  2 p^{-nu} (p/(2 nu))^{nu/2} = 2 (2 p nu)^{-nu/2}")
    worst = max(abs(2 * p ** (-(n + 1)) * (p / (2.0 * (n + 1))) ** ((n + 1) / 2.0)
                    / (2 * (2.0 * p * (n + 1)) ** (-(n + 1) / 2.0)) - 1)
                for p in [5, 7, 11, 13, 23, 113] for n in [1, 2, 3, 4])
    print(f"    max relative error: {worst:.3e}  -> {'VERIFIED' if worst < 1e-12 else 'FAILS'}")
    rep["V4"] = worst

    print("=" * 104)
    print("V1/V2/V3  dictionary, margin bound (L1+L2), per-neuron budget (2.2)")
    print(f"{'p':>4}{'n':>3}{'m':>4} | {'V1 dict relerr':>15} {'V2 margin/bound':>16} "
          f"{'V3 budget q=2':>14} {'q=2.5':>9} {'q=nu':>9}")
    v1w = 0.0; v2w = -np.inf; v3w = 0.0
    for (p, n, m) in [(5, 2, 3), (7, 2, 4), (11, 2, 5), (13, 2, 3), (5, 3, 4), (7, 3, 3), (5, 4, 3)]:
        e1 = 0.0; mb = -np.inf; bud = [0.0, 0.0, 0.0]
        for trial in range(5):
            E = rng.standard_normal((n, m, p)); W = rng.standard_normal((p, m))
            if trial == 1:   # adversarial: one balanced neuron carrying every frequency
                E = rng.standard_normal((n, 1, p)); W = rng.standard_normal((p, 1))
                E, W = rebalance(E, W, n)
            Eh, Wh = fourier(E, W, p, n)
            G, _ = clean_and_junk(Eh, Wh, p, n)
            nf = (p - 1) // 2
            g = clean_g(E, W, p, n)
            ghat = np.array([np.sum(g * np.exp(-2j * np.pi * k * np.arange(p) / p)) for k in range(p)])
            pred = np.array([G[k] * p ** (-n) for k in range(1, nf + 1)])
            e1 = max(e1, float(np.max(np.abs(ghat[1:nf + 1] - pred) / (np.abs(pred) + 1e-300))))
            mar = margin_of(E, W, p, n)
            ks = np.arange(1, nf + 1); Dm = np.arange(1, p)
            Gm = 1 - np.cos(2 * np.pi * np.outer(ks, Dm) / p)
            bound = 2 * p ** (-(n + 1)) * float((np.abs(G[1:nf + 1]) @ Gm).min())
            if bound > 0:
                mb = max(mb, mar / bound)
            for qi, q in enumerate([2.0, 2.5, float(n + 1)]):
                r = q / (n + 1)
                t = np.abs(np.prod(Eh[:, :, 1:nf + 1], axis=0) * np.conj(Wh[:, 1:nf + 1]))
                lhs = (t ** r).sum(1)
                nrm2 = (E ** 2).sum(axis=(0, 2)) + (W ** 2).sum(axis=0)
                rhs = (p / (2.0 * (n + 1))) ** (q / 2.0) * nrm2 ** (q / 2.0)
                bud[qi] = max(bud[qi], float(np.max(lhs / rhs)))
        v1w = max(v1w, e1); v2w = max(v2w, mb); v3w = max(v3w, max(bud))
        print(f"{p:>4}{n:>3}{m:>4} | {e1:>15.3e} {mb:>16.9f} {bud[0]:>14.9f} {bud[1]:>9.6f} {bud[2]:>9.6f}")
    print(f"  worst: V1 relerr {v1w:.3e} (want ~1e-14) | V2 margin/bound {v2w:.9f} (want <= 1) "
          f"| V3 budget {v3w:.9f} (want <= 1)")
    rep.update(V1=v1w, V2=v2w, V3=v3w)

    print("=" * 104)
    print("V5/V6  Theorem D end to end on (H)-networks (gauge-symmetrised: impure, width (n+1)m)")
    print(f"{'p':>4}{'n':>3}{'q':>5} | {'max same-|k| junk':>18} {'Gamma_s':>12} "
          f"{'max gamma/(F*tgt)':>18} {'max gamma/tgt (F=1 CLAIM)':>26}")
    rows = []
    Fv = {(2, 2.0): 1.000905, (3, 2.0): 1.030841, (2, 2.5): 1.000451, (3, 2.5): 1.022630}
    for (p, n, m) in [(7, 2, 3), (11, 2, 3), (13, 2, 3), (5, 3, 3), (7, 3, 2)]:
        for q in [2.0, 2.5]:
            nu = n + 1
            Gs, _ = Gamma_s(p, nu / q, restarts=80, seed=p * 10 + n)
            B = 2 * (2.0 * p * nu) ** (-nu / 2.0)
            tgt = nu ** (-(nu - q) / q) * B * Gs
            jm = 0.0; best = -np.inf
            for trial in range(8):
                sc = [0.4, 1.0, 2.5, 1.0][trial % 4]
                E = rng.standard_normal((n, m, p)) * sc
                W = rng.standard_normal((p, m))
                En, Wn = gauge_symmetrise(E, W, p, n)
                En, Wn = rebalance(En, Wn, n)
                Eh2, Wh2 = fourier(En, Wn, p, n)
                G2, J2 = clean_and_junk(Eh2, Wh2, p, n)
                sc2 = max(np.abs(G2).max(), 1e-300)
                jm = max(jm, max(abs(v) for v in J2.values()) / sc2)
                best = max(best, margin_of(En, Wn, p, n) / Pq(En, Wn, q) ** (nu / q) / tgt)
            F = Fv.get((n, q), 1.0)
            print(f"{p:>4}{n:>3}{q:>5} | {jm:>18.3e} {Gs:>12.8f} {best/F:>18.9f} {best:>26.9f}")
            rows.append(dict(p=p, n=n, q=q, junk=jm, Gamma=Gs, ratio=best))
    rep["V5V6"] = rows
    mx = max(r["ratio"] for r in rows)
    print(f"  max ratio over all (H) nets = {mx:.9f} -> {'no violation' if mx <= 1 else '*** VIOLATION ***'}")

    with open(os.path.join(OUT, "network.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote", os.path.join(OUT, "network.json"))


if __name__ == "__main__":
    run()
