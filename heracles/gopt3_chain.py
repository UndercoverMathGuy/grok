"""gopt3_chain.py -- the NETWORK-level reduction: from an arbitrary gated chain to the atom programme,
with NO purity assumption, and the resulting certified form of (T).

What is verified here, all in fp64 against brute-force logit tables:

(V1) DICTIONARY.  For ANY gated chain (any width, neurons arbitrary vectors -- no single-frequency
     assumption), with  ehat_ij(k) = sum_a e_ij(a) w^{-ka},  w = e^{2 pi i / p}:
         clean part  g(D) := mean over rows a of z(a, s(a) - D)  =  p^{-nu} sum_k G_k w^{kD},
         G_k = sum_i prod_j ehat_ij(k) * conj(what_i(k)).
     Each neuron contributes to frequency k the ATOM
         c_ik = prod_j ehat_ij(k) conj(what_i(k)),   |c_ik| = t_ik = prod_j |ehat_ij(k)| |what_i(k)|,
     with phase data  v_ijk = conj(ehat_ij(k)) / ehat_ij(k)  on the unit circle, and the
     "same-|k| junk" coefficients (the Fourier coefficients of the logit tensor at multi-index
     (eps_1 k, ..., eps_n k ; -k) with eps in {+-1}^n, eps != all-plus) are exactly
         J_{k,T} = sum_i c_ik prod_{j in T} v_ijk ,   T = {j : eps_j = -1} nonempty.
     So the KL-constraint system of LEMMA_ATTEMPT holds at every frequency, with N = width atoms,
     for IMPURE neurons too.  (This is why "G2 = impure neurons" and "G1 = N > n+1" are the SAME gap.)

(V2) MARGIN.  margin <= margin(g) <= (2 p^{-nu}) min_{D != 0} sum_{k=1}^{(p-1)/2} |G_k| (1 - cos(2 pi k D/p)).

(V3) BUDGET (per neuron, impure allowed).  With m_ik = sum_j |ehat_ij(k)|^2 + |what_i(k)|^2:
         t_ik <= (m_ik/nu)^{nu/2}   (AM-GM over the nu factors),
         sum_{k=1}^{(p-1)/2} t_ik^r <= nu^{-q/2} sum_k m_ik^{q/2} <= (p/(2 nu))^{q/2} ||theta_i||^q,
     valid for every q >= 2 (r = q/nu).

(V4) THE CERTIFIED THEOREM.  If the same-|k| junk vanishes at every frequency, the peeling ladder of
     gopt3_peel.py gives sum_i t_ik^r >= L_r(n) |G_k|^r, hence

         gamma_q  <=  L_r(n)^{-1/r} * B * Gamma_{nu/q}(p),      B = 2 (2 p (n+1))^{-nu/2},

     i.e. (T) up to the factor  ((n+1)^{1-r} / L_r(n))^{1/r}  (1.0009 at nu 3 q 2).
     Tested on gauge-symmetrised random networks (junk-free by construction, impure, any width) and on
     the (n+1)-block (which attains (n+1)^{-(nu-q)/q} B Gamma exactly).

fp64, local CPU.  Usage: uv run python heracles/gopt3_chain.py --out heracles/FINAL_RUNS/GOPT3
"""
import argparse, itertools, json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from scipy.optimize import minimize

sys.path.insert(0, str(ROOT))
from gopt3_peel import L_r_vertex, peel_numbers_kernel


# ------------------------------------------------------------------ task / forward / margin
def table(p, n):
    r = np.arange(p ** n)
    ops = np.stack([(r // (p ** j)) % p for j in range(n)], 1)
    return ops, ops.sum(1) % p


def logits(E, W, p, n):
    """E: (n, m, p), W: (p, m).  Returns z: (p^n, p)."""
    ops, y = table(p, n)
    h = np.ones((ops.shape[0], E.shape[1]))
    for j in range(n):
        h = h * E[j][:, ops[:, j]].T
    return h @ W.T, y


def margin_of(z, y):
    zy = z[np.arange(len(y)), y]
    zz = z.copy(); zz[np.arange(len(y)), y] = -np.inf
    return float(np.min(zy - zz.max(1)))


def clean_g(z, y, p, n):
    """g(D) = mean over rows of z(row, s(row) - D)."""
    idx = (y[:, None] - np.arange(p)[None, :]) % p
    return z[np.arange(len(y))[:, None], idx].mean(0)


# ------------------------------------------------------------------ Fourier atoms
def atoms(E, W, p, n):
    Eh = np.fft.fft(E, axis=2)            # (n, m, p)   ehat_ij(k)
    Wh = np.fft.fft(W.T, axis=1)          # (m, p)      what_i(k)
    c = np.prod(Eh, axis=0) * np.conj(Wh)  # (m, p)     atoms c_ik
    G = c.sum(0)                           # (p,)       clean coefficients G_k
    return Eh, Wh, c, G


def same_k_junk(Eh, Wh, c, p, n, k):
    """J_{k,T} = sum_i c_ik prod_{j in T} v_ijk  for every T subset of {1..n}."""
    v = np.conj(Eh[:, :, k]) / np.where(np.abs(Eh[:, :, k]) > 0, Eh[:, :, k], 1.0)   # (n, m)
    out = {}
    for sz in range(n + 1):
        for T in itertools.combinations(range(n), sz):
            w = np.ones(c.shape[0], dtype=complex)
            for j in T:
                w = w * v[j]
            out[T] = complex(np.sum(c[:, k] * w))
    return out


def tensor_coeff(E, W, p, n, ks, kc):
    """Fourier coefficient of the logit tensor at multi-index (ks; kc), directly from the weights."""
    Eh = np.fft.fft(E, axis=2); Wh = np.fft.fft(W.T, axis=1)
    return complex(np.sum(np.prod([Eh[j, :, ks[j] % p] for j in range(n)], axis=0) * Wh[:, kc % p]))


# ------------------------------------------------------------------ gauge symmetrisation (kills same-|k| junk)
def gauge(E, W, p, n, phi):
    """Rotate every positive-frequency Fourier coefficient of each e by e^{i phi}, of w by e^{i n phi}.

    Orthogonal on each vector (a rotation inside every 2-D frequency plane), so norms are preserved.
    """
    nf = (p - 1) // 2
    mult_e = np.ones(p, dtype=complex); mult_w = np.ones(p, dtype=complex)
    for k in range(1, nf + 1):
        mult_e[k] = np.exp(1j * phi); mult_e[p - k] = np.exp(-1j * phi)
        mult_w[k] = np.exp(1j * n * phi); mult_w[p - k] = np.exp(-1j * n * phi)
    Eg = np.real(np.fft.ifft(np.fft.fft(E, axis=2) * mult_e[None, None, :], axis=2))
    Wg = np.real(np.fft.ifft(np.fft.fft(W.T, axis=1) * mult_w[None, :], axis=1)).T
    return Eg, Wg


def symmetrise(E, W, p, n):
    """Average over the n+1 gauge phases pi l/(n+1): kills all same-|k| junk.  Width becomes (n+1) m."""
    Es, Ws = [], []
    s = (n + 1) ** (-1.0 / (n + 1))          # scale each copy so the SUM is the average
    for l in range(n + 1):
        Eg, Wg = gauge(E, W, p, n, math.pi * l / (n + 1))
        Es.append(Eg * s); Ws.append(Wg * s)
    return np.concatenate(Es, axis=1), np.concatenate(Ws, axis=1)


# ------------------------------------------------------------------ the programme Gamma_s
def Gamma_s(p, s, restarts=60, seed=0):
    """max { min_{D != 0} sum_k x_k (1 - cos 2 pi k D/p) : x >= 0, sum_k x_k^{1/s} <= 1 }."""
    nf = (p - 1) // 2
    D = np.arange(1, p)
    g = 1 - np.cos(2 * math.pi * np.outer(np.arange(1, nf + 1), D) / p)   # (nf, p-1)
    e = s                                                                 # x = b^s, sum b = 1
    rng = np.random.default_rng(seed)
    best, bb = -np.inf, None
    for t in range(restarts + 1):
        b0 = np.full(nf, 1.0 / nf) if t == 0 else rng.dirichlet(np.full(nf, rng.uniform(0.2, 3.0)))
        f = lambda x: -x[-1]
        c1 = dict(type="ineq", fun=lambda x: (np.clip(x[:nf], 0, None) ** e) @ g - x[-1])
        c2 = dict(type="eq", fun=lambda x: x[:nf].sum() - 1)
        x0 = np.r_[b0, ((b0 ** e) @ g).min()]
        r = minimize(f, x0, method="SLSQP", bounds=[(0, 1)] * nf + [(None, None)],
                     constraints=[c1, c2], options=dict(maxiter=400, ftol=1e-14))
        b = np.clip(r.x[:nf], 0, None)
        if b.sum() <= 0:
            continue
        b = b / b.sum()
        v = float(((b ** e) @ g).min())
        if v > best:
            best, bb = v, b
    return best, bb


def build_chain(p, n, S):
    """(n+1)-neuron common-phase junk-free block on support S."""
    a = np.arange(p) * 2 * math.pi / p
    E = np.zeros((n, len(S) * (n + 1), p)); W = np.zeros((p, len(S) * (n + 1))); i = 0
    for k in S:
        for l in range(n + 1):
            ph = math.pi * l / (n + 1)
            for j in range(n):
                E[j, i] = np.cos(k * a + ph)
            W[:, i] = np.cos(k * a + n * ph); i += 1
    return E, W


# ------------------------------------------------------------------ main checks
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "GOPT3"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rep = {}

    print("== V1/V2/V3: dictionary, margin bound, per-neuron budget on RANDOM (impure) networks ==")
    e1 = e2 = e3 = 0.0; v2 = v3 = -np.inf
    for p, n, m in [(5, 2, 3), (7, 2, 5), (11, 2, 4), (13, 2, 3), (5, 3, 4), (7, 3, 3)]:
        nu = n + 1
        nS = min(3, (p - 1) // 2)
        Eb, Wb = build_chain(p, n, list(range(1, nS + 1)))          # positive-margin reference
        for trial in range(10):
            if trial < 4:                                            # blocks + noise: POSITIVE margin
                sc = [0.0, 0.05, 0.2, 0.5][trial]
                mm = Eb.shape[1]
                E = Eb + sc * rng.normal(size=(n, mm, p)); W = Wb + sc * rng.normal(size=(p, mm))
            else:
                E = rng.normal(size=(n, m, p)); W = rng.normal(size=(p, m))
            z, y = logits(E, W, p, n)
            Eh, Wh, c, G = atoms(E, W, p, n)
            # (V1a) clean part
            g = clean_g(z, y, p, n)
            grec = np.real(np.fft.ifft(G)) * p / p ** nu * p        # p^{-nu} sum_k G_k w^{kD}
            grec = np.array([np.real(np.sum(G * np.exp(2j * math.pi * np.arange(p) * D / p))) / p ** nu
                             for D in range(p)])
            e1 = max(e1, float(np.max(np.abs(g - grec)) / max(np.max(np.abs(g)), 1e-300)))
            # (V1b) same-|k| junk == tensor coefficients
            for k in (1, 2):
                J = same_k_junk(Eh, Wh, c, p, n, k)
                scale = float(np.sum(np.abs(c[:, k]))) + 1e-300      # natural size of every J_{k,T}
                for T, val in J.items():
                    ks = [(-k if j in T else k) for j in range(n)]
                    e1 = max(e1, abs(val - tensor_coeff(E, W, p, n, ks, -k)) / scale)
            # (V2) margin bound
            nf = (p - 1) // 2
            mg = margin_of(z, y)
            bd = min(2 * sum(abs(G[k]) * (1 - math.cos(2 * math.pi * k * D / p)) for k in range(1, nf + 1)) / p ** nu
                     for D in range(1, p))
            v2 = max(v2, mg / bd if bd > 0 else -np.inf)
            # (V3) per-neuron budget
            for q in (2.0, 2.5, float(nu)):
                r = q / nu
                for i in range(E.shape[1]):
                    t = np.abs(np.prod(Eh[:, i, :], axis=0) * np.conj(Wh[i]))[1:nf + 1]
                    th2 = float(np.sum(E[:, i, :] ** 2) + np.sum(W[:, i] ** 2))
                    lhs = float(np.sum(t ** r)); rhs = (p / (2 * nu)) ** (q / 2) * th2 ** (q / 2)
                    v3 = max(v3, lhs / rhs)
    print(f"  (V1) dictionary relative error                        : {e1:.2e}   (must be ~1e-15)")
    print(f"  (V2) max margin / [2 p^-nu min_D sum_k |G_k| g_k(D)]  : {v2:.6f}   (must be <= 1)")
    print(f"  (V3) max [sum_k t_ik^r] / [(p/2nu)^{{q/2}} ||th_i||^q]  : {v3:.6f}   (must be <= 1)")
    rep.update(V1=e1, V2=v2, V3=v3)

    print("\n== V4: the certified theorem on gauge-symmetrised (junk-free, IMPURE) random networks ==")
    rows = []
    for p, n in [(7, 2), (11, 2), (13, 2), (7, 3)]:
        nu = n + 1
        D = peel_numbers_kernel(n)
        for q in (2.0, 2.5):
            if q > nu:
                continue
            r = q / nu
            Lr, _ = L_r_vertex(n, D, r)
            loss = (nu ** (1 - r) / Lr) ** (1 / r)
            Gs, _ = Gamma_s(p, nu / q)
            B = 2 * (2 * p * nu) ** (-nu / 2.0)
            tgt = nu ** (-(nu - q) / q) * B * Gs                    # the target (T)
            cert = loss * tgt                                       # what is PROVED here
            worst_sym, worst_asc = -np.inf, -np.inf
            junkmax = 0.0

            def gam_of(Eraw, Wraw):
                Es, Ws = symmetrise(Eraw, Wraw, p, n)
                z, y = logits(Es, Ws, p, n)
                Pq = float(np.sum((np.sum(Es ** 2, axis=(0, 2)) + np.sum(Ws ** 2, axis=0)) ** (q / 2)))
                return margin_of(z, y) / Pq ** (nu / q), Es, Ws

            nS = min(4, (p - 1) // 2)
            Eb0, Wb0 = build_chain(p, n, list(range(1, nS + 1)))
            trials = [(Eb0 + s * rng.normal(size=Eb0.shape), Wb0 + s * rng.normal(size=Wb0.shape))
                      for s in (0.0, 0.05, 0.2, 0.5, 1.0)]
            trials += [(rng.normal(size=(n, mw, p)) / math.sqrt(p), rng.normal(size=(p, mw)) / math.sqrt(p))
                       for mw in (2, 4, 8) for _ in range(3)]
            for Eraw, Wraw in trials:
                gam, Es, Ws = gam_of(Eraw, Wraw)
                worst_sym = max(worst_sym, gam / cert)
                Eh, Wh, cc, G = atoms(Es, Ws, p, n)
                for k in range(1, (p - 1) // 2 + 1):
                    J = same_k_junk(Eh, Wh, cc, p, n, k)
                    sc = max(abs(J[()]), 1e-300)
                    if abs(J[()]) > 1e-8 * np.abs(cc).max() * cc.shape[0]:
                        junkmax = max(junkmax, max(abs(J[T]) for T in J if T) / sc)

            # --- margin ASCENT inside the junk-free class (parametrised by the raw net; symmetrise is linear)
            mw = (n + 1) * 2
            beta = 400.0

            def negobj(x):
                Eraw = x[:n * mw * p].reshape(n, mw, p); Wraw = x[n * mw * p:].reshape(p, mw)
                Es, Ws = symmetrise(Eraw, Wraw, p, n)
                z, y = logits(Es, Ws, p, n)
                zy = z[np.arange(len(y)), y]; zz = z.copy(); zz[np.arange(len(y)), y] = -np.inf
                gaps = (zy[:, None] - zz).ravel(); gaps = gaps[np.isfinite(gaps)]
                soft = -np.log(np.sum(np.exp(-beta * (gaps - gaps.min())))) / beta + gaps.min()
                Pq = float(np.sum((np.sum(Es ** 2, axis=(0, 2)) + np.sum(Ws ** 2, axis=0)) ** (q / 2)))
                return -soft / Pq ** (nu / q)

            # exact warm start: ONE PURE NEURON per frequency of the programme's optimiser.
            # symmetrise() then turns it into the (n+1)-block, so this start certifies the parametrisation.
            _, bs0 = Gamma_s(p, nu / q)
            S0 = [k for k in range(1, (p - 1) // 2 + 1) if bs0[k - 1] > 1e-4]
            aa = np.arange(p) * 2 * math.pi / p
            mw = max(mw, len(S0))
            Ew = np.zeros((n, mw, p)); Ww = np.zeros((p, mw))
            for idx, k in enumerate(S0):
                sc = bs0[k - 1] ** 0.5
                for j in range(n):
                    Ew[j, idx] = sc * np.cos(k * aa)
                Ww[:, idx] = sc * np.cos(k * aa)
            gwarm, _, _ = gam_of(Ew, Ww)
            worst_asc = max(worst_asc, gwarm / cert)

            for st in range(3):
                if st == 0:
                    x0 = np.concatenate([(Eb0[:, :mw] if Eb0.shape[1] >= mw else
                                          np.pad(Eb0, ((0, 0), (0, max(0, mw - Eb0.shape[1])), (0, 0))))[:, :mw].ravel(),
                                         (Wb0[:, :mw] if Wb0.shape[1] >= mw else
                                          np.pad(Wb0, ((0, 0), (0, max(0, mw - Wb0.shape[1])))))[:, :mw].ravel()])
                    x0 = x0 + 1e-3 * rng.normal(size=x0.shape)
                else:
                    x0 = rng.normal(size=n * mw * p + p * mw) / math.sqrt(p)
                res = minimize(negobj, x0, method="L-BFGS-B", options=dict(maxiter=1500))
                Eraw = res.x[:n * mw * p].reshape(n, mw, p); Wraw = res.x[n * mw * p:].reshape(p, mw)
                gam, _, _ = gam_of(Eraw, Wraw)
                worst_asc = max(worst_asc, gam / cert)

            # the (n+1)-block on the programme's optimal support
            _, bstar = Gamma_s(p, nu / q)
            S = [k for k in range(1, (p - 1) // 2 + 1) if bstar[k - 1] > 1e-3]
            Eb, Wb = build_chain(p, n, S)
            # give each frequency the programme's amplitude
            for idx, k in enumerate(S):
                sc = bstar[k - 1] ** 0.5
                Eb[:, idx * (n + 1):(idx + 1) * (n + 1)] *= sc
                Wb[:, idx * (n + 1):(idx + 1) * (n + 1)] *= sc
            zb, yb = logits(Eb, Wb, p, n)
            Pqb = float(np.sum((np.sum(Eb ** 2, axis=(0, 2)) + np.sum(Wb ** 2, axis=0)) ** (q / 2)))
            gb = margin_of(zb, yb) / Pqb ** (nu / q)
            print(f"  p {p:3d} n {n} q {q:g}:  Gamma_s {Gs:.5f}  L_r {Lr:.6f}  certified factor {loss:.5f}"
                  f"\n              junk-free residual {junkmax:.2e};  gamma/certified: sampled {worst_sym:.4f},"
                  f" ASCENT {worst_asc:.4f};  block/target {gb/tgt:.5f}", flush=True)
            rows.append(dict(p=p, n=n, q=q, Gamma=Gs, L_r=Lr, loss=loss, junk=junkmax,
                             sym=worst_sym, ascent=worst_asc, block=gb / tgt))
    rep["V4"] = rows
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "chain.json").write_text(json.dumps(rep, indent=1, default=float))
    print("\nwrote", Path(a.out) / "chain.json")
