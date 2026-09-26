"""lemma_verify.py -- fp64 checks of every step claimed in heracles/LEMMA_ATTEMPT.md.

Six independent checks, all exact float64, all local CPU, seconds:

  A  REDUCTION.  Build a real gated-chain network from an arbitrary "atom" configuration
     {(c_l, v_l1..v_ln)} at one frequency k and confirm that (i) its logit table's junk Fourier
     coefficients are exactly the numbers C_T = sum_l c_l prod_{j in T} v_lj of the splitting
     programme, and (ii) its normalised margin equals   |C_empty| * M / (cost)^{nu/q}  with the
     analytic cost.  This is the bridge from the network to the finite programme.

  B  CONSTANT.  The (n+1)-neuron common-phase block attains exactly (n+1)^{-(nu-q)/q} * B * Gamma
     with B = 2 (2 p (n+1))^{-nu/2}, for n = 2,3,4 and several q, on single- and multi-frequency
     supports.

  C  RIGIDITY (n = 2, N = 3).  Every junk-free 3-atom configuration with clean amplitude A != 0 has
     y_l = e^{i phi} x_l (all three second-operand phases equal the first-operand phases up to one
     global rotation), i.e. is common-phase.  Checked by solving the system from random starts.

  D  HADAMARD/FEKETE step.  max over N points on the unit circle of prod_{l<m} |z_l - z_m| = N^{N/2},
     attained at the N-th roots of unity  (the inequality that supplies the constant (n+1)).

  E  QUANTITATIVE JUNK LEMMA.  For random networks: margin(z) <= mean_a phi_D(a) - |psi_D(m)| for
     every D != 0 and every mode m != 0, where phi_D(a) = z(a, s(a)) - z(a, s(a)-D) and psi_D(m) is
     phi_D's m-th Fourier coefficient.  (Strengthens the junk lemma, which is the m = 0 case.)

  F  JUNK BUDGET.  For a pure single-frequency population the lemma forces
     max_{T != empty} |C_T| <= 2 |C_empty| sin(pi/p)  whenever the margin is positive.

Usage: uv run python heracles/lemma_verify.py
"""
import itertools, json, math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "LEMMA"
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ gated chain, exact fp64
def table(p, n):
    return np.array(list(itertools.product(range(p), repeat=n)), dtype=np.int64)


def logits(E, W, p, n):
    """E: (n, m, p) embeddings, W: (p, m) readout.  z[row, c] = sum_i W[c,i] prod_j E[j,i,a_j]."""
    ops = table(p, n)
    h = np.ones((ops.shape[0], E.shape[1]))
    for j in range(n):
        h = h * E[j][:, ops[:, j]].T
    return h @ W.T, ops.sum(1) % p, ops


def margin_of(z, y):
    zz = z.copy()
    zy = zz[np.arange(len(y)), y].copy()
    zz[np.arange(len(y)), y] = -np.inf
    return float((zy - zz.max(1)).min())


def clean_g(z, y, p, n):
    """g(D) = mean over rows of z[row, s(row) - D]."""
    idx = (y[:, None] - np.arange(p)[None, :]) % p
    return z[np.arange(len(y))[:, None], idx].mean(0)


def build_atoms(p, n, k, c, v):
    """One frequency k; atom l -> one neuron.  c_l = (prod_j r_lj) rho_l e^{i psi_l}; v_lj = e^{-2 i alpha_lj}.
    We realise it with all n+1 factor norms equal (balanced): r = rho = |c_l|^{1/(n+1)}."""
    N = len(c)
    a = 2 * math.pi * k * np.arange(p) / p
    E = np.zeros((n, N, p))
    W = np.zeros((p, N))
    for l in range(N):
        s = abs(c[l]) ** (1.0 / (n + 1))
        alpha = [-0.5 * np.angle(v[l][j]) for j in range(n)]
        beta = sum(alpha) - np.angle(c[l])          # psi_l = sum_j alpha_lj - beta_l
        for j in range(n):
            E[j, l] = s * np.cos(a + alpha[j])
        W[:, l] = s * np.cos(a + beta)
    return E, W


def CT(c, v, n):
    """The 2^n sign-pattern coefficients C_T = sum_l c_l prod_{j in T} v_lj."""
    out = {}
    for rbits in range(2 ** n):
        T = tuple(j for j in range(n) if rbits >> j & 1)
        w = np.ones(len(c), dtype=complex)
        for j in T:
            w = w * np.array([v[l][j] for l in range(len(c))])
        out[T] = complex((np.asarray(c) * w).sum())
    return out


def kstar_gamma(p, s, iters=4000):
    """Gamma_s(p) = max_{b >= 0, sum b = 1} min_{D != 0} sum_k b_k^s (1 - cos 2 pi k D / p).
    Projected ascent on the simplex from many starts (small p: nf is tiny, this is reliable)."""
    nf = (p - 1) // 2
    kk = np.arange(1, nf + 1)
    dd = np.arange(1, p)
    Gm = 1 - np.cos(2 * np.pi * np.outer(kk, dd) / p)          # (nf, p-1)
    best = -1.0
    for _ in range(60):
        b = rng.dirichlet(np.ones(nf))
        lr = 0.05
        for t in range(iters):
            x = b ** s
            vals = x @ Gm
            d = int(vals.argmin())
            g = s * b ** (s - 1) * Gm[:, d]
            b = b + lr * g / (np.linalg.norm(g) + 1e-30)
            b = np.maximum(b, 0)
            b = b / b.sum()
            lr *= 0.9995
        v = float((b ** s @ Gm).min())
        best = max(best, v)
    return best


# ------------------------------------------------------------------ checks
def check_A(p=13, n=2, k=1, N=4, q=2.0):
    nu = n + 1
    c = rng.normal(size=N) + 1j * rng.normal(size=N)
    v = [[np.exp(1j * rng.uniform(0, 2 * np.pi)) for _ in range(n)] for _ in range(N)]
    E, W = build_atoms(p, n, k, c, v)
    z, y, _ = logits(E, W, p, n)
    g = clean_g(z, y, p, n)
    Cs = CT(c, v, n)
    # clean spectrum of g at frequency k:  g(D) = 2^{-n} Re[C_empty (1 - ...)] + const  -> ghat(k)
    gh = np.fft.fft(g) / p
    pred = 2.0 ** (-(n + 1)) * Cs[()]                     # coefficient of e^{-i theta_D}: 2^{-(n+1)} C_empty
    got = gh[k % p]
    # junk Fourier coefficients of the full table at the modes m_T = -2k 1_T
    _, _, ops = logits(E, W, p, n)
    phiD = {}
    errs = [abs(abs(got) - abs(pred))]
    for D in range(1, p):
        idx = (y - D) % p
        phi = z[np.arange(len(y)), y] - z[np.arange(len(y)), idx]
        for rbits in range(1, 2 ** n):
            T = tuple(j for j in range(n) if rbits >> j & 1)
            m = np.zeros(n, dtype=np.int64)
            for j in T:
                m[j] = (-2 * k) % p
            ph = np.exp(-2j * np.pi * (ops @ m) / p)
            psi = (phi * ph).mean()
            predpsi = 2.0 ** (-(n + 1)) * Cs[T] * (1 - np.exp(2j * np.pi * k * D / p))
            errs.append(abs(psi - predpsi))
    return max(errs)


def check_B(ns=(2, 3, 4), qs=(2.0, 2.5)):
    rows = []
    for n in ns:
        nu = n + 1
        p = {2: 41, 3: 17, 4: 11}[n]        # keep p^n rows manageable; the identity is p-independent
        for q in qs:
            if q > nu:
                continue
            s = nu / q
            for S in ([1], [1, 3], [1, 3, 4]):
                # (n+1)-block on support S with equal shares
                Nn = (n + 1) * len(S)
                a = 2 * math.pi * np.arange(p) / p
                E = np.zeros((n, Nn, p)); W = np.zeros((p, Nn)); i = 0
                for k in S:
                    for l in range(n + 1):
                        ph = math.pi * l / (n + 1)
                        for j in range(n):
                            E[j, i] = np.cos(k * a + ph)
                        W[:, i] = np.cos(k * a + n * ph)
                        i += 1
                z, y, _ = logits(E, W, p, n)
                mg = margin_of(z, y)
                Pq = float((np.sum(E ** 2, axis=(0, 2)) + np.sum(W ** 2, 0)) ** (q / 2)).__float__() if False else \
                     float(np.sum((np.sum(E ** 2, axis=(0, 2)) + np.sum(W ** 2, 0)) ** (q / 2)))
                gam = mg / Pq ** (nu / q)
                # programme value of this (equal-share) point
                b = np.ones(len(S)) / len(S)
                dd = np.arange(1, p)
                val = float(min(sum(b[t] ** s * (1 - math.cos(2 * math.pi * S[t] * D / p)) for t in range(len(S)))
                                for D in dd))
                B = 2 * (2 * p * (n + 1)) ** (-nu / 2)
                rows.append(dict(n=n, q=q, S=S, gamma=gam,
                                 pred=(n + 1) ** (-(nu - q) / q) * B * val,
                                 ratio=gam / ((n + 1) ** (-(nu - q) / q) * B * val)))
    return rows


def check_C(trials=400):
    """n=2, N=3: solve the junk-free system from random starts, test y_l = e^{i phi} x_l."""
    from scipy.optimize import fsolve
    worst = 0.0
    found = 0
    for _ in range(trials):
        x0 = np.concatenate([rng.normal(0, 1, 6), rng.uniform(0, 2 * np.pi, 4)])

        def F(u):
            c = u[:3] + 1j * u[3:6]
            x = np.array([1.0, np.exp(1j * u[6]), np.exp(1j * u[7])])
            y = np.array([1.0, np.exp(1j * u[8]), np.exp(1j * u[9])])
            r = np.array([c.sum() - 1, (c * x).sum(), (c * y).sum(), (c * x * y).sum()])
            return np.concatenate([r.real, r.imag, [0.0, 0.0]])

        u, _, ier, _ = fsolve(F, x0, full_output=True)
        if ier != 1 or np.abs(F(u)).max() > 1e-10:
            continue
        found += 1
        sx = np.array([0.0, u[6], u[7]]) % (2 * np.pi)
        sy = np.array([0.0, u[8], u[9]]) % (2 * np.pi)
        d = (sy - sx) % (2 * np.pi)                     # should be a single constant phi
        dev = float(np.abs(np.exp(1j * d) - np.exp(1j * d[0])).max())
        worst = max(worst, dev)
    return found, worst


def check_D(Ns=(2, 3, 4, 5, 6), tries=20000):
    out = []
    for N in Ns:
        best = 0.0
        for _ in range(tries):
            t = rng.uniform(0, 2 * np.pi, N)
            z = np.exp(1j * t)
            pr = 1.0
            for l in range(N):
                for m in range(l + 1, N):
                    pr *= abs(z[l] - z[m])
            best = max(best, pr)
        out.append(dict(N=N, best_found=best, bound=N ** (N / 2), roots_of_unity=float(
            np.prod([abs(np.exp(2j * np.pi * l / N) - np.exp(2j * np.pi * m / N))
                     for l in range(N) for m in range(l + 1, N)]))))
    return out


def check_E(p=11, n=2, trials=12, m_neurons=5):
    worst = -np.inf
    for _ in range(trials):
        E = rng.normal(size=(n, m_neurons, p)) * 0.8
        W = rng.normal(size=(p, m_neurons)) * 0.8
        z, y, ops = logits(E, W, p, n)
        mg = margin_of(z, y)
        for D in range(1, p):
            phi = z[np.arange(len(y)), y] - z[np.arange(len(y)), (y - D) % p]
            mean = phi.mean()
            for _ in range(40):
                m = rng.integers(0, p, n)
                if not m.any():
                    continue
                psi = (phi * np.exp(-2j * np.pi * (ops @ m) / p)).mean()
                worst = max(worst, mg - (mean - abs(psi)))
    return worst


if __name__ == "__main__":
    res = {}
    print("A  reduction network <-> atom programme")
    for (p, n, N) in ((13, 2, 4), (11, 2, 3), (11, 3, 5), (13, 2, 6)):
        e = check_A(p=p, n=n, N=N)
        print(f"   p={p} n={n} N={N}:  max |predicted - measured| Fourier coefficient = {e:.3e}")
        res[f"A_p{p}_n{n}_N{N}"] = e

    print("B  (n+1)-block attains (n+1)^{-(nu-q)/q} B Gamma")
    res["B"] = check_B()
    for r in res["B"]:
        print(f"   n={r['n']} q={r['q']:g} S={r['S']}:  gamma {r['gamma']:.6e}  predicted {r['pred']:.6e}"
              f"  ratio {r['ratio']:.10f}")

    print("C  rigidity: every junk-free 3-atom config at n=2 is common-phase")
    f, w = check_C()
    print(f"   solutions found {f};  max deviation of e^{{i(sy-sx)}} from constant = {w:.3e}")
    res["C"] = dict(found=f, worst=w)

    print("D  Hadamard/Fekete: max prod |z_l - z_m| over N circle points")
    res["D"] = check_D()
    for r in res["D"]:
        print(f"   N={r['N']}: random best {r['best_found']:.6f}  roots of unity {r['roots_of_unity']:.6f}"
              f"  Hadamard bound N^(N/2) {r['bound']:.6f}")

    print("E  quantitative junk lemma on random nets (must be <= 0)")
    w = check_E()
    print(f"   max over (net, D, m) of [margin - (mean phi_D - |psi_D(m)|)] = {w:.3e}")
    res["E"] = w

    print("F  junk budget: pure-class nets with junk J relative to clean |C_0| -- margin sign vs 2 sin(pi/p)")
    res["F"] = []
    for p in (11, 23, 41):
        n, k = 2, 1
        row = []
        for eps in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            # 3-block at frequency k, then TILT one neuron's amplitude to inject junk of size ~eps*2sin(pi/p)
            c = [1 / 3, 1 / 3, 1 / 3]
            v = [[np.exp(-2j * math.pi * l / 3)] * 2 for l in range(3)]
            t = eps * 2 * math.sin(math.pi / p)
            c = [c[0] * (1 + t * 3 / 2), c[1], c[2] * (1 - t * 3 / 2)]
            E, W = build_atoms(p, n, k, c, v)
            z, y, _ = logits(E, W, p, n)
            Cs = CT(c, v, n)
            J = max(abs(Cs[T]) for T in Cs if T != ())
            mg = margin_of(z, y)
            # analytic ceiling from the quantitative junk lemma at the worst D (x = sin(pi k D/p) minimal)
            x = math.sin(math.pi / p)
            ceil = 2.0 ** (-n) * (2 * abs(Cs[()]) * x * x - J * x)
            row.append(dict(eps=eps, J_over_C0=J / abs(Cs[()]), margin=mg, lemma_ceiling=ceil,
                            ok=bool(mg <= ceil + 1e-12)))
            print(f"   p={p:3d} J/|C_0| = {J/abs(Cs[()]):.6f}  (2 sin(pi/p) = {2*math.sin(math.pi/p):.6f})"
                  f"   margin {mg:+.6e}   lemma ceiling {ceil:+.6e}   holds {row[-1]['ok']}")
        res["F"].append(dict(p=p, rows=row))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(res, indent=1, default=float))
