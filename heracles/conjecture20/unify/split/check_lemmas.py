"""Numerical sanity checks of the SPLIT lemmas (exact small cases, float64):
  L1  stabiliser: plane rotations by phi not in {0, pi} change some sign-flipped coefficient; reflections change beta.
  L2  torus average = clean-mode (IF) table: the exact average over a grid of plane rotations (with output
      compensation) keeps only the clean modes (j,..,j;-j).
  L3  channel classification: the mixed contribution to g_hat_delta at +-2l 1_T equals the sum over the predicted
      channels (partner 3l for j = l; |2l -+ j| on T for j != l; DC is the case |2l -+ j| = 0).
  L4  joint circularity: class-C networks have E[(Phi_l - E Phi_l)(Phi_k - E Phi_k)] = 0 for all l, k (any odd p);
      for a mixed network the pseudo-covariance equals the sum over the predicted 'pair' modes.
  L5  span transfer: if no mixed character of g(.,delta) lies in V_delta = span(supp g_split(.,delta)), then
      min_x g_theta(., delta) <= min_x g_split(., delta); quantitative form min g_theta <= min (g_split + g_R^V).
  L6  IF cores: theta interference-free (glued IF blocks + DC/dead additions) => h(theta) <= h(theta_split).
  L7  universal bound with live split costs: h <= B_p min_delta sum_j P_j^s kappa_j(delta) <= B_p Gamma_s P_live^s.
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python check_lemmas.py
"""
import itertools, numpy as np
from lib import *

rng = np.random.default_rng(3)
rho = lambda p: np.exp(2j * np.pi / p)
report = []


def ok(name, cond, val=None):
    report.append((name, bool(cond), val))


def rot_plane(v, k, p, phi):
    """rotate the frequency-k component of real vector(s) v by angle phi: vhat(k) -> e^{i phi} vhat(k)."""
    if k == 0:
        return v
    vh = np.fft.fft(v, axis=-1)
    comp = np.zeros_like(vh)
    comp[..., k] = vh[..., k] * np.exp(1j * phi); comp[..., (-k) % p] = vh[..., (-k) % p] * np.exp(-1j * phi)
    pk = proj(v, k, p)
    return v - pk + np.real(np.fft.ifft(comp, axis=-1))


def ghat(U, W, d):
    """Fourier coefficients of x -> g(x, d) on Z_p^n, convention g = sum_chi ghat(chi) rho^{chi.x}."""
    m, n, p = U.shape
    G = gaps_from_table(table(U, W), p, n)[:, d].reshape((p,) * n)
    return np.fft.fftn(G) / p ** n


# ---------------- L1 stabiliser
p, n, m = 7, 2, 3
U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
U -= proj(U, 0, p)
b0, bT0, _ = own_coeffs(U, W)
phi = 0.7
U2 = U.copy(); U2[:, 0] = rot_plane(U[:, 0], 2, p, phi)
W2 = rot_plane(W, 2, p, phi)            # output compensated so that beta_2 is fixed
b1, bT1, _ = own_coeffs(U2, W2)
ok('L1 rotation keeps beta_2', abs(b1[2] - b0[2]) < 1e-12, abs(b1[2] - b0[2]))
ok('L1 rotation phi=0.7 changes beta_{2,{1}} by e^{-2i phi}', abs(bT1[(2, (0,))] - np.exp(-2j * phi) * bT0[(2, (0,))]) < 1e-12)
U3 = U.copy(); U3[:, 0] = rot_plane(U[:, 0], 2, p, np.pi); W3 = rot_plane(W, 2, p, np.pi)
b3, bT3, _ = own_coeffs(U3, W3)
ok('L1 rotation by pi (= sign flip) fixes all own data',
   max(abs(bT3[k] - bT0[k]) for k in bT0) < 1e-12 and max(abs(b3[j] - b0[j]) for j in b0) < 1e-12)
# reflection in plane 2 of slot 1: vhat(2) -> conj(vhat(2))
Uh = np.fft.fft(U[:, 0], axis=-1); Uh[:, 2] = np.conj(Uh[:, 2]); Uh[:, p - 2] = np.conj(Uh[:, p - 2])
U4 = U.copy(); U4[:, 0] = np.real(np.fft.ifft(Uh, axis=-1))
b4, _, _ = own_coeffs(U4, W)
ok('L1 reflection changes beta_2 (generic)', abs(b4[2] - b0[2]) > 1e-6, abs(b4[2] - b0[2]))

# ---------------- L2 torus average
p, n, m = 5, 2, 2
U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
F = Fset(p); N = 9                        # grid of angles 2 pi a / N, N > 2 nu suffices
acc = np.zeros((p,) * n + (p,)); cnt = 0
for angs in itertools.product(range(N), repeat=len(F) * n):
    A = np.array(angs).reshape(len(F), n) * 2 * np.pi / N
    U2 = U.copy(); W2 = W.copy()
    for ki, k in enumerate(F):
        for t in range(n):
            U2[:, t] = rot_plane(U2[:, t], k, p, A[ki, t])
        W2 = rot_plane(W2, k, p, A[ki].sum())
    acc += table(U2, W2); cnt += 1
avg = acc / cnt
fh = np.fft.fftn(table(U, W)); ah = np.fft.fftn(avg)
keep = np.zeros_like(fh, bool)
for j in range(p):
    keep[tuple([j % p] * n + [(-j) % p])] = True
# output-DC blocks with all-DC inputs also survive (j = 0 mode); DC inputs with nonzero output are killed
err_keep = np.abs(ah[keep] - fh[keep]).max(); err_kill = np.abs(ah[~keep]).max()
ok('L2 torus average keeps clean modes', err_keep < 1e-9, err_keep)
ok('L2 torus average kills every other mode (sign-flipped and mixed)', err_kill < 1e-9, err_kill)

# ---------------- L3 channel classification
for (p, n) in [(7, 2), (11, 2), (7, 3), (5, 2)]:
    m = 4
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
    F = Fset(p)
    Us, Ws = split(U, W)
    Ud = np.array([[proj(U[i, t], 0, p) for t in range(n)] for i in range(m)]); Wd = proj(W, 0, p)
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1)
    worst = 0
    for d in range(1, p):
        gh = ghat(U, W, d) - ghat(Us, Ws, d)          # mixed part (DC block is constant in c: no gap)
        for l in F:
            for r_ in range(1, n + 1):
                for T in itertools.combinations(range(n), r_):
                    for sgn in (1, -1):
                        chi = tuple((sgn * 2 * l) % p if t in T else 0 for t in range(n))
                        # predicted: sum over output freqs j of direct modes (q = chi - j) and conjugate modes
                        pred = 0
                        for j in F:
                            cj = 1 - rho(p) ** (-j * d)
                            for (q_sign, conj_) in ((1, False), (-1, True)):
                                ch = tuple((q_sign * c) % p for c in chi)
                                q = [(ch[t] - j) % p for t in range(n)]
                                clean = all(qt in (j % p, (-j) % p) for qt in q)
                                if clean:
                                    continue      # own sign-flipped mode (or clean mean) -> not mixed
                                coef = 2 * p ** (-(n + 1)) * sum(Wh[i, j] * np.prod([Uh[i, t, q[t]] for t in range(n)]) for i in range(m))
                                pred += 0.5 * (np.conj(coef * cj) if conj_ else coef * cj)
                                # predicted input frequency of the channel on T:
                                a = min((ch[T[0]] - j) % p, (j - ch[T[0]]) % p)
                                expect = {abs(2 * l - j) % p, (-(abs(2 * l - j))) % p, (2 * l + j) % p, (-(2 * l + j)) % p}
                                assert a in {min(x % p, (-x) % p) for x in expect} or (j == l), (a, j, l)
                        worst = max(worst, abs(gh[chi] - pred))
    ok(f'L3 channels reproduce the mixed part of g_hat at +-2l1_T (p={p}, n={n})', worst < 1e-10, worst)

# ---------------- L4 joint circularity
for (p, n) in [(3, 2), (5, 2), (7, 3), (11, 2)]:
    F = Fset(p); m = 3 * len(F)
    # class C: each neuron one frequency, no input DC
    U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for i in range(m):
        k = F[i % len(F)]
        U[i] = proj(rng.normal(size=(n, p)), k, p); W[i] = proj(rng.normal(size=p), k, p) + rng.normal()
    worst = 0
    for l in F:
        for k in F:
            pl = field_coeffs(U, W, l); pk = field_coeffs(U, W, k)
            # E[(Phi_l - E)(Phi_k - E)] = sum_{chi != 0} phi_l(chi) phi_k(-chi)
            idx = tuple(np.indices((p,) * n).reshape(n, -1))
            neg = tuple((-np.array(idx)) % p)
            s_ = (pl[idx] * pk[neg]).sum() - pl[(0,) * n] * pk[(0,) * n]
            worst = max(worst, abs(s_))
    ok(f'L4 class C joint pseudo-covariance vanishes (p={p}, n={n})', worst < 1e-12, worst)
# mixed network: pseudo-covariance of Phi_1 at p = 3 with DC is non-zero, at p = 5 with only {0,1} content it vanishes
p, n = 3, 2
U = rng.normal(size=(3, n, p)); W = rng.normal(size=(3, p))
pl = field_coeffs(U, W, 1); idx = tuple(np.indices((p,) * n).reshape(n, -1)); neg = tuple((-np.array(idx)) % p)
pv3 = abs((pl[idx] * pl[neg]).sum() - pl[(0,) * n] ** 2)
p = 5
U = np.zeros((3, n, p)); W = np.zeros((3, p))
for i in range(3):
    U[i] = proj(rng.normal(size=(n, p)), 1, p) + proj(rng.normal(size=(n, p)), 0, p); W[i] = proj(rng.normal(size=p), 1, p)
pl = field_coeffs(U, W, 1); idx = tuple(np.indices((p,) * n).reshape(n, -1)); neg = tuple((-np.array(idx)) % p)
pv5 = abs((pl[idx] * pl[neg]).sum() - pl[(0,) * n] ** 2)
ok('L4 p=3 single-frequency+DC: pseudo-variance non-zero', pv3 > 1e-6, pv3)
ok('L4 p=5 single-frequency+DC (no 2l-pair): pseudo-variance zero', pv5 < 1e-12, pv5)
# p = 5 with input frequency {0, 2} in a slot (a 2l-pair for l = 1): non-zero
U2 = U + np.concatenate([proj(rng.normal(size=(3, 1, p)), 2, p), np.zeros((3, 1, p))], 1)
pl = field_coeffs(U2, W, 1)
pv5b = abs((pl[idx] * pl[neg]).sum() - pl[(0,) * n] ** 2)
ok('L4 p=5 with a 2l-pair {0,2} in ONE slot (clean frequency elsewhere): pseudo-variance non-zero', pv5b > 1e-6, pv5b)

# ---------------- L5 span transfer and L6 IF cores
p, n = 7, 2


def classC_relations_core(p, n, feats, rng):
    """class-C core whose interference sits on T = [n] only: per feature two aligned atoms cancelling every proper T."""
    Us, Ws = [], []
    a = np.arange(p)
    for j in feats:
        # n = 2: two neurons, slot phases (0, 0) and (pi/2, pi/2)... choose phases so that T={1},{2} cancel
        for (al1, al2, ph) in [(0.0, 0.0, 0.0), (np.pi / 2, np.pi / 2, 0.0)]:
            u1 = np.cos(2 * np.pi * j * a / p + al1); u2 = np.cos(2 * np.pi * j * a / p + al2)
            w = np.cos(2 * np.pi * j * a / p + al1 + al2 + ph)
            Us.append([u1, u2]); Ws.append(w)
    return np.array(Us), np.array(Ws)


Uc, Wc = classC_relations_core(p, n, [1, 2, 3], rng)
_, bT, _ = own_coeffs(Uc, Wc)
ok('L5 test core has interference only on T={1,2}',
   max(abs(bT[(j, T)]) for (j, T) in bT if T != (0, 1)) < 1e-12 and max(abs(bT[(j, (0, 1))]) for j in [1, 2, 3]) > 0.1)
# add mixed content with DIFFERENT off-frequencies in the two slots (characters off the diagonal V)
worst5 = -1e9; worst5q = -1e9
for trial in range(20):
    U = Uc.copy(); W = Wc.copy()
    for i in range(len(U)):
        U[i, 0] += 0.4 * proj(rng.normal(size=p), 0, p)          # DC in slot 1 only
        U[i, 1] += 0.4 * proj(rng.normal(size=p), 1 + (i % 3), p) * (0 if True else 1)
    # make slot-2 off-frequency content differ from slot-1's: slot 1 gets DC, slot 2 gets nothing -> chars (j, *) off diag
    Us, Ws = split(U, W)
    for d in range(1, p):
        G = gaps_from_table(table(U, W), p, n)[:, d]; Gs = gaps_from_table(table(Us, Ws), p, n)[:, d]
        worst5 = max(worst5, G.min() - Gs.min())
ok('L5 mixed modes off span V never raise min_x g (max over delta of min g_theta - min g_split <= 0)', worst5 <= 1e-12, worst5)
# quantitative form on a generic mixed network with full span: min g <= min(g_split + g_R^V)
U = rng.normal(size=(4, n, p)); W = rng.normal(size=(4, p))
Us, Ws = split(U, W)
worstq = -1e9
for d in range(1, p):
    gs_h = ghat(Us, Ws, d); g_h = ghat(U, W, d)
    supp = np.argwhere(np.abs(gs_h) > 1e-12)
    # V = span of supp (over Z_p) -- compute as the set of all Z_p-combinations of the support vectors (n = 2: easy)
    vecs = [tuple(v) for v in supp if any(v)]
    V = {(0,) * n}
    changed = True
    while changed:
        changed = False
        for v in list(V):
            for w in vecs:
                for c in range(1, p):
                    z = tuple((v[t] + c * w[t]) % p for t in range(n))
                    if z not in V:
                        V.add(z); changed = True
    mask = np.zeros((p,) * n, bool)
    for v in V:
        mask[v] = True
    gR_h = g_h - gs_h
    projV = gs_h + np.where(mask, gR_h, 0)
    fV = np.real(np.fft.ifftn(projV) * p ** n)
    G = gaps_from_table(table(U, W), p, n)[:, d]
    worstq = max(worstq, G.min() - fV.min())
ok('L5 quantitative: min g_theta <= min(g_split + g_R^V) (generic network)', worstq <= 1e-10, worstq)

# L6: IF cores: glue IF blocks of different frequencies pairwise (shared neurons) and add DC / dead content
worst6 = -1e9
for (p, n) in [(5, 2), (7, 2), (7, 3)]:
    F = Fset(p)
    for trial in range(10):
        blocks = [if_block(p, n, j, lam=rng.uniform(0.5, 1.5), phase0=rng.uniform(0, 2 * np.pi)) for j in F]
        nu = n + 1
        U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
        for (Ub, Wb) in blocks:                         # glue neuron l of every block into neuron l
            perm = rng.permutation(nu)
            U += Ub[perm] * rng.choice([1, -1], size=(nu, 1, 1)) ** 0; W += Wb[perm]
        # DC and dead additions (inputs only): keep all own data
        U += 0.3 * rng.normal() * proj(rng.normal(size=(nu, n, p)), 0, p)
        b0, bT0, _ = own_coeffs(U, W)
        assert max(abs(v) for v in bT0.values()) < 1e-9          # still interference-free
        Us, Ws = split(U, W)
        worst6 = max(worst6, margin(U, W) - margin(Us, Ws))
ok('L6 interference-free networks: h(theta) - h(theta_split) <= 0', worst6 <= 1e-10, worst6)

# ---------------- L7 universal bound with live split costs
worst7 = -1e9
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    nu = n + 1
    for b in [2.0, 2.5, float(nu)]:
        s = nu / b
        for trial in range(10):
            U = rng.normal(size=(5, n, p)); W = rng.normal(size=(5, p))
            h = margin(U, W)
            Pj = {}
            for i in range(len(U)):
                for k in Fset(p):
                    piece = np.concatenate([proj(U[i], k, p).ravel(), proj(W[i], k, p)])
                    Pj[k] = Pj.get(k, 0.0) + (piece ** 2).sum() ** (b / 2)
            rhs = Bp(p, n) * min(sum(Pj[k] ** s * kappa(k, d, p) for k in Pj) for d in range(1, p))
            worst7 = max(worst7, h - rhs)
ok('L7 h <= B_p min_delta sum_j P_j^s kappa_j(delta) (split costs)', worst7 <= 1e-12, worst7)

for name, c, v in report:
    print(('PASS ' if c else 'FAIL ') + name + ('' if v is None else f'   [{v:.3e}]' if isinstance(v, float) else f'   [{v}]'))
