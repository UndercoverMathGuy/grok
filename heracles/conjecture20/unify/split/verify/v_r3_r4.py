"""Referee checks of SPLIT R3 (stabiliser, torus average) and R4 (mode/character bijection, channels).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python v_r3_r4.py
"""
import itertools, numpy as np
from vlib import *

rng = np.random.default_rng(99)
out = []
def rep(name, val, ok):
    out.append(('PASS ' if ok else 'FAIL ') + f'{name}   [{val:.3e}]')


def act(v, k, p, z):
    """Multiply vhat(k) by complex unit z (and vhat(-k) by conj z): a rotation of plane k."""
    vh = np.fft.fft(v, axis=-1)
    vh[..., k % p] *= z; vh[..., (-k) % p] *= np.conj(z)
    return np.real(np.fft.ifft(vh, axis=-1))


def reflect(v, k, p, z):
    """Reflection of plane k: vhat(k) -> z * conj(vhat(k))."""
    vh = np.fft.fft(v, axis=-1)
    a = vh[..., k % p].copy()
    vh[..., k % p] = z * np.conj(a); vh[..., (-k) % p] = np.conj(z) * a
    return np.real(np.fft.ifft(vh, axis=-1))


# ---------- R3(a): stabiliser.  Random plane rotations with output compensation psi_k = sum_t phi_{t,k}
p, n, m = 7, 3, 4
U, W = rand_net(rng, m, n, p)
f = table(U, W); b0, bT0 = own_from_table(f)
F = F_(p)
worst_pred = 0; min_change_generic = np.inf; max_change_pi = 0
for trial in range(200):
    phis = rng.uniform(0, 2 * np.pi, size=(len(F), n))
    if trial % 4 == 0:
        phis = np.pi * rng.integers(0, 2, size=(len(F), n))      # rotations by 0 or pi
    U2, W2 = U.copy(), W.copy()
    for kk, k in enumerate(F):
        for t in range(n):
            U2[:, t] = act(U2[:, t], k, p, np.exp(1j * phis[kk, t]))
        W2 = act(W2, k, p, np.exp(1j * phis[kk].sum()))
    bb, bbT = own_from_table(table(U2, W2))
    # predicted: beta_k fixed; beta_{k,T} multiplied by exp(-2i sum_{t in T} phi_{t,k})
    for (k, T), v in bbT.items():
        pred = bT0[(k, T)] * np.exp(-2j * sum(phis[k - 1, t] for t in T))
        worst_pred = max(worst_pred, abs(v - pred))
    worst_pred = max(worst_pred, max(abs(bb[k] - b0[k]) for k in F))
    ch = max(abs(bbT[key] - bT0[key]) for key in bT0)
    if trial % 4 == 0:
        max_change_pi = max(max_change_pi, ch)
    else:
        min_change_generic = min(min_change_generic, ch)
rep('R3a rotations: beta fixed, beta_{k,T} -> e^{-2i sum_T phi} beta_{k,T} (formula error)', worst_pred, worst_pred < 1e-12)
rep('R3a rotations by 0/pi fix all own data (max change)', max_change_pi, max_change_pi < 1e-12)
rep('R3a generic rotations change some beta_{k,T} (min change)', min_change_generic, min_change_generic > 1e-6)
# reflections in one slot plane change beta_k for generic networks, whatever the output map
mn = np.inf
for trial in range(50):
    k = int(rng.choice(F)); t = int(rng.integers(n)); z = np.exp(1j * rng.uniform(0, 2 * np.pi))
    U2 = U.copy(); U2[:, t] = reflect(U[:, t], k, p, z)
    best = np.inf
    for psi in np.linspace(0, 2 * np.pi, 721):      # any output rotation
        W2 = act(W, k, p, np.exp(1j * psi))
        bb, _ = own_from_table(table(U2, W2))
        best = min(best, abs(bb[k] - b0[k]))
    mn = min(mn, best)
rep('R3a slot reflection + best output rotation still changes beta_k (min |change|)', mn, mn > 1e-3)

# ---------- R3(b): torus average keeps exactly clean modes (+ all-DC mode)
p, n, m = 5, 2, 2
U, W = rand_net(rng, m, n, p)
F = F_(p); N = 5
acc = np.zeros((p,) * (n + 1)); cnt = 0
for angs in itertools.product(range(N), repeat=len(F) * n):
    A = np.array(angs).reshape(len(F), n) * 2 * np.pi / N
    U2, W2 = U.copy(), W.copy()
    for kk, k in enumerate(F):
        for t in range(n):
            U2[:, t] = act(U2[:, t], k, p, np.exp(1j * A[kk, t]))
        W2 = act(W2, k, p, np.exp(1j * A[kk].sum()))
    acc += table(U2, W2); cnt += 1
fh = np.fft.fftn(table(U, W)); ah = np.fft.fftn(acc / cnt)
keep = np.zeros(fh.shape, bool)
for j in range(p):
    keep[tuple([j] * n + [(-j) % p])] = True
rep('R3b torus average keeps clean modes (+all-DC)', np.abs(ah[keep] - fh[keep]).max(), np.abs(ah[keep] - fh[keep]).max() < 1e-9)
rep('R3b torus average kills all other modes', np.abs(ah[~keep]).max(), np.abs(ah[~keep]).max() < 1e-9)
# remark check: for an IF network the clean projection IS the split table (so 'cannot attain' is not universal)
from vlib import F_ as _F
nu = n + 1; a = np.arange(p)
Ui = np.zeros((nu, n, p)); Wi = np.zeros((nu, p))
for l in range(nu):
    al = np.pi * l / nu
    for t in range(n):
        Ui[l, t] = np.cos(2 * np.pi * 1 * a / p + al)
    Wi[l] = np.cos(2 * np.pi * 1 * a / p + n * al)
fi = table(Ui, Wi); fih = np.fft.fftn(fi)
clean = np.real(np.fft.ifftn(np.where(keep, fih, 0)))
Us, Ws = split(Ui, Wi)
rep('R3 remark: IF block: clean projection == split table, split cost == cost (b=2)',
    max(np.abs(clean - table(Us, Ws)).max(), abs(cost(Us, Ws, 2) - cost(Ui, Wi, 2))),
    np.abs(clean - table(Us, Ws)).max() < 1e-12)

# ---------- R4(a)/(b)
for (p, n) in [(5, 2), (7, 2), (7, 3), (11, 2), (13, 2), (5, 3)]:
    m = 3; F = F_(p)
    for kind in ['gauss', 'heavydc']:
        U, W = rand_net(rng, m, n, p, kind)
        f = table(U, W); Us, Ws = split(U, W); fs = table(Us, Ws)
        R = f - fs - alldc_table(f)
        Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1)
        nu = n + 1
        b0, bT0 = own_from_table(f)
        err_formula = 0; err_clean = 0; err_mixed_clean = 0; err_chan = 0
        for j in F:
            phi = fcoef(field(f, j)); phiR = fcoef(field(R, j))
            # formula phi_j(chi) = 2 p^{-nu} sum_i what_i(j) prod_t uhat_it(chi_t - j)
            for chi in itertools.product(range(p), repeat=n):
                v = 2 * p ** (-nu) * sum(Wh[i, j] * np.prod([Uh[i, t, (chi[t] - j) % p] for t in range(n)]) for i in range(m))
                err_formula = max(err_formula, abs(v - phi[chi]))
            for Tb in itertools.product([0, 1], repeat=n):
                chi = tuple((2 * j) % p if Tb[t] else 0 for t in range(n))
                T = tuple(t for t in range(n) if Tb[t])
                ref = np.conj(b0[j]) if not T else np.conj(bT0[(j, T)])
                err_clean = max(err_clean, abs(phi[chi] - ref))
                err_mixed_clean = max(err_mixed_clean, abs(phiR[chi]))
        # channels at +-chi_{l,T}: mixed part of g_hat_delta equals the 2|F|-1 listed channels
        rho = np.exp(2j * np.pi / p)
        for d in range(1, p):
            gR = ghat(f, d) - ghat(fs, d)
            for l in F:
                for r_ in range(1, n + 1):
                    for T in itertools.combinations(range(n), r_):
                        for sg in (1, -1):
                            chi = tuple((sg * 2 * l) % p if t in T else 0 for t in range(n))
                            pred = 0; nch = 0
                            for j in F:
                                cj = 1 - rho ** (-j * d)
                                # direct: phi_j(chi), input q = chi - j ; conj: conj(phi_j(-chi) c_j)
                                for (sgn, conj_) in ((1, False), (-1, True)):
                                    ch = tuple((sgn * c) % p for c in chi)
                                    q = [(ch[t] - j) % p for t in range(n)]
                                    if all(qq in (j % p, (-j) % p) for qq in q):
                                        continue
                                    # channel input on T must be |3l| (j=l) or |2l -+ j| (j != l); off T on-frequency
                                    lab = lambda x: min(x % p, (-x) % p)
                                    qT = {lab(q[t]) for t in T}
                                    assert len(qT) == 1
                                    qa = qT.pop()
                                    allowed = {lab(3 * l)} if j == l else {lab(2 * l - j), lab(2 * l + j)}
                                    assert qa in allowed, (p, l, j, qa)
                                    assert all(lab(q[t]) == j for t in range(n) if t not in T)
                                    nch += 1
                                    c = 2 * p ** (-nu) * sum(Wh[i, j] * np.prod([Uh[i, t, q[t]] for t in range(n)]) for i in range(m))
                                    pred += 0.5 * (np.conj(c * cj) if conj_ else c * cj)
                            assert nch == 2 * len(F) - 1, nch
                            err_chan = max(err_chan, abs(gR[chi] - pred))
        tag = f'(p={p},n={n},{kind})'
        rep('R4a phi_j formula vs FFT ' + tag, err_formula, err_formula < 1e-12)
        rep('R4a phi_j(0)=conj beta, phi_j(2j1_T)=conj beta_T ' + tag, err_clean, err_clean < 1e-12)
        rep('R4a mixed table has no content on {0,2j}^n ' + tag, err_mixed_clean, err_mixed_clean < 1e-12)
        rep('R4b mixed g_hat at +-chi_{l,T} = sum of exactly 2|F|-1 channels ' + tag, err_chan, err_chan < 1e-12)
        rep('R4a E_x g_R = 0 ' + tag, np.abs(gaps(R).mean(tuple(range(n)))).max(), np.abs(gaps(R).mean(tuple(range(n)))).max() < 1e-12)

# R4(c): p = 3 decomposition
ok = True
for n in range(1, 7):
    for chi in itertools.product(range(3), repeat=n):
        T = [t for t in range(n) if chi[t] == 2]; S = [t for t in range(n) if chi[t] == 1]
        v = tuple((2 * (t in T) - 2 * (t in S)) % 3 for t in range(n))
        ok &= (v == chi)
rep('R4c Z_3^n = {chi_{1,T} - chi_{1,S}} (n<=6, exhaustive)', 0.0, ok)
for line in out:
    print(line)
