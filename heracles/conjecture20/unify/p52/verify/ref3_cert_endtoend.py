"""Referee-3: independent check of the HS-certificate identity (cert_hs) and end-to-end comparison of the certified bound
with direct network evaluations at (p, n) = (5, 2).

My own network code (from the BRIEF's definitions only):
   f(x)[c] = sum_i w_i(c) u_i1(x1) u_i2(x2),  y = x1 + x2,  g(x,d) = f(x)[y] - f(x)[y-d],  h = min g,
   beta_ij = (2/125) conj(what_i(j)) uhat_i1(j) uhat_i2(j),  hat = sum_x . rho^{-kx},
   cost P = sum ||omega_i||^b,  gamma = h / P^s,  gamma_IF = 3^{1-s} B_5 Gamma_s,  B_5 = 2*30^{-3/2}, Gamma_s = 2.5*2^{-s}.

Part 1 (identity): for certificates G (from the authors' SDP; validity is re-derived here, not trusted), build
   q(x,d) = v(theta(x))^* G~_d v(theta(x)) / (25 M)  explicitly on the 25 x 4 grid, check q >= 0, sum q = 1,
   then for random networks with beta_2 = 0 (DC + all frequencies, beta_2 killed by a pure-frequency-2 neuron):
     h <= sum q g  (direct)  ==  Re conj(A) a0 + sum_T Re conj(beta_T) a_T + <nonown phi, X>  (my own FFT pairing),
     |<nonown phi, X>| <= ||nonown phi|| * n_mine,  ||nonown phi|| <= R = 2 sum_i sqrt(e_i^2 - |beta_i|^2),
   and compare my (a0, a_T, n) with cert_hs.cert_data's enclosures.
Part 2 (end-to-end): for each network, normalise its own data with the symmetries, compute the certified bound at that point
   (payment on the point box, trivial bound, rigorous eval of a degree-2 certificate solved at that point) and compare with
   the directly evaluated gamma/gamma_IF at s = 3/2, 1.25, 1.
Part 3 (random domain points): sample (alpha, m1>=m2, m12, phi_pay) uniformly, realise the own data by an explicit 4-atom
   network (Hadamard: beta_ab = (A + aB + bC + abD)/4 on sign patterns (a,b)), evaluate gamma/gamma_IF directly, compare.
"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/p52')
from cert_hs import cert_data, eval_cells
from hs2 import elim_ratio
from pay_adapt import payment

p = 5
rho = np.exp(2j * np.pi / p)
X1, X2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
Y = (X1 + X2) % p
B5 = 2 * 30 ** -1.5
KA = 1 - np.cos(2 * np.pi / 5)
rng = np.random.default_rng(2026)


def hat(v):            # v: (..., 5) real -> (..., 5) complex, hat[k] = sum_x v[x] rho^{-kx}
    k = np.arange(p)
    return v @ (rho ** (-np.outer(np.arange(p), k)))


def table(U1, U2, W):
    return np.einsum('ic,ia,ib->abc', W, U1, U2)


def gaps(U1, U2, W):
    f = table(U1, U2, W)
    g = np.zeros((p, p, 4))
    for di, d in enumerate((1, 2, 3, 4)):
        g[:, :, di] = f[X1, X2, Y] - f[X1, X2, (Y - d) % p]
    return g


def atoms(U1, U2, W, j=1):
    w = hat(W); a = hat(U1); b = hat(U2)
    beta = (2 / 125) * np.conj(w[:, j]) * a[:, j] * b[:, j]
    bT = {(1,): (2 / 125) * np.conj(w[:, j]) * a[:, -j % p] * b[:, j],
          (2,): (2 / 125) * np.conj(w[:, j]) * a[:, j] * b[:, -j % p],
          (1, 2): (2 / 125) * np.conj(w[:, j]) * a[:, -j % p] * b[:, -j % p]}
    return beta, bT


def field_coeffs(U1, U2, W):
    """phi_j(chi) for j=1,2 and chi in Z_5^2, from the definition Phi_j(x) = (2/5) rho^{j y} sum_c f(x)[c] rho^{-jc}."""
    f = table(U1, U2, W)
    out = {}
    for j in (1, 2):
        Fx = np.einsum('abc,c->ab', f, rho ** (-j * np.arange(p)))
        Phi = (2 / p) * rho ** (j * Y) * Fx
        ph = np.zeros((p, p), complex)
        for c1 in range(p):
            for c2 in range(p):
                ph[c1, c2] = np.mean(Phi * rho ** (-(c1 * X1 + c2 * X2)))
        out[j] = ph
    return out


def random_net(m, dc=True):
    U1 = rng.normal(size=(m, p)); U2 = rng.normal(size=(m, p)); W = rng.normal(size=(m, p))
    if not dc:
        U1 -= U1.mean(1, keepdims=True); U2 -= U2.mean(1, keepdims=True)
    # kill beta_2 with one pure-frequency-2 neuron: u(x) = cos(4 pi x/5) (uhat(2) = 5/2), w = Re(z rho^{2c})
    w2 = hat(W); a2 = hat(U1); b2 = hat(U2)
    beta2 = np.sum((2 / 125) * np.conj(w2[:, 2]) * a2[:, 2] * b2[:, 2])
    c = np.cos(4 * np.pi * np.arange(p) / p)
    # neuron (c, c, w): beta_2 = (2/125) conj(what(2)) (5/2)^2 ; need = -beta2
    target_what2 = np.conj(-beta2 / ((2 / 125) * 6.25))
    # w(c) = 2/5 Re(conj(t) ... ) : choose w(c) = (2/5) Re(t rho^{2c})  -> what(2) = sum w rho^{-2c} = t (since conj part goes to -2)
    wn = (2 / 5) * np.real(target_what2 * rho ** (2 * np.arange(p)))
    U1 = np.vstack([U1, c]); U2 = np.vstack([U2, c]); W = np.vstack([W, wn])
    return U1, U2, W


def ratio(U1, U2, W, s):
    b = 3 / s
    h = gaps(U1, U2, W).min()
    nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    P = np.sum(nrm2 ** (b / 2))
    return h / (P ** s) / (3 ** (1 - s) * B5 * 2.5 * 2 ** -s), h


def q_grid(Gs_rig):
    """explicit q(x, d) from rigorised Gram matrices (lam, U) -- float evaluation"""
    E = [(a, b) for a in range(3) for b in range(3)]
    th1 = 2 * np.pi * (2 * X1) / 5; th2 = 2 * np.pi * (2 * X2) / 5
    V = np.stack([np.exp(1j * (e[0] * th1 + e[1] * th2)) for e in E], axis=-1)   # (5,5,9)
    w = np.zeros((p, p, 4)); M = 0.0
    for di, G in enumerate(Gs_rig):
        w[:, :, di] = np.real(np.einsum('abe,ef,abf->ab', np.conj(V), G, V))
        M += np.real(np.trace(G))
    return w / (25 * M), w


def rigorised(Gs):
    out = []
    for G in Gs:
        G = (G + G.conj().T) / 2
        lam, U = np.linalg.eigh(G)
        lam = np.maximum(lam, 0.0)
        out.append((U * lam) @ U.conj().T)
    return out


# ------------------------------------------------------------------------------------------------ Part 1
t0 = time.time()
pts = [(0.0, 1.0, 1.0, 1.0, 0.0), (0.3, 0.6, 0.5, 0.8, 0.4), (1.0, 0.2, 0.1, 0.9, -2.0), (0.1, 0.9, 0.9, 0.3, 2.5)]
maxdiff = 0.0; bad = 0
certs = []
for (al, m1, m2, m12, phpay) in pts:
    F = payment(np.array([m1, m2, m12]), np.array([m1, m2, m12]), phpay, phpay)
    val, Gs = elim_ratio(al, m1, m2, m12, phpay - al, F, deg=2, ret=True)
    cd = cert_data(Gs)
    Gr = rigorised(Gs)
    q, w = q_grid(Gr)
    certs.append((cd, Gr, q))
    if q.min() < -1e-12 or abs(q.sum() - 1) > 1e-9:
        bad += 1; print("q not a probability", q.min(), q.sum())
    # my own pairings: X_j(chi) = sum_{x,d} q(x,d) c_j(d) rho^{<chi,x>}
    X = {}
    for j in (1, 2):
        cj = np.array([1 - rho ** (-j * d) for d in (1, 2, 3, 4)])
        Cx = (q * cj[None, None, :]).sum(-1)
        X[j] = np.array([[np.sum(Cx * rho ** (c1 * X1 + c2 * X2)) for c2 in range(p)] for c1 in range(p)])
    a0 = X[1][0, 0]; aT = [X[1][2, 0], X[1][0, 2], X[1][2, 2]]
    own1 = {(0, 0), (2, 0), (0, 2), (2, 2)}
    nn = np.sqrt(sum(abs(X[1][c]) ** 2 for c in np.ndindex(5, 5) if c not in own1) +
                 sum(abs(X[2][c]) ** 2 for c in np.ndindex(5, 5) if c != (0, 0)))
    for mine, iv in [(a0, cd['a0'])] + list(zip(aT, cd['aT'])):
        for comp, ivc in [(mine.real, iv[0]), (mine.imag, iv[1])]:
            lo, hi = float(np.atleast_1d(ivc.lo)[0]), float(np.atleast_1d(ivc.hi)[0])
            maxdiff = max(maxdiff, abs(comp - 0.5 * (lo + hi)))
            if not (lo - 1e-12 <= comp <= hi + 1e-12):
                bad += 1; print("pairing outside enclosure", comp, lo, hi)
    nlo, nhi = float(np.atleast_1d(cd['n'].lo)[0]), float(np.atleast_1d(cd['n'].hi)[0])
    if not (nlo - 1e-12 <= nn <= nhi + 1e-12):
        bad += 1; print("n outside enclosure", nn, nlo, nhi)
    # identity on random networks
    for t in range(40):
        U1, U2, W = random_net(rng.integers(1, 7), dc=(t % 2 == 0))
        beta, bT = atoms(U1, U2, W)
        beta2, _ = atoms(U1, U2, W, j=2)
        assert abs(beta2.sum()) < 1e-12
        g = gaps(U1, U2, W)
        h = g.min(); S = np.sum(q * g)
        A = beta.sum(); BT = [bT[(1,)].sum(), bT[(2,)].sum(), bT[(1, 2)].sum()]
        ph = field_coeffs(U1, U2, W)
        # own coefficient identities (Lemma 6 / R4 step 3)
        assert abs(ph[1][0, 0] - np.conj(A)) < 1e-10 and abs(ph[2][0, 0]) < 1e-10
        for k, c in enumerate([(2, 0), (0, 2), (2, 2)]):
            assert abs(ph[1][c] - np.conj(BT[k])) < 1e-10
        pair = np.real(sum(ph[j][c] * X[j][c] for j in (1, 2) for c in np.ndindex(5, 5)))
        ownpart = np.real(np.conj(A) * a0 + sum(np.conj(BT[k]) * aT[k] for k in range(3)))
        nonown_phi = np.sqrt(sum(abs(ph[1][c]) ** 2 for c in np.ndindex(5, 5) if c not in own1) +
                             sum(abs(ph[2][c]) ** 2 for c in np.ndindex(5, 5) if c != (0, 0)))
        nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
        e = B5 * nrm2 ** 1.5
        R = 2 * np.sum(np.sqrt(np.maximum(e ** 2 - np.abs(beta) ** 2, 0)))
        if not (h <= S + 1e-12): bad += 1; print("h > sum q g")
        if abs(S - pair) > 1e-9 * (1 + abs(S)): bad += 1; print("pairing identity fails", S, pair)
        if not (abs(pair - ownpart) <= nonown_phi * nn + 1e-10): bad += 1; print("CS step fails")
        if not (nonown_phi <= R + 1e-12): bad += 1; print("non-own mass > R", nonown_phi, R)
print(f"Part 1: identity checks bad={bad}, max |my pairing - enclosure centre| = {maxdiff:.2e} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ Part 2/3 helpers
def point_bound(al, m1, m2, m12, phpay):
    """certified bound at a single normalised data point (s = 3/2 dominates; R8)."""
    m = np.array([m1, m2, m12])
    if m.max() >= 1.4:
        F = max(1.0, m.max()) / np.sqrt(3)
        return KA * np.cos(al) / (2.5 * 2 ** -1.5 * F), F, 'large'
    F = payment(m, m, phpay, phpay)
    triv = KA * np.cos(al) / (2.5 * 2 ** -1.5 * F)
    try:
        val, Gs = elim_ratio(al, m1, m2, m12, phpay - al, F, deg=2, ret=True)
        cd = cert_data(Gs)
        r = eval_cells(cd, np.array([al]), np.array([al]), m[None, :], m[None, :], np.array([phpay - al]),
                       np.array([phpay - al]), np.array([F]))[0]
    except Exception:
        r = np.inf
    return min(triv, r), F, 'cert'


def normalise_data(A, B, C, D):
    """apply the symmetries: |A|=1, alpha in [0, pi/2] (conjugation), m1 >= m2 (slot swap), torus: B, C real."""
    sc = abs(A); A, B, C, D = A / sc, B / sc, C / sc, D / sc
    al = np.angle(A)
    if al < 0:
        A, B, C, D = np.conj(A), np.conj(B), np.conj(C), np.conj(D); al = -al
    if abs(C) > abs(B):
        B, C = C, B
    phpay = np.angle(A * D * np.conj(B * C)) if abs(B * C * D) > 0 else 0.0
    return al, abs(B), abs(C), abs(D), phpay


t0 = time.time()
worst_gap = np.inf; maxratio = -np.inf; nchk = 0; maxbound = -np.inf
for t in range(60):
    U1, U2, W = random_net(rng.integers(1, 6), dc=(t % 3 != 0))
    # sometimes make it close to an IF block for feature 1 plus noise
    if t % 4 == 0:
        k = np.arange(p)
        U1 = np.vstack([U1 * 0.05, np.cos(2 * np.pi * k / 5), np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5)])
        U2 = np.vstack([U2 * 0.05, np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5), np.cos(2 * np.pi * k / 5), np.sin(2 * np.pi * k / 5)])
        W = np.vstack([W * 0.05, np.cos(2 * np.pi * k / 5), -np.sin(2 * np.pi * k / 5), -np.sin(2 * np.pi * k / 5), -np.cos(2 * np.pi * k / 5)])
        U1, U2, W = random_net(0) if False else (U1, U2, W)
    beta, bT = atoms(U1, U2, W)
    b2, _ = atoms(U1, U2, W, j=2)
    if abs(b2.sum()) > 1e-9:
        # re-kill beta_2
        c = np.cos(4 * np.pi * np.arange(p) / p)
        tw = np.conj(-b2.sum() / ((2 / 125) * 6.25))
        U1 = np.vstack([U1, c]); U2 = np.vstack([U2, c]); W = np.vstack([W, (2 / 5) * np.real(tw * rho ** (2 * np.arange(p)))])
        beta, bT = atoms(U1, U2, W)
    A = beta.sum(); B = bT[(1,)].sum(); C = bT[(2,)].sum(); D = bT[(1, 2)].sum()
    if abs(A) < 1e-9:
        continue
    al, m1, m2, m12, phpay = normalise_data(A, B, C, D)
    if al > np.pi / 2:
        continue                                  # Re A < 0: h <= 0 by Lemma 2
    bnd, F, kind = point_bound(al, m1, m2, m12, phpay)
    maxbound = max(maxbound, bnd)
    for s in (1.5, 1.25, 1.0):
        r, h = ratio(U1, U2, W, s)
        nchk += 1
        maxratio = max(maxratio, r)
        worst_gap = min(worst_gap, bnd - r)
        if r > bnd + 1e-9:
            print("END-TO-END VIOLATION", s, r, bnd, (al, m1, m2, m12, phpay))
print(f"Part 2: {nchk} (network, s) checks: max ratio {maxratio:.4f}, max point bound {maxbound:.4f}, min(bound - ratio) = {worst_gap:.4f} ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------------------------------------ Part 3
t0 = time.time()
k = np.arange(p)
cosv = np.cos(2 * np.pi * k / 5); sinv = np.sin(2 * np.pi * k / 5)
worst_gap3 = np.inf; maxb3 = -np.inf; maxr3 = -np.inf
for t in range(40):
    al = rng.uniform(0, np.pi / 2); m = np.sort(rng.uniform(0, 1.4, 2))[::-1]; m12 = rng.uniform(0, 1.4); phpay = rng.uniform(-np.pi, np.pi)
    A = np.exp(1j * al); B = m[0]; C = m[1]; D = m12 * np.exp(1j * (phpay - al))
    U1l = []; U2l = []; Wl = []
    for a in (1, -1):
        for b in (1, -1):
            bab = (A + a * B + b * C + a * b * D) / 4
            # neuron: u1 = cos (xi=+1) or sin (xi=-1: uhat(-1)/uhat(1) = -1), same for u2; w chosen so beta = bab
            u1 = cosv if a == 1 else sinv; u2 = cosv if b == 1 else sinv
            h1 = hat(u1[None, :])[0, 1]; h2 = hat(u2[None, :])[0, 1]
            # beta = (2/125) conj(what(1)) h1 h2  ->  what(1) = conj(bab / ((2/125) h1 h2))
            tw = np.conj(bab / ((2 / 125) * h1 * h2))
            wv = (2 / 5) * np.real(tw * rho ** k)
            # balance norms (cost-optimal rescaling at b=2 keeps beta): scale factors with product 1
            n1, n2, n3 = np.linalg.norm(u1), np.linalg.norm(u2), np.linalg.norm(wv)
            gm = (n1 * n2 * n3) ** (1 / 3)
            U1l.append(u1 * gm / n1); U2l.append(u2 * gm / n2); Wl.append(wv * gm / max(n3, 1e-300))
    U1 = np.array(U1l); U2 = np.array(U2l); W = np.array(Wl)
    beta, bT = atoms(U1, U2, W)
    assert abs(beta.sum() - A) < 1e-9 and abs(bT[(1,)].sum() - B) < 1e-9 and abs(bT[(2,)].sum() - C) < 1e-9 and abs(bT[(1, 2)].sum() - D) < 1e-9
    assert abs(atoms(U1, U2, W, j=2)[0].sum()) < 1e-12
    bnd, F, kind = point_bound(al, m[0], m[1], m12, phpay)
    maxb3 = max(maxb3, bnd)
    for s in (1.5, 1.25, 1.0):
        r, h = ratio(U1, U2, W, s)
        maxr3 = max(maxr3, r); worst_gap3 = min(worst_gap3, bnd - r)
        if r > bnd + 1e-9:
            print("DOMAIN-POINT VIOLATION", s, r, bnd)
print(f"Part 3: 40 random domain points x 3 s: max point bound {maxb3:.4f} (<= 0.995 expected), max direct ratio {maxr3:.4f}, "
      f"min(bound - ratio) = {worst_gap3:.4f} ({time.time()-t0:.0f}s)")
