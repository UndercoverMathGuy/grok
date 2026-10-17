# Cross-feature test at p=5: slot-symmetric neurons with 1-D tilted sign-vector inputs (type A: freq-1 pattern,
# type B: relabelled a->2a), outputs w in span{freq1, freq2}; exact LP for outputs at fixed slopes.
import math, sys, itertools, numpy as np
from scipy.optimize import linprog
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from famp import counts, log_gamma_if

def sign_pattern(p):
    eps = np.array([1 if a < p / 2 else -1 for a in range(p)]); e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)])
    return eps, e

def inputs(lams, p, typ):
    eps, e = sign_pattern(p)
    U = []
    for lam in lams:
        u = eps * np.exp(lam * e); u = u / np.linalg.norm(u)
        if typ == 2:                     # relabel a -> 2a : u2(a) = u(2a)
            u = u[(2 * np.arange(p)) % p]
        U.append(u)
    return np.array(U)

def lp_joint(U, p, n, b=2.0, freqs=(1, 2), K=16):
    N, y, _ = counts(n, p); N = N.numpy(); y = y.numpy()
    m = len(U); nu = n + 1
    logP = N @ np.log(np.abs(U)).T                      # (classes, m)
    sgn = (-1.0) ** (N @ (U < 0).T)
    # output basis: for each freq l: cos(2 pi l c/p), sin(2 pi l c/p), normalised
    cs = np.arange(p); Bs = []
    for l in freqs:
        for f in (np.cos, np.sin):
            v = f(2 * math.pi * l * cs / p); Bs.append(v / np.linalg.norm(v))
    B = np.array(Bs).T                                  # (p, 2*len(freqs))
    d = B.shape[1]
    rows = []; ccoef = []
    rowmax = logP.max(1)
    for delta in range(1, p):
        D = B[y] - B[(y - delta) % p]                   # (classes, d)
        Pm = sgn * np.exp(logP - rowmax[:, None])        # scaled
        # gap = sum_i Pm[:,i] * D @ z_i  >= h * exp(-rowmax)
        A = (Pm[:, :, None] * D[:, None, :]).reshape(len(y), m * d)
        rows.append(A); ccoef.append(np.exp(-rowmax))
    A = np.vstack(rows); cc = np.concatenate(ccoef); sc = cc.max(); cc = cc / sc
    # variables: z (m*d), t (m), tq (m*len(freqs)), h
    nf = len(freqs); nz = m * d
    nv = nz + m + m * nf + 1
    Aub = []; bub = []
    # -A z + cc h <= 0
    Aub.append(np.hstack([-A, np.zeros((A.shape[0], m + m * nf)), cc[:, None]])); bub.append(np.zeros(A.shape[0]))
    ph = 2 * math.pi * np.arange(K) / K
    for i in range(m):
        for jf in range(nf):
            for k in range(K):
                r = np.zeros(nv); base = i * d + 2 * jf
                r[base] = math.cos(ph[k]); r[base + 1] = math.sin(ph[k]); r[nz + m + i * nf + jf] = -1
                Aub.append(r[None, :]); bub.append([0.0])
        if nf == 2:
            for k in range(9):                           # sqrt(tq1^2+tq2^2) <= t (quarter circle, outer approx)
                a = (math.pi / 2) * k / 8; r = np.zeros(nv)
                r[nz + m + i * nf] = math.cos(a); r[nz + m + i * nf + 1] = math.sin(a); r[nz + i] = -1
                Aub.append(r[None, :]); bub.append([0.0])
        else:
            r = np.zeros(nv); r[nz + m + i * nf] = 1; r[nz + i] = -1; Aub.append(r[None, :]); bub.append([0.0])
    r = np.zeros(nv); r[nz:nz + m] = 1; Aub.append(r[None, :]); bub.append([1.0])
    Aub = np.vstack(Aub); bub = np.concatenate([np.atleast_1d(x) for x in bub])
    cobj = np.zeros(nv); cobj[-1] = -1
    bounds = [(None, None)] * nz + [(0, None)] * (m + m * nf) + [(None, None)]
    res = linprog(cobj, A_ub=Aub, b_ub=bub, bounds=bounds, method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0: return None
    z = res.x[:nz].reshape(m, d); h = res.x[-1] / sc
    a = np.linalg.norm(z, axis=1)
    # true efficiency with l_r cost (b=2): P = sum nu a^{2/nu}, ratio = h / P^{nu/2} / gamma_IF; verify h exactly
    W = z @ B.T                                          # (m, p) output vectors
    f = (sgn * np.exp(logP)) @ W                         # logits (classes, p)
    gaps = np.array([f[np.arange(len(y)), y] - f[np.arange(len(y)), (y - dl) % p] for dl in range(1, p)])
    htrue = gaps.min()
    keep = a > 1e-9 * a.max()
    P = (nu * a[keep] ** (2 / nu)).sum()
    lr = math.log(htrue) - (nu / 2) * math.log(P) - log_gamma_if(p, n, b) if htrue > 0 else -np.inf
    frac2 = (np.linalg.norm(z[:, 2:], axis=1) ** 2).sum() / (a ** 2).sum() if d == 4 else 0.0
    return math.exp(lr), htrue, frac2, z

if __name__ == "__main__":
    p, n = 5, int(sys.argv[1])
    f1 = sys.argv[2]                              # 1-D solution params (lam, lv, av)
    x = np.load(f1); m = len(x) // 3; lam = np.sort(x[:m])
    UA = inputs(lam, p, 1)
    r1 = lp_joint(UA, p, n, freqs=(1,))
    print("n=%d m=%d type A, freq-1 outputs only (1-D family):   eff %.5f" % (n, m, r1[0]))
    r2 = lp_joint(UA, p, n, freqs=(1, 2))
    print("            type A, outputs freq 1+2:                eff %.5f  (freq-2 energy fraction %.3f)" % (r2[0], r2[2]))
    UB = inputs(lam, p, 2)
    r3 = lp_joint(np.vstack([UA, UB]), p, n, freqs=(1, 2))
    print("            types A+B (2m neurons), outputs freq 1+2: eff %.5f  (freq-2 energy fraction %.3f)" % (r3[0], r3[2]))
