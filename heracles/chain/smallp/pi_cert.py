"""Rigorous certificate of the matched population inequality PI(Psi):
  for every population (atoms c_l, unit v_l1, v_l2) with G = sum c_l,
      sum |c_l|^{2/3} >= 3^{1/3} ( |G| - Psi(|J1|, |J2|, |J12|) )_+^{2/3},   strictly if J != 0,
  Psi(j) = sum_sigma lam_sigma beta_sigma(j) (12 orders; see transfer2.py), with inner/outer depths (A_in, A_out).
Reduction (fable0923 Lemma 3, both chain orders): with G = 1, e = sum|c_l|, u = sqrt(2e(e-1)),
  cost >= LHS(e,j) = max(e^{2/3}, 3^{1/3} e^{4/3} (3/B)^{1/3}),  B = 3e^2 + 2|j|^2 + 2u^2 + 4u min(j1, j2);  j_T <= e.
For e >= sqrt3, LHS >= 3^{1/3} and PI holds. Remaining domain e in [1, sqrt3], j in [0, e]^3:
 * corner {e <= 1 + x0, |j|_2 <= eta0}: Psi(j) >= cmin |j|_1 >= cmin |j|_2 (cmin = min of Psi on the simplex, certified here
   by 2-D interval B&B), and fable0923's corner lemma with A = cmin (Delta >= eta (6A - eta * bracket) > 0, strict);
 * rest: 4-D box cover, each box checked in mpmath interval arithmetic: Psi_lo >= 1 or Psi_lo > need_hi,
   need = 1 - (LHS / 3^{1/3})^{3/2}.
"""
import sys, json, numpy as np
from mpmath import iv, mpf
iv.dps = 25
C3 = iv.mpf(3) ** (iv.mpf(1) / 3)
ORD = [('1', 'F', '2', 'in'), ('1', '2', 'F', 'in'), ('2', 'F', '1', 'in'), ('2', '1', 'F', 'in'), ('F', '1', '2', 'out'), ('F', '2', '1', 'out')]
IX = {'1': 0, '2': 1, 'F': 2}

def lo(x): return x.a if hasattr(x, 'a') else x
def hi(x): return x.b if hasattr(x, 'b') else x
def ivmax_lo(*xs): return max(mpf(lo(x)) for x in xs)

def psi_lo(j, lam, Ai, Ao):
    """j: list of 3 iv intervals. returns a rigorous lower bound (mpf) of Psi over the box."""
    tot = iv.mpf(0)
    for w, (o, f, c, kind) in zip(lam[:6], ORD):
        if w == 0: continue
        A1, A2 = (Ai, Ao) if kind == 'in' else (Ao, Ao)
        x, y, ww = j[IX[f]], j[IX[o]], j[IX[c]]
        opt1 = A1 * x + A2 * abs(y - A1 * ww)
        opt2 = A2 * (y + A1 * ww) - A1 * x
        tot += iv.mpf(w) * ivmax_lo(opt1, opt2)
    for w, T in zip(lam[6:9], range(3)):
        if w: tot += iv.mpf(w) * Ai * iv.mpf(lo(j[T]))
    def hol(x, y):
        s = x * x + y * y
        if hi(s) == 0: return mpf(0)
        h = s ** iv.mpf(1.5) / iv.sqrt(s * s + 2 * x * x * y * y) if lo(s) > 0 else iv.mpf(0)
        return max(mpf(lo(x)), mpf(lo(y)), mpf(lo(h)) if lo(s) > 0 else mpf(0))
    for w, (a, b, A) in zip(lam[9:12], ((2, 1, Ai), (2, 0, Ai), (0, 1, Ao))):
        if w: tot += iv.mpf(w) * A * iv.mpf(hol(j[a], j[b]))
    return mpf(lo(tot))

def need_hi(e, j):
    u = iv.sqrt(2 * e * (e - 1))
    mn = iv.mpf([min(lo(j[0]), lo(j[1])), min(hi(j[0]), hi(j[1]))])
    B = 3 * e * e + 2 * (j[0] * j[0] + j[1] * j[1] + j[2] * j[2]) + 2 * u * u + 4 * u * mn
    br1 = iv.mpf(lo(e)) ** (iv.mpf(2) / 3)
    br2 = C3 * iv.mpf(lo(e)) ** (iv.mpf(4) / 3) * (3 / iv.mpf(hi(B))) ** (iv.mpf(1) / 3)
    L = max(mpf(lo(br1)), mpf(lo(br2)))
    return mpf(hi(1 - (iv.mpf(L) / C3) ** iv.mpf(1.5)))

def box_ok(e0, e1, jl, jh, lam, Ai, Ao, x0, eta0):
    if e1 <= 1 + x0 and float(np.linalg.norm(jh)) <= eta0: return True          # corner lemma region
    if min(jl) > e1: return True                                                 # infeasible (j_T <= e)
    e = iv.mpf([e0, e1]); j = [iv.mpf([a, b]) for a, b in zip(jl, jh)]
    P = psi_lo(j, lam, Ai, Ao)
    return P >= 1 or P > need_hi(e, j)

def corner_check(A, x0, eta0):
    A = iv.mpf(A); x0 = iv.mpf(x0); eta0 = iv.mpf(eta0)
    br = 6 + 3 * A ** 2 + 10 * A ** 2 * x0 + 7 * A ** 2 * x0 ** 2 + 2 * A ** 2 * eta0 ** 2 + 4 * A ** 2 * eta0 * iv.sqrt(2 * x0 * (1 + x0))
    return lo(6 * A) > hi(eta0 * br)

def cmin_cert(lam, Ai, Ao, tol=0.03):
    # min over simplex {m >= 0, sum m = 1} of Psi(m): 2-D interval B&B on (m1, m2), m3 = 1 - m1 - m2
    best_lo = mpf(9); stack = [(0.0, 1.0, 0.0, 1.0)]; certified_lo = mpf(9)
    while stack:
        a0, a1, b0, b1 = stack.pop()
        if a0 + b0 > 1: continue
        m = [iv.mpf([a0, a1]), iv.mpf([b0, b1]), iv.mpf([max(0.0, 1 - a1 - b1), max(0.0, 1 - a0 - b0)])]
        L = psi_lo(m, lam, Ai, Ao)
        if a1 - a0 < tol and b1 - b0 < tol:
            certified_lo = min(certified_lo, L); continue
        # split the longer side
        if a1 - a0 >= b1 - b0: mid = (a0 + a1) / 2; stack += [(a0, mid, b0, b1), (mid, a1, b0, b1)]
        else: mid = (b0 + b1) / 2; stack += [(a0, a1, b0, mid), (a0, a1, mid, b1)]
    return certified_lo

def cover(lam, Ai, Ao, x0, eta0, min_side=2e-3):
    S3 = float(np.sqrt(3)); stack = [(1.0, S3, (0.0, 0.0, 0.0), (S3, S3, S3))]; n_ok = 0; bad = []
    while stack:
        e0, e1, jl, jh = stack.pop()
        if box_ok(e0, e1, jl, jh, lam, Ai, Ao, x0, eta0): n_ok += 1; continue
        widths = [e1 - e0] + [b - a for a, b in zip(jl, jh)]
        if max(widths) < min_side: bad.append((e0, e1, jl, jh)); continue
        d = int(np.argmax(widths))
        if d == 0:
            m = (e0 + e1) / 2; stack += [(e0, m, jl, jh), (m, e1, jl, jh)]
        else:
            i = d - 1; m = (jl[i] + jh[i]) / 2
            jh1 = list(jh); jh1[i] = m; jl2 = list(jl); jl2[i] = m
            stack += [(e0, e1, jl, tuple(jh1)), (e0, e1, tuple(jl2), jh)]
    return n_ok, bad

if __name__ == '__main__':
    Ai, Ao = float(sys.argv[1]), float(sys.argv[2]); lam = [float(x) for x in sys.argv[3].split(',')]
    assert all(l >= 0 for l in lam) and sum(lam) <= 1 + 1e-15
    lam = [l / max(1.0, sum(lam)) for l in lam]
    cmin = cmin_cert(lam, Ai, Ao); print(f"cmin (certified lower bound of Psi on the simplex) = {float(cmin):.5f}", flush=True)
    x0 = 0.2; eta0 = 0.01
    while corner_check(float(cmin), x0, eta0 * 1.1): eta0 *= 1.1
    print(f"corner lemma: A = cmin, x0 = {x0}, eta0 = {eta0:.4f}: {corner_check(float(cmin), x0, eta0)}", flush=True)
    n_ok, bad = cover(lam, Ai, Ao, x0, eta0)
    print(json.dumps(dict(A_in=Ai, A_out=Ao, lam=lam, cmin=float(cmin), eta0=eta0, boxes_ok=n_ok, unresolved=len(bad),
                          example_bad=[(b[0], b[1], list(b[2]), list(b[3])) for b in bad[:3]])), flush=True)
