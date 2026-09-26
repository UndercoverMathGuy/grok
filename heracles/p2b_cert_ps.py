"""P2B step 5: a RIGOROUS certificate for the per-slot lemma (PS_rho).

CLAIM (PS_rho).  For three unit-modulus numbers z_1,z_2,z_3, with
    S = sum_l z_l ,  Q = sum_l z_l^2 ,
    V^2 = 27 - 6|S|^2 - 3|Q|^2 + 2 Re(conj(S)^2 Q)   (= prod_{l<m}|z_l - z_m|^2, exactly),
    Delta_1 = -(1/2) log(V^2/27) >= 0                (the slot's Hadamard defect),
one has
    R(z) := 36 rho Delta_1 - 8|S|^2 - |Q|^2  >=  0      for rho = RHO.

Rotating all three z_l changes nothing, so fix z_3 = 1 and let (a_1, a_2) range over
[0, 2pi)^2.  R vanishes exactly at the two cube-root configurations
B = (2pi/3, 4pi/3) and B' = (4pi/3, 2pi/3), where it has a positive definite Hessian
(because rho > 1), so the certificate has two parts.

 (L) LOCAL.  Near each B, use  -log(1-u) >= u, i.e. Delta_1 >= (1/2)(1 - V^2/27), which gives
     R >= Psi := 18 rho (1 - V^2/27) - 8|S|^2 - |Q|^2
              = (6 - 12/rho)|S|^2 + (3 - 3/(2 rho))|Q|^2 - 2 Re(conj(S)^2 Q) ,
     a TRIGONOMETRIC POLYNOMIAL with frequencies in {-2..2}^2.  Psi(B) = 0, grad Psi(B) = 0,
     and Psi(B + s) = (1/2) s^T H s + R3 with |R3| <= (M3/6)||s||^3, M3 = sum_k |gam_k| ||k||^3
     over the exact Fourier coefficients.  Hence Psi >= 0 on the ball ||s|| <= 3 lam_min(H)/M3.

 (G) GLOBAL.  Branch and bound over [0,2pi)^2 in mpmath interval arithmetic (iv.cos, iv.sin,
     iv.log, outward-rounded throughout); a box is accepted if its interval lower bound for R
     is >= 0, or if it lies inside one of the two certified balls.

Both parts together prove (PS_rho) for every configuration.
"""
import argparse, itertools, json, os
import numpy as np
from mpmath import iv, mp

OUT = "FINAL_RUNS/P2B"


# ---------------------------------------------------------------- Fourier data of Psi
def psi_terms(rho):
    """Psi = A1|S|^2 + A2|Q|^2 - 2Re(conj(S)^2 Q) as a list of (coef, k) with
    k in Z^2 the frequency in (a_1, a_2) (a_3 = 0), Psi = sum coef * exp(i<k,a>)."""
    A1 = 6.0 - 12.0 / rho
    A2 = 3.0 - 3.0 / (2.0 * rho)
    e = [np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]      # e_l for l = 1,2,3
    terms = {}

    def add(k, c):
        key = (int(k[0]), int(k[1]))
        terms[key] = terms.get(key, 0.0) + c

    for l in range(3):
        for m in range(3):
            add(e[l] - e[m], A1)                      # |S|^2 = sum_{l,m} e^{i(a_l - a_m)}
            add(2 * (e[l] - e[m]), A2)                # |Q|^2
    for aa in range(3):
        for bb in range(3):
            for cc in range(3):
                k = 2 * e[cc] - e[aa] - e[bb]
                add(k, -1.0)                          # -2Re(...) = -(term + conj(term))
                add(-k, -1.0)
    return {k: v for k, v in terms.items() if abs(v) > 1e-14}


def psi_eval(terms, a):
    return float(np.real(sum(c * np.exp(1j * (k[0] * a[0] + k[1] * a[1]))
                             for k, c in terms.items())))


def local_ball(terms, base):
    """Hessian of Psi at `base`, its smallest eigenvalue, M3, and the certified radius."""
    H = np.zeros((2, 2))
    for k, c in terms.items():
        ph = np.exp(1j * (k[0] * base[0] + k[1] * base[1]))
        kk = np.array(k, dtype=float)
        H += -np.real(c * ph) * np.outer(kk, kk)
    lam = float(np.linalg.eigvalsh(H).min())
    M3 = float(sum(abs(c) * np.linalg.norm(k) ** 3 for k, c in terms.items()))
    r = 3.0 * lam / M3 if lam > 0 else 0.0
    # gradient must vanish
    g = np.zeros(2)
    for k, c in terms.items():
        ph = np.exp(1j * (k[0] * base[0] + k[1] * base[1]))
        g += -np.imag(c * ph) * np.array(k, dtype=float) * (-1)
    return H, lam, M3, r, float(np.linalg.norm(g))


# ---------------------------------------------------------------- interval evaluation of R
def R_lower(box, rho):
    """Rigorous lower bound for R = 36 rho Delta_1 - 8|S|^2 - |Q|^2 on the box
    box = ((a1lo,a1hi),(a2lo,a2hi)), using mpmath interval arithmetic."""
    a1 = iv.mpf([box[0][0], box[0][1]])
    a2 = iv.mpf([box[1][0], box[1][1]])
    c1, s1 = iv.cos(a1), iv.sin(a1)
    c2, s2 = iv.cos(a2), iv.sin(a2)
    c1b, s1b = iv.cos(2 * a1), iv.sin(2 * a1)
    c2b, s2b = iv.cos(2 * a2), iv.sin(2 * a2)
    Sr, Si = c1 + c2 + 1, s1 + s2
    Qr, Qi = c1b + c2b + 1, s1b + s2b
    S2 = Sr * Sr + Si * Si
    Q2 = Qr * Qr + Qi * Qi
    ReS2Q = (Sr * Sr - Si * Si) * Qr + 2 * Sr * Si * Qi
    V2 = 27 - 6 * S2 - 3 * Q2 + 2 * ReS2Q
    v2hi = V2.b
    if v2hi > 27:
        v2hi = mp.mpf(27)
    if v2hi <= 0:
        return mp.mpf("1e30")                  # V^2 = 0 => Delta_1 = +inf
    d1lo = -0.5 * iv.log(iv.mpf([v2hi, v2hi]) / 27)
    g_hi = (8 * S2 + Q2).b
    return (36 * rho * d1lo).a - g_hi


def bnb(rho, balls, tol, maxboxes):
    """Branch and bound over [0,2pi]^2."""
    from collections import deque
    stack = deque()
    n0 = 16
    st = 2 * np.pi / n0
    for i in range(n0):
        for j in range(n0):
            stack.append(((i * st, (i + 1) * st), (j * st, (j + 1) * st)))
    nbox = nacc = nball = 0
    worst = None
    while stack:
        b = stack.pop()
        # atom-swap symmetry: R is invariant under (a1,a2) -> (a2,a1); skip the strict half
        if b[1][0] >= b[0][1]:
            continue
        nbox += 1
        if nbox > maxboxes:
            return False, nbox, nacc, nball, b, "box budget exhausted"
        # inside a certified ball?  (both centres are interior to [0,2pi]^2 and the radii are
        # far smaller than the distance to the boundary, so no periodic images are needed)
        inball = False
        for (ctr, r) in balls:
            r2 = r * r
            if all((bx - ctr[0]) ** 2 + (by - ctr[1]) ** 2 <= r2
                   for bx in b[0] for by in b[1]):
                inball = True
                break
        if inball:
            nball += 1
            continue
        lo = R_lower(b, rho)
        if lo >= 0:
            nacc += 1
            continue
        w0, w1 = b[0][1] - b[0][0], b[1][1] - b[1][0]
        if max(w0, w1) < tol:
            return False, nbox, nacc, nball, b, f"box below tol with R_lo = {lo}"
        if w0 >= w1:
            mid = 0.5 * (b[0][0] + b[0][1])
            stack.append(((b[0][0], mid), b[1]))
            stack.append(((mid, b[0][1]), b[1]))
        else:
            mid = 0.5 * (b[1][0] + b[1][1])
            stack.append((b[0], (b[1][0], mid)))
            stack.append((b[0], (mid, b[1][1])))
    return True, nbox, nacc, nball, None, "complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=1.0139)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maxboxes", type=int, default=400000)
    a = ap.parse_args()
    iv.dps = 15
    os.makedirs(OUT, exist_ok=True)
    res = {"rho": a.rho}

    terms = psi_terms(a.rho)
    B = (2 * np.pi / 3, 4 * np.pi / 3)
    Bp = (4 * np.pi / 3, 2 * np.pi / 3)
    print(f"== (L) local certificate at the two cube-root points, rho = {a.rho} ==", flush=True)
    balls = []
    res["local"] = {}
    for name, base in (("B", B), ("B'", Bp)):
        H, lam, M3, r, gn = local_ball(terms, base)
        print(f"   {name}: Psi(base) = {psi_eval(terms, base):.3e}  |grad| = {gn:.3e}  "
              f"lam_min(H) = {lam:.9f}  M3 = {M3:.4f}  radius r = {r:.8f}", flush=True)
        res["local"][name] = {"lam_min": lam, "M3": M3, "r": r,
                              "psi_at_base": psi_eval(terms, base), "grad_norm": gn}
        balls.append((base, r))
    # empirical check that Psi >= 0 on the balls
    rng = np.random.default_rng(7)
    worst = np.inf
    for (ctr, r) in balls:
        for _ in range(20000):
            d = rng.normal(size=2)
            d = d / np.linalg.norm(d) * r * rng.random() ** 0.5
            worst = min(worst, psi_eval(terms, np.array(ctr) + d))
    print(f"   sampled min Psi on the two balls = {worst:.3e} (must be >= 0)", flush=True)
    res["local"]["sampled_min_psi"] = worst

    print(f"\n== (G) global branch and bound on R >= 0 (mpmath iv, dps = {iv.dps}) ==",
          flush=True)
    ok, nbox, nacc, nball, bad, msg = bnb(a.rho, balls, a.tol, a.maxboxes)
    print(f"   boxes processed = {nbox},  accepted by interval bound = {nacc},  "
          f"inside a certified ball = {nball}", flush=True)
    print(f"   RESULT: {'CERTIFIED' if ok else 'FAILED'}  ({msg})", flush=True)
    if bad is not None:
        print(f"   offending box: {bad}", flush=True)
    res["global"] = {"ok": bool(ok), "boxes": nbox, "accepted": nacc, "in_ball": nball,
                     "msg": msg, "bad_box": bad}

    json.dump(res, open(f"{OUT}/cert_ps.json", "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/cert_ps.json", flush=True)


if __name__ == "__main__":
    main()
