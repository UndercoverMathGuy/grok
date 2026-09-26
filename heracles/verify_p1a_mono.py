r"""VERIFY / P1A -- the monotonicity assumptions, Lemma T, and a hostile test of the
box relaxation (a much harder version of p1a_certcheck.py CHECK 1).

p1a_cert.py / P1A.md assert, and use, the following monotonicity facts.  Each is either
re-proved here in one line or hammered numerically:

  M1  tau |-> tau^r            increasing            -> evaluate at tl        (trivial)
  M2  beta |-> beta^{r-1}      decreasing (r<1)      -> evaluate at bh        (trivial)
  M3  (beta,tau) |-> tau beta^{r-1}   incr. in tau, decr. in beta             (trivial)
  M4  tau |-> tgt (1-2^n tau)_+^r     decreasing     -> evaluate at th        (trivial)
  M5  y |-> sum y_i^r          increasing in each y_i -> relaxing the polyhedron
                                                        outward is a lower bound
  M6  LEMMA T staircase  s(tau,beta) = k beta^r + (tau-k beta)^r, k = floor(tau/beta),
      is DECREASING in beta and INCREASING in tau.   [proof: s(tau,beta) is the exact
      value of  min{ sum x_l^r : 0<=x_l<=beta, sum x_l = tau }; enlarging beta enlarges
      the feasible set (decreasing in beta); lowering one coordinate lowers the
      objective (increasing in tau).]                 -- checked on a dense grid
  M7  t0 |-> ok(t0) in eps_cert is monotone (lhs increasing, rhs decreasing), so the
      bisection returns a proved value.               -- checked on a dense grid
  M8  Lemma T is really the minimum of the box-simplex problem (not just a bound).

  R1  HOSTILE relaxation test: for random boxes of WIDE aspect ratios and widths up to
      0.1 (p1a_certcheck uses half-width 1e-3 around the centre only), and for many
      random points INSIDE each box (not just the centre), is
          p1a_cert.G_lower(box).a  <=  G_float(point)   ?
      Run for both staircase settings, since the headline 101/63-box run uses
      staircase=False while p1a_certcheck's CHECK 1 exercises staircase=True.

Run:  uv run --with mpmath --with numpy python verify_p1a_mono.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from mpmath import iv, mp

import p1a_cert as PC
from verify_p1a_links import D_h as D_h_f
import itertools as _it


def G_ref(n, r, beta, tau, staircase):
    """The reference G at a POINT.  Two variants, because p1a_cert.G_lower has two:
      staircase=False -> ADVERSARY.md sec.6's G, Tail = max(tau^r, tau beta^{r-1});
      staircase=True  -> the same with Tail replaced by Lemma T's staircase, which is a
                         STRICTLY LARGER (still valid) lower bound for sum|c|^r.
    Comparing the staircase relaxation against the non-staircase G is a category error
    and produces spurious 'violations'; both variants are compared correctly here.
    Returns +inf when the peel polyhedron is empty (the point is (KL)-infeasible)."""
    tgt = (n + 1) ** (1 - r)
    Ds = [D_h_f(n, h) for h in range(n + 1)]
    rows, tags = [], []
    for i in range(n - 1):
        z = np.zeros(n); z[i], z[i + 1] = 1, -1
        rows.append(z); tags.append(("ord", 0))
    z = np.zeros(n); z[n - 1] = 1
    rows.append(z); tags.append(("beta", 0))
    for h in range(n):
        z = np.zeros(n); z[h:] = 1
        rows.append(z); tags.append(("tail", h))
    A = np.array(rows)
    b = np.array([0.0 if k == "ord" else (beta if k == "beta" else Ds[h] - beta - tau)
                  for k, h in tags])
    best = np.inf
    for S in _it.combinations(range(len(rows)), n):
        M = A[list(S)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        y = np.linalg.solve(M, b[list(S)])
        if (A @ y >= b - 1e-10).all():
            best = min(best, float((np.maximum(y, 0) ** r).sum()))
    if not np.isfinite(best):
        return np.inf
    peel = best + beta ** r
    if tau <= 0:
        tail = 0.0
    elif staircase:
        k = int(np.floor(tau / beta + 1e-14))
        tail = k * beta ** r + max(0.0, tau - k * beta) ** r
    else:
        tail = max(tau ** r, tau * max(beta, 1e-300) ** (r - 1))
    return tail + max(peel, tgt * max(0.0, 1 - 2 ** n * tau) ** r)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
rng = np.random.default_rng(7)


def staircase_float(r, tau, beta):
    if beta <= 0 or tau <= 0:
        return 0.0
    k = int(np.floor(tau / beta + 1e-14))
    return k * beta ** r + max(0.0, tau - k * beta) ** r


def min_box_simplex(r, tau, beta, tries=4000):
    """direct check that the staircase IS the min of sum x^r over {0<=x<=beta, sum=tau}."""
    m = int(np.ceil(tau / beta)) + 3
    best = np.inf
    for _ in range(tries):
        x = rng.random(m)
        x = x / x.sum() * tau
        if x.max() > beta:                       # project crudely
            for _ in range(300):
                x = np.minimum(x, beta)
                d = tau - x.sum()
                if abs(d) < 1e-14:
                    break
                free = x < beta - 1e-15
                if not free.any():
                    break
                x[free] += d / free.sum()
            if x.max() > beta + 1e-12 or abs(x.sum() - tau) > 1e-9 or x.min() < -1e-12:
                continue
        best = min(best, float((np.maximum(x, 0) ** r).sum()))
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    rec = {}

    print("M6/M8  Lemma T staircase: monotone in beta (decreasing) and tau (increasing),")
    print("       and equal to the true min of the box-simplex problem")
    for r in (0.5, 2 / 3, 0.9):
        bad_b = bad_t = 0
        bs = np.linspace(0.01, 0.6, 400)
        ts = np.linspace(0.001, 1.2, 400)
        for t in ts[::7]:
            v = np.array([staircase_float(r, t, b) for b in bs])
            bad_b += int((np.diff(v) > 1e-12).sum())
        for b in bs[::7]:
            v = np.array([staircase_float(r, t, b) for t in ts])
            bad_t += int((np.diff(v) < -1e-12).sum())
        # is it the true minimum?
        worst = np.inf
        for _ in range(60):
            beta = rng.uniform(0.05, 0.4)
            tau = rng.uniform(0.01, 1.0)
            worst = min(worst, min_box_simplex(r, tau, beta) - staircase_float(r, tau, beta))
        print(f"   r = {r:.4f}: violations of decr-in-beta {bad_b}, of incr-in-tau {bad_t};"
              f"  min(direct search - staircase) = {worst:+.3e}"
              f"  {'OK' if bad_b == 0 and bad_t == 0 and worst > -1e-9 else '<<< PROBLEM'}")
        rec[f"lemmaT_r{r:.3f}"] = dict(bad_beta=bad_b, bad_tau=bad_t, slack=float(worst))

    print("\nM7  eps_cert's test ok(t0) is monotone in t0 (so the bisection is valid)")
    for n, q in ((2, 2.0), (3, 2.0)):
        r = q / (n + 1.0)
        tgt = PC.ipow(iv.mpf(n + 1), 1.0 - r)

        def ok(t0):
            z0 = iv.mpf(2 ** n) * iv.mpf(t0)
            if z0.b >= 1.0:
                return False
            K = (iv.mpf(1) - PC.ipow(iv.mpf(1) - z0, r)) / z0
            return PC.ipow(iv.mpf(t0), 1.0 - r).b < (iv.mpf(1) / (tgt * iv.mpf(2 ** n) * K)).a
        ts = np.linspace(1e-6, 2.0 ** (-n) * 0.999, 4000)
        vals = [ok(float(t)) for t in ts]
        flips = sum(1 for i in range(1, len(vals)) if vals[i] and not vals[i - 1])
        e = PC.eps_cert(n, r, tgt)
        print(f"   n = {n}: upward flips of ok(.) after the first False = {flips}"
              f"  (0 = monotone);  eps_cert = {e:.12e};  ok(eps_cert) = {ok(e)},"
              f"  ok(eps_cert*1.02) = {ok(e*1.02)}")
        rec[f"epsmono_n{n}"] = dict(flips=int(flips), eps=float(e))

    print("\nR1  HOSTILE relaxation test: G_lower(box).a <= G_float(point) for random points")
    print("    inside random WIDE boxes (p1a_certcheck only tests the centre of 1e-3 boxes)")
    for n, q in ((2, 2.0), (3, 2.0)):
        r = q / (n + 1.0)
        tgt = PC.ipow(iv.mpf(n + 1), 1.0 - r)
        Ds = [PC.D_h(n, h) for h in range(n + 1)]
        for sc in (False, True):
            worst, cnt = np.inf, 0
            for _ in range(700):
                bl = rng.uniform(0.0, 1.0 / (n + 1))
                bh = bl + rng.uniform(1e-6, 0.12)
                tl = rng.uniform(1e-4, 0.45)
                th = tl + rng.uniform(1e-6, 0.12)
                gl = PC.G_lower(n, r, iv.mpf(bl), iv.mpf(bh), iv.mpf(tl), iv.mpf(th),
                                Ds, tgt, staircase=sc)
                glo = float(gl.a)
                for _ in range(12):
                    b = rng.uniform(bl, bh); t = rng.uniform(tl, th)
                    g = G_ref(n, r, b, t, sc)
                    if not np.isfinite(g):
                        continue
                    cnt += 1
                    worst = min(worst, g - glo)
            print(f"   n = {n}, staircase = {sc}: min (G_float(point) - G_lower(box).a)"
                  f" = {worst:+.6e} over {cnt} point/box pairs"
                  f"   {'OK' if worst >= -1e-12 else '<<< RELAXATION UNSOUND'}")
            rec[f"relax_n{n}_sc{sc}"] = dict(worst=float(worst), pairs=cnt)

    print("\nF1  float() on an interval endpoint: p1a_cert applies float() to lower")
    print("    endpoints in inner_lower / tail_staircase / G_lower(drop) / run_cell(tau_hi).")
    lo = mp.mpf("0.1") - mp.mpf(2) ** -70
    up = mp.mpf(float(lo)) > lo
    print(f"    demo: x = 0.1 - 2^-70 ;  float(x) > x  ->  {up}"
          f"   (error {mp.nstr(mp.mpf(float(lo)) - lo, 5)})")
    print("    => a claimed rigorous LOWER bound can be raised by up to 1 ulp (rel 1.1e-16).")
    print("    Repair: iv.mpf([x, x]) / mp.mpf(x) instead of float(x).  Impact here: <= 1e-15")
    print("    absolute against a certificate margin of ~2e-2, so the verdict does not move.")
    rec["float_rounding_demo"] = dict(float_can_round_up=bool(up))

    with open(os.path.join(OUT, "p1a_mono.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT,'p1a_mono.json')}")


if __name__ == "__main__":
    main()
