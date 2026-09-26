"""Certificate of the matched population inequality PI(Psi) (see pi_cert.py for the reduction) with a float64 box cover.
Rigour: every evaluated quantity is a composition of <= ~60 IEEE operations on numbers of size <= 20 (sqrt, pow via libm,
accurate to < 1 ulp), so its absolute error is < 1e-12. A box passes only if (Psi_lo - need_hi) > 1e-9 (or Psi_lo >= 1 + 1e-9),
so the exact inequality holds on it with room. In addition the 3000 tightest boxes and 3000 random boxes are replayed in
mpmath interval arithmetic (pi_cert.box_ok). cmin on the simplex: float B&B with the same margin, then the corner lemma in
mpmath intervals (pi_cert.corner_check).
"""
import sys, json, heapq, random, numpy as np
from pi_float import psi_lo, need_hi
import pi_cert
MARGIN = 1e-9
def cmin_float(lam, Ai, Ao, tol=0.01):
    stack = [(0.0, 1.0, 0.0, 1.0)]; best = 9.0
    while stack:
        a0, a1, b0, b1 = stack.pop()
        if a0 + b0 > 1: continue
        jl = (a0, b0, max(0.0, 1 - a1 - b1)); jh = (a1, b1, max(0.0, 1 - a0 - b0))
        L = psi_lo(jl, jh, lam, Ai, Ao)
        if a1 - a0 < tol and b1 - b0 < tol: best = min(best, L - MARGIN); continue
        if a1 - a0 >= b1 - b0: m = (a0 + a1) / 2; stack += [(a0, m, b0, b1), (m, a1, b0, b1)]
        else: m = (b0 + b1) / 2; stack += [(a0, a1, b0, m), (a0, a1, m, b1)]
    return best
def cover(Ai, Ao, lam, x0, eta0, min_side=1e-3):
    S3 = np.sqrt(3); stack = [(1.0, S3, (0.0,) * 3, (S3,) * 3)]; ok = 0; bad = []; tight = []; sample = []; n = 0
    while stack:
        e0, e1, jl, jh = stack.pop(); n += 1
        if e1 <= 1 + x0 and np.linalg.norm(jh) <= eta0: ok += 1; continue
        if min(jl) > e1: ok += 1; continue
        P = psi_lo(jl, jh, lam, Ai, Ao); N = need_hi(e0, e1, jh)
        if P >= 1 + MARGIN or P - N > MARGIN:
            ok += 1; slack = P - N
            if len(tight) < 3000: heapq.heappush(tight, (-slack, (e0, e1, jl, jh)))
            elif -tight[0][0] > slack: heapq.heapreplace(tight, (-slack, (e0, e1, jl, jh)))
            if random.random() < 3000 / 1e6 and len(sample) < 3000: sample.append((e0, e1, jl, jh))
            continue
        wd = [e1 - e0] + [b - a for a, b in zip(jl, jh)]
        if max(wd) < min_side: bad.append((e0, e1, jl, jh)); continue
        d = int(np.argmax(wd))
        if d == 0: m = (e0 + e1) / 2; stack += [(e0, m, jl, jh), (m, e1, jl, jh)]
        else:
            i = d - 1; m = (jl[i] + jh[i]) / 2; h1 = list(jh); h1[i] = m; l2 = list(jl); l2[i] = m
            stack += [(e0, e1, jl, tuple(h1)), (e0, e1, tuple(l2), jh)]
    return ok, bad, n, [b for _, b in tight], sample
if __name__ == '__main__':
    random.seed(0)
    tag = sys.argv[1]; Ai, Ao = float(sys.argv[2]), float(sys.argv[3]); lam = [float(x) for x in sys.argv[4].split(',')]
    assert all(l >= 0 for l in lam); s = sum(lam); lam = [l / s for l in lam] if s > 1 else lam
    cmin = cmin_float(lam, Ai, Ao); x0 = 0.2; eta0 = 0.01
    while pi_cert.corner_check(cmin, x0, eta0 * 1.05): eta0 *= 1.05
    corner = pi_cert.corner_check(cmin, x0, eta0)
    ok, bad, n, tight, sample = cover(Ai, Ao, lam, x0, eta0)
    replay_fail = sum(1 for (e0, e1, jl, jh) in tight + sample if not pi_cert.box_ok(e0, e1, jl, jh, lam, Ai, Ao, x0, eta0))
    res = dict(tag=tag, A_in=Ai, A_out=Ao, lam=lam, cmin=cmin, corner_x0=x0, corner_eta0=eta0, corner_ok=corner, boxes_processed=n,
               boxes_ok=ok, unresolved=len(bad), mpmath_replayed=len(tight) + len(sample), mpmath_replay_failures=replay_fail,
               CERTIFIED=bool(corner and not bad and replay_fail == 0 and cmin > 0))
    print(json.dumps(res)); json.dump(res, open(f'pi_cert_{tag}.json', 'w'), indent=1)
