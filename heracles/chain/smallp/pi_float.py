# Float-only dry run of the pi_cert box cover (counts boxes, finds hard regions). Not a certificate.
import sys, numpy as np
from transfer2 import hol
ORD = [('1', 'F', '2', 'in'), ('1', '2', 'F', 'in'), ('2', 'F', '1', 'in'), ('2', '1', 'F', 'in'), ('F', '1', '2', 'out'), ('F', '2', '1', 'out')]
IX = {'1': 0, '2': 1, 'F': 2}
def psi_lo(jl, jh, lam, Ai, Ao):
    tot = 0.0
    for w, (o, f, c, kind) in zip(lam[:6], ORD):
        if not w: continue
        A1, A2 = (Ai, Ao) if kind == 'in' else (Ao, Ao)
        xl, xh = jl[IX[f]], jh[IX[f]]; yl, yh = jl[IX[o]], jh[IX[o]]; wl, wh = jl[IX[c]], jh[IX[c]]
        dl, dh = yl - A1 * wh, yh - A1 * wl
        absl = 0.0 if dl <= 0 <= dh else min(abs(dl), abs(dh))
        tot += w * max(A1 * xl + A2 * absl, A2 * (yl + A1 * wl) - A1 * xh)
    for w, T in zip(lam[6:9], range(3)):
        tot += w * Ai * jl[T]
    for w, (a, b, A) in zip(lam[9:12], ((2, 1, Ai), (2, 0, Ai), (0, 1, Ao))):
        if not w: continue
        s_lo = jl[a] ** 2 + jl[b] ** 2; s_hi = jh[a] ** 2 + jh[b] ** 2
        h = s_lo ** 1.5 / np.sqrt(s_hi ** 2 + 2 * jh[a] ** 2 * jh[b] ** 2) if s_lo > 0 else 0.0
        tot += w * A * max(jl[a], jl[b], h)
    return tot
def need_hi(e0, e1, jh):
    u = np.sqrt(2 * e1 * (e1 - 1)); B = 3 * e1 ** 2 + 2 * sum(x * x for x in jh) + 2 * u * u + 4 * u * min(jh[0], jh[1])
    L = max(e0 ** (2 / 3), 3 ** (1 / 3) * e0 ** (4 / 3) * (3 / B) ** (1 / 3))
    return 1 - (L / 3 ** (1 / 3)) ** 1.5
def run(Ai, Ao, lam, x0, eta0, min_side=2e-3, cap=3e6):
    S3 = np.sqrt(3); stack = [(1.0, S3, (0.0,) * 3, (S3,) * 3)]; ok = 0; bad = 0; n = 0; minslack = 9
    while stack:
        e0, e1, jl, jh = stack.pop(); n += 1
        if n > cap: return dict(status='cap', ok=ok, stack=len(stack))
        if e1 <= 1 + x0 and np.linalg.norm(jh) <= eta0: ok += 1; continue
        if min(jl) > e1: ok += 1; continue
        P = psi_lo(jl, jh, lam, Ai, Ao); N = need_hi(e0, e1, jh)
        if P >= 1 or P > N + 1e-12: ok += 1; continue
        wd = [e1 - e0] + [b - a for a, b in zip(jl, jh)]
        if max(wd) < min_side: bad += 1; minslack = min(minslack, P - N); continue
        d = int(np.argmax(wd))
        if d == 0: m = (e0 + e1) / 2; stack += [(e0, m, jl, jh), (m, e1, jl, jh)]
        else:
            i = d - 1; m = (jl[i] + jh[i]) / 2; h1 = list(jh); h1[i] = m; l2 = list(jl); l2[i] = m
            stack += [(e0, e1, jl, tuple(h1)), (e0, e1, tuple(l2), jh)]
    return dict(status='done', ok=ok, bad=bad, processed=n, worst_bad_slack=minslack)
if __name__ == '__main__':
    Ai, Ao = float(sys.argv[1]), float(sys.argv[2]); lam = [float(x) for x in sys.argv[3].split(',')]
    print(run(Ai, Ao, lam, 0.2, float(sys.argv[4])))
