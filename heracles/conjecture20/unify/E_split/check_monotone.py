"""Sanity check of R4/R6 calculus on synthetic envelopes (finite families of fixed 'networks' = (h, a) pairs, with
log gamma(theta; s) = log h - s log sum_i a_i^{1/s}).  W = envelope over A u B, D = envelope over B, with the s=1
maximiser in B (so L(1) = 0).  Checks: right derivative of M = L/(s-1) equals -J/(s-1)^2 with
J = [log W + (s-1) H*_min] - [log D + (s-1) H^D_min]; and the bifurcation inequality H(theta_k, s0) <= H^D(s0) + o(1)."""
import math
import numpy as np

rng = np.random.default_rng(11)
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def curve(h, a, s):
    return math.log(h) - s * math.log((a ** (1 / s)).sum())


def ent(a, s):
    w = a ** (1 / s); w = w / w.sum(); return float(-(w * np.log(w)).sum())


worst = 0; checked = 0; bif = []
for trial in range(200):
    B = [(rng.uniform(0.5, 1.5), rng.uniform(0.1, 1, rng.integers(2, 8))) for _ in range(6)]
    A = [(rng.uniform(0.5, 1.5), rng.uniform(0.1, 1, rng.integers(1, 5))) for _ in range(6)]
    # make the s = 1 optimum lie in B: scale A's h so that every A curve is <= max B at s = 1
    mB1 = max(curve(h, a, 1.0) for h, a in B)
    A = [(h * math.exp(min(0.0, mB1 - curve(h, a, 1.0)) - 1e-3), a) for h, a in A]
    fam = A + B
    S = np.linspace(1.0, 3.0, 2001)
    def env(fm, s):
        vals = [curve(h, a, s) for h, a in fm]; k = int(np.argmax(vals)); return vals[k], fm[k]
    for s in S[1:-1:97]:
        e = 1e-7
        lw, tw = env(fam, s); ld, td = env(B, s)
        L = lw - ld
        Lp = (env(fam, s + e)[0] - env(B, s + e)[0])
        Mp = ((Lp / (s + e - 1)) - (L / (s - 1))) / e
        J = (lw + (s - 1) * ent(tw[1], s)) - (ld + (s - 1) * ent(td[1], s))
        if abs(Mp + J / (s - 1) ** 2) < 1e-3 or True:
            worst = max(worst, abs(Mp + J / (s - 1) ** 2) / (1 + abs(Mp)))
            checked += 1
    # bifurcation: first s0 where L becomes positive
    Ls = np.array([env(fam, s)[0] - env(B, s)[0] for s in S])
    pos = np.nonzero(Ls > 1e-12)[0]
    if len(pos) and pos[0] > 0:
        i0 = pos[0] - 1; s0 = S[i0]
        _, td0 = env(B, s0)
        HD = ent(td0[1], s0)
        for k in [1, 3, 10]:
            sk = S[min(i0 + k, len(S) - 1)]
            _, tk = env(fam, sk)
            bif.append(ent(tk[1], s0) - HD)
say(f"M'_+ = -J/(s-1)^2 on {checked} points of 200 synthetic envelopes: max relative error {worst:.2e} (finite differences, kinks included)")
if bif:
    say(f"bifurcation: winners at s_k just above the first departure point have H(theta_k, s0) - H^D(s0) <= {max(bif):.3e} over {len(bif)} cases (predicted <= o(1))")
open("check_monotone.out", "w").write("\n".join(out) + "\n")
