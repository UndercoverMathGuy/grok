"""Referee check of E4/E5 calculus on envelopes of genuine temperature curves phi_k(s) = log h_k - s log sum_i a_ki^{1/s}.
W = max_k phi_k (k in family), D = phi_0 (a designated 'decoupled' curve that is optimal at s=1 by construction).
Checks: (1) right derivative of log W = -min H over active curves (Danskin);
(2) M = L/(s-1), L = log W - log D: M'_+ = -J/(s-1)^2 with J = min_active F - F_D;
(3) at the first departure s0 of L from 0, winners just right of s0 have H(theta_k, s0) <= H_D(s0) + o(1)."""
import math
import numpy as np
rng = np.random.default_rng(5)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def curve(h, a, s):
    c = a ** (1 / s); Z = c.sum(); w = c / Z
    return math.log(h) - s * math.log(Z), float(-(w * np.log(w)).sum())

worstD = 0; worstM = 0; bif = []
for env in range(200):
    K = rng.integers(3, 8)
    fam = []
    # decoupled curve: wide (high entropy), optimal at s=1
    a0 = rng.dirichlet(np.ones(rng.integers(6, 12))); fam.append((1.0, a0))
    for k in range(K):
        m = rng.integers(2, 10); a = rng.dirichlet(np.ones(m) * rng.uniform(0.3, 3))
        h = rng.uniform(0.7, 0.999)   # below D at s=1
        fam.append((h, a))
    S = np.linspace(1.0, 2.5, 3001)
    vals = np.array([[curve(h, a, s)[0] for (h, a) in fam] for s in S])
    Hs = np.array([[curve(h, a, s)[1] for (h, a) in fam] for s in S])
    logW = vals.max(1); logD = vals[:, 0]; L = logW - logD
    for q in range(10, len(S) - 10, 97):
        s = S[q]
        act = np.where(vals[q] >= logW[q] - 1e-13)[0]
        dplus = -Hs[q, act].min()
        e = 1e-7
        fd = (max(curve(h, a, s + e)[0] for (h, a) in fam) - logW[q]) / e
        worstD = max(worstD, abs(fd - dplus))
        if s > 1.01:
            J = min(vals[q, k] + (s - 1) * Hs[q, k] for k in act) - (logD[q] + (s - 1) * Hs[q, 0])
            M = lambda ss: (max(curve(h, a, ss)[0] for (h, a) in fam) - curve(*fam[0], ss)[0]) / (ss - 1)
            fdM = (M(s + e) - M(s)) / e
            worstM = max(worstM, abs(fdM + J / (s - 1) ** 2) / (1 + abs(fdM)))
    # first departure
    pos = np.where(L > 1e-12)[0]
    if len(pos):
        q0 = pos[0] - 1; s0 = S[q0]
        for q in range(pos[0], min(pos[0] + 5, len(S))):
            k = int(np.argmax(vals[q]))
            bif.append(curve(*fam[k], s0)[1] - Hs[q0, 0])
say(f"(1) Danskin right derivative of log W vs FD: max err {worstD:.2e}")
say(f"(2) M'_+ = -J/(s-1)^2: max relative err {worstM:.2e}")
say(f"(3) first-departure winners: max H(theta_k, s0) - H_D(s0) = {max(bif):+.4f} over {len(bif)} cases (E5: <= o(1))")
open(__file__.replace('.py', '.out'), 'w').write("\n".join(out) + "\n")
