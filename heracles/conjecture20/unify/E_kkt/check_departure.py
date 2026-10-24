# Check of the departure lemma K4 (second order in s) on fixed networks, and of the level-crossing exhibit:
# at a programme transition the lower-entropy decoupled branch overtakes the higher-entropy one; both are KKT points
# with x-uniform measures and satisfy every KKT identity.
import numpy as np
from klib import *
from scipy.integrate import quad


def curve(net, s):
    a = net.a(); Z = np.sum(a ** (1 / s)); w = a ** (1 / s) / Z
    H = -np.sum(w * np.log(w)); V = np.sum(w * np.log(w) ** 2) - H ** 2
    return np.log(net.margin()) - s * np.log(Z), H, V


# (1) Taylor identity with integral remainder for phi = log gamma(theta) - log gamma(Theta), random fixed networks
rng = np.random.default_rng(3); mx = 0
for trial in range(20):
    p, n = 5, 2
    U, W = if_block(p, n, 1, 1.0); U2, W2 = if_block(p, n, 2, 0.8)
    Th = Net(np.concatenate([U, U2]), np.concatenate([W, W2]), 2.0)
    th = Net(np.concatenate([U * rng.uniform(0.5, 1.5), U2 * rng.uniform(0.5, 1.5), 0.2 * rng.standard_normal((2, n, p))]),
             np.concatenate([W, W2, 0.2 * rng.standard_normal((2, p))]), 2.0)
    if th.margin() <= 0: continue
    s0, s1 = 1.1, 1.45
    f = lambda s: curve(th, s)[0] - curve(Th, s)[0]
    d1 = curve(Th, s0)[1] - curve(th, s0)[1]
    rem = quad(lambda sig: quad(lambda tau: (curve(Th, tau)[2] - curve(th, tau)[2]) / tau, s0, sig, epsabs=1e-13)[0], s0, s1, epsabs=1e-12)[0]
    mx = max(mx, abs(f(s1) - (f(s0) + d1 * (s1 - s0) + rem)))
print(f'(1) Taylor identity phi(s1) = phi(s0) + (H_Theta - H_theta)(s0)(s1-s0) + int int (Var_Theta - Var_theta)/tau : max error {mx:.1e}')

# (2) level crossing among decoupled IF networks at the p=7 transition s_c ~ 1.5475 (n=3: s ranges to 2)
p, n = 7, 3; nu = n + 1
for (s, lab) in [(1.50, 'left'), (1.5475377, 'at s_c'), (1.60, 'right')]:
    b = nu / s
    vals = []
    for S in ([1, 2, 3], [1, 2]):
        import itertools
        # restricted programme by vertex enumeration
        F = 3; r = 1 / s; K = kappa_mat(p)[:F]; best = -1
        for k in [len(S)]:                      # exact support S (the branch with |S| features)
            for Ss in [tuple(j - 1 for j in S)]:
                for R in itertools.combinations(range(F), k):
                    M = K[np.ix_(R, Ss)]
                    if abs(np.linalg.det(M)) < 1e-12: continue
                    v = np.linalg.solve(M, np.ones(k))
                    if np.any(v <= 1e-12): continue
                    full = np.zeros(F); full[list(Ss)] = v
                    if np.any(K @ full < 1 - 1e-9): continue
                    val = np.sum(v ** r) ** (-s)
                    if val > best: best = val; bv = full
        y = bv / np.sum(bv ** r) ** s
        net = if_network(p, n, y, b)
        act, q, res, h = kkt_measure(net, rtol=1e-9)
        Q = q_full(net, act, q); t, cres, M = critical_values(net, Q)
        w = net.shares(); H = net.H()
        idd = np.log(nu ** (nu / 2) * net.gamma()) - (np.sum(w * np.log(t)) - (s - 1) * H)
        vals.append((S, net.gamma(), H, res, idd, np.sum(w * np.log(t))))
    (S1, g1, H1, r1, i1, e1), (S2, g2, H2, r2, i2, e2) = vals
    print(f'(2) p=7,n=3, s={s} ({lab}): gamma(3 feats)/gamma(2 feats) = {g1/g2:.6f};  H = {H1:.4f} vs {H2:.4f};  E_w log t = {e1:.4f} vs {e2:.4f};  KKT res {r1:.0e},{r2:.0e}; identity {i1:.0e},{i2:.0e}')
