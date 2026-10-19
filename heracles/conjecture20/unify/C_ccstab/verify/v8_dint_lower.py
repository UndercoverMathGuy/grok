# Referee check of Prop O(ii) (class C): d_int N >= min_d sum_l kappa_l(d) E_l / O_d >= kappa_min sqrt(sum E)/(4 sqrt((2^n-1)K')).
import numpy as np, math
from vlib import *
from v2_theoremL import rand_net
rng = np.random.default_rng(5)
w1 = np.inf; w2 = np.inf; cnt = 0
for (p, n, b) in [(3, 2, 2.0), (3, 4, 2.0), (5, 2, 2.0), (5, 3, 2.0), (7, 2, 2.5), (11, 2, 2.0), (5, 4, 2.0)]:
    for t in range(25):
        net, S = rand_net(p, n, b)
        g = net.gaps(); h = g.min(); B = Bp(p, n); G1 = p / (p - 1); K = kappa(p)
        at = net.atoms(); beta = np.real(at.sum(0)); N = B * G1 * net.masses().sum()
        dintN = (K @ beta).min() - h
        Phi = net.fields(); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
        Od = np.array([max(np.ptp(g[:, i]), np.ptp(g[:, p - 2 - i])) for i in range(p - 1)])
        mid = min((K[i] * E).sum() / Od[i] if Od[i] > 0 else 0 for i in range(p - 1))
        Kp = (E > 1e-12 * max(E.max(), 1e-300)).sum()
        low = (1 - math.cos(2 * math.pi / p)) * math.sqrt(E.sum()) / (4 * math.sqrt((2 ** n - 1) * max(Kp, 1)))
        sc = np.abs(beta).sum() + 1e-300
        w1 = min(w1, (dintN - mid) / sc); w2 = min(w2, (mid - low) / sc); cnt += 1
print('networks %d; min[d_int N - middle]/sum|A| = %.2e ; min[middle - final]/sum|A| = %.2e' % (cnt, w1, w2))
