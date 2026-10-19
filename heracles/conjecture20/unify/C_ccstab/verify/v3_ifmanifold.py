# Referee test: the IF manifold through the aligned IF block contains NON-ALIGNED IF configurations
# (atom phase spread eps, interference exactly 0).  Along this curve:
#   - d_int = 0 and every interference energy E_l = 0 (no first-order detection possible),
#   - RD-IF slack log(gamma_IF/gamma) = O(eps^2) while the distance to the aligned configuration is ~eps,
#   - Theorem L has X < 0 at b = 2 (silent), L2's condition fails.
# Construction (every n): nodes a_k = exp(i alpha_k), k = 0..n, xi_{kt} = a_k for all slots t, so chi_k(T) = a_k^{|T|};
# IF <=> sum_k z_k a_k^j = 0 for j = 1..n  <=> z spans the null space of the n x nu Vandermonde block.
import numpy as np, math, itertools
from vlib import *
from v2_theoremL import ledger

def block_specs(l, n, alphas, scale):
    nu = n + 1
    V = np.array([[np.exp(1j * j * a) for a in alphas] for j in range(1, n + 1)])     # n x nu
    _, _, vh = np.linalg.svd(V); z = np.conj(vh[-1])                                  # null vector
    z = z * np.exp(-1j * np.angle(z.sum()))                                           # make A = sum z real > 0
    z = z / z.sum() * scale
    specs = []
    for k in range(nu):
        rho, th = abs(z[k]), np.angle(z[k])
        amp = (rho * 2 ** (nu - 1)) ** (1 / nu)                                       # atom modulus 2^{1-nu} amp^nu
        phis = np.full(n, -alphas[k] / 2)                                             # xi = exp(-2i phi) = a_k
        specs.append((l, amp * np.ones(nu), phis, phis.sum() - th))                   # atom phase = sum phi - psi = th
    return specs, z

def run(p, n, b, S, y, eps_list, label):
    nu = n + 1; s = nu / b; B = Bp(p, n); GamS = programme_S(p, s, S); gIF = nu ** (1 - s) * B * GamS
    print('--- %s  (p,n,b)=(%d,%d,%g) S=%s' % (label, p, n, b, S))
    base = None; rows = []
    for eps in eps_list:
        specs = []; spread = 0
        for idx, l in enumerate(S):
            alphas = 2 * np.pi * np.arange(nu) / nu
            if idx == 0: alphas = alphas.copy(); alphas[0] += eps          # deform only the first block
            sp, z = block_specs(l, n, alphas, y[idx]); specs += sp
            ph = np.angle(z); spread = max(spread, np.ptp(np.unwrap(ph)))
        net = classC(p, n, specs, b)
        g = net.gaps(); h = g.min(); Phi = net.fields(); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0).sum()
        at = net.atoms(); A = np.abs(at.sum(0)); L = np.abs(at).sum(0)
        par = np.concatenate([net.U.ravel(), net.W.ravel()])
        if base is None: base = par
        d = ledger(net, S)
        K = kappa(p); beta = np.real(at.sum(0)); G1 = p / (p - 1); N = B * G1 * net.masses().sum()
        dint = ((K @ beta).min() - h) / N
        rows.append((eps, spread, np.linalg.norm(par - base), E / A.sum() ** 2, dint, math.log(gIF / net.gamma()), d['X'],
                     np.max((L - A)[A > 0] / L[A > 0]), h > 0))
    print('   eps      phase-spread  |theta-theta0|  E/|A|^2     d_int       RD-IF slack   X(TheoremL)   (L-|A|)/L   h>0')
    for r_ in rows: print('   %.1e  %.3e     %.3e       %.1e   %+.1e   %.4e    %+.3e    %.2e   %s' % r_)
    es = np.array([r_[0] for r_ in rows[1:]]); sl = np.array([r_[5] for r_ in rows[1:]]); dist = np.array([r_[2] for r_ in rows[1:]])
    print('   log-log slope of RD-IF slack vs eps: %.3f ; vs parameter distance: %.3f ; slope of distance vs eps: %.3f'
          % (np.polyfit(np.log(es), np.log(sl), 1)[0], np.polyfit(np.log(dist), np.log(sl), 1)[0], np.polyfit(np.log(es), np.log(dist), 1)[0]))

if __name__ == '__main__':
    eps_list = [0.0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    run(5, 2, 2.0, [1], [1.0], eps_list, 'single feature, n=2 (pi = 3^{1-s}: RD equality case)')
    run(5, 2, 2.0, [1, 2], [1.0, 1.0], eps_list, 'decoupled b=2 maximiser at (5,2): two equal blocks')
    run(7, 3, 2.0, [1], [1.0], eps_list, 'single feature, n=3')
    run(5, 4, 2.0, [1], [1.0], eps_list, 'single feature, n=4')
    run(3, 4, 2.0, [1], [1.0], eps_list, 'p=3, n=4')
