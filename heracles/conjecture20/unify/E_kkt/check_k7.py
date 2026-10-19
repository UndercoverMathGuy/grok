# Checks of K2(a) critical values t_j = nu^{nu/2} B_p K_j(mu*) for decoupled IF unions with non-uniform allocations,
# and of K7: for any probability q on argmin g (delta-marginal mu):
#   sum_i rho_i^nu iota_i = h - sum_delta mu_delta E_x g(.,delta) = -Det_mu <= 0,  iota_i = t_i - c_i,
#   and in class C:  Det_mu >= sum_delta mu_delta max_{l,T} |beta_{l,T}| sqrt(kappa_l(delta)/2).
import numpy as np, itertools
from klib import *
from check_lemmas import restricted_programme, programme_dual

rng = np.random.default_rng(7)
print('== K2(a): critical values of decoupled IF unions')
for (p, n, b, S) in [(11, 2, 2.2, [1, 2, 4]), (13, 2, 2.4, [1, 2, 3, 5]), (7, 3, 2.5, [1, 3])]:
    nu = n + 1; s = nu / b
    G, y = restricted_programme(p, s, S)
    net = if_network(p, n, y, b)
    mu, _, _ = programme_dual(p, s, y)
    Q = np.tile(mu / net.N, (net.N, 1))
    t, cres, M = critical_values(net, Q)
    K = kappa_mat(p).T @ mu
    feats = [int(np.argmax(np.abs(net.atoms()[i]))) for i in range(net.m)]
    pred = np.array([nu ** (nu / 2) * Bp(p, n) * K[f] for f in feats])
    h = net.margin(); tk = nu ** (nu / 2) * (h / net.cost()) * net.a() ** (-(s - 1) / s)
    print(f'  ({p},{n},{b},S={S}): max|t/pred-1| = {np.max(np.abs(t/pred-1)):.1e}; max|t/t_KKT-1| = {np.max(np.abs(t/tk-1)):.1e}; '
          f'critical-point residual {cres.max():.1e}; distinct K_j(mu*)/K_max on S = {np.round(K[np.array(S)-1]/K.max(),5)}')

print('== K7: interference parts of critical values (class C, q uniform on argmin g)')
worst_id = 0; worst_det = -np.inf; nonIF_strict = True
for trial in range(150):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])); F = (p - 1) // 2; m = int(rng.integers(2, 7))
    a_ = np.arange(p); U = []; W = []
    for i in range(m):
        j = int(rng.integers(1, F + 1))
        U.append([rng.standard_normal() * np.cos(2 * np.pi * j * a_ / p) + rng.standard_normal() * np.sin(2 * np.pi * j * a_ / p) for t in range(n)])
        W.append(rng.standard_normal() * np.cos(2 * np.pi * j * a_ / p) + rng.standard_normal() * np.sin(2 * np.pi * j * a_ / p))
    net = Net(np.array(U), np.array(W), 2.0)
    g = net.gaps(); h = g.min()
    A = np.argwhere(g <= h + 1e-12 * abs(h))
    Q = np.zeros_like(g); Q[A[:, 0], A[:, 1]] = 1.0 / len(A)
    mu = Q.sum(0)
    t, _, _ = critical_values(net, Q)
    rho = np.prod(np.linalg.norm(net.U, axis=2), axis=1) * np.linalg.norm(net.W, axis=1)
    c = np.array([T for T in (np.sum(np.real(np.conj(net.atoms()[i] / rho[i]) * np.array([np.sum(mu * (1 - np.exp(-2j * np.pi * l * np.arange(1, p) / p))) for l in range(1, F + 1)]))) for i in range(net.m))])
    iota = t - c
    lhs = np.sum(rho * iota); Det = np.sum(mu * g.mean(0)) - h
    worst_id = max(worst_id, abs(lhs + Det) / (1 + abs(Det)))
    sf = net.signflip(); kap = kappa_mat(p)
    det_lb = np.sum([mu[d - 1] * max(abs(sf[(l, T)]) * np.sqrt(kap[d - 1, l - 1] / 2) for l in range(1, F + 1) for T in itertools.product([0, 1], repeat=n) if any(T)) for d in range(1, p)])
    worst_det = max(worst_det, det_lb - Det)
print(f'  max |sum rho^nu iota + Det_mu| (rel) = {worst_id:.1e};  max (detection lower bound - Det_mu) = {worst_det:.2e} (must be <= 0)')
