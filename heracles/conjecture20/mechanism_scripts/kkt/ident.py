# Check the KKT margin identity gamma = nu^{-nu/2} (sum_i t_i^{-a})^{-1/a}, a=1/(s-1), t_i = T_q(unit-factor neuron i),
# and split each t_i into clean part (uniform-in-x test with the delta-marginal of q) and interference part.
import numpy as np, sys
sys.path.insert(0, '.')
from kkt import Net
for f in sys.argv[1:]:
    d = np.load(f); net = Net(d['U'], d['W'], int(d['n']), 2.0); p, n, m, nu = net.p, net.n, net.m, net.nu
    b = 2.0; s = nu / b; a = 1 / (s - 1)
    A, q, lam, res, h = net.kkt(1e-7)
    Gi = net.neuron_gaps()     # classes x p x m
    norms = np.sqrt(n * (net.U ** 2).sum(1) + (net.W ** 2).sum(1))
    Tq = np.array([sum(qq * Gi[k, dl, i] for (k, dl), qq in zip(A, q)) for i in range(m)])
    mu = np.zeros(p)
    for (k, dl), qq in zip(A, q): mu[dl] += qq
    Tu = np.array([sum(mu[dl] * (net.cs * Gi[:, dl, i]).sum() / p ** n for dl in range(1, p)) for i in range(m)])
    sc = (norms / np.sqrt(nu)) ** nu
    t = Tq / sc; tc = Tu / sc
    gam = net.gamma(); gid = nu ** (-nu / 2) * ((t ** (-a)).sum()) ** (-1 / a)
    print(f, 'p', p, 'n', n, 'm', m, 'gamma', gam, 'identity', gid, 'rel err', abs(gam - gid) / gam)
    print('  cost-share check T_q(w_i)/((h/P)||w_i||^b):', np.round(Tq / (h / net.cost() * norms ** b), 8))
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    print('  t_i/(nu^{nu/2} B_p):', np.round(t / (nu ** (nu / 2) * Bp), 4))
    print('  clean part:', np.round(tc / (nu ** (nu / 2) * Bp), 4), ' interference part:', np.round((t - tc) / (nu ** (nu / 2) * Bp), 4))
    print('  mu(delta)', np.round(mu[1:], 4))
