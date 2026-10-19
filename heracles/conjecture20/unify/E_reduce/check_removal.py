# Removal inequality at a global maximiser: deleting the frequency-j output of every neuron gives theta^{-j} with
#   P(theta^{-j}) <= (1 - E_j) P(theta*)   and hence   h(theta^{-j}) <= h(theta*) (1 - E_j)^s.
# Checked on IF optima (exact global optima at (5,2),(7,2) per p52/GLOBAL numerics) and on found optima.
import numpy as np
from elib import Net, programme_vertices, if_network, Evec, freq_proj, optimise_w, balance, prune, Bp
rng = np.random.default_rng(2)
for (p, n, b) in [(5, 2, 2.5), (7, 2, 2.2), (7, 2, 3.0), (11, 2, 2.0)]:
    nu = n + 1; s = nu / b; F = (p - 1) // 2
    G, maxs, _ = programme_vertices(p, s); y = maxs[0]
    nets = [if_network(p, n, y, b)]
    U = rng.normal(size=(3 * F + 1, n, p)); W = rng.normal(size=(3 * F + 1, p))
    nets.append(prune(balance(optimise_w(U, W, b, np.ones(F))), 1e-8))
    for k, net in enumerate(nets):
        h = net.margin(); P = net.cost(); E = Evec(net)
        out = []
        for j in range(1, F + 1):
            Wm = net.W @ (np.eye(p) - freq_proj(p, j))
            nm = Net(net.U, Wm, b)
            out.append((nm.cost() / P - (1 - E[j - 1]), nm.margin() / h - (1 - E[j - 1]) ** s))
        out = np.array(out)
        print(f"p={p} n={n} b={b} {'IF' if k==0 else 'found'} (gamma/gamma_IF={net.gamma()/(nu**(1-s)*Bp(p,n)*G):.6f}): "
              f"max[P^-j/P-(1-E_j)]={out[:,0].max():+.2e} (<=0), max[h^-j/h-(1-E_j)^s]={out[:,1].max():+.2e} (<=0 at maximisers)")
