import numpy as np, json, sys
sys.path.insert(0, '.')
from kkt import Net, gamma_IF

def build1d(path, p=None):
    d = json.load(open(path)); p = d.get('p', p); n = d['n']
    lam, lv, av = np.array(d['lam']), np.array(d['logV']), np.array(d['argV'])
    m = len(lam); lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], dtype=float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    U, W = [], []
    for i in range(m):
        u = eps * np.exp(lam[i] * e)
        w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        nu_, nw = np.linalg.norm(u), np.linalg.norm(w)
        t = (nw * nu_ ** n) ** (1.0 / (n + 1))
        U.append(u / nu_ * t); W.append(w / nw * t)
    return Net(U, W, n, 2.0), p, n

if __name__ == '__main__':
    path = sys.argv[1]
    net, p, n = build1d(path)
    c = net.cost(); net.U /= c ** 0.5; net.W /= c ** 0.5  # cost 1 at b=2 (homogeneous degree 2)
    print('cost', net.cost(), 'margin', net.margin())
    from math import cos, pi
    gIF = gamma_IF(p, n, 2.0, 1 - cos(2 * pi / p))
    print('gamma/gIF', net.gamma() / gIF)
    for rtol in [1e-9, 1e-7, 1e-5, 1e-3]:
        A, q, lam, res, h = net.kkt(rtol)
        print('rtol', rtol, '|A|', len(A), 'KKT rel residual', res, 'lam', lam, 'nnz q', (q > 1e-10).sum())
