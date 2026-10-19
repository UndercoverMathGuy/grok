# depth m_d, sigma_d, skew S_d of the deficit d_d(x) = g(x,d) - E_x g(x,d) on adversarial networks.
import sys, glob, itertools, numpy as np
from math import factorial
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint'); sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import EW
def stats_from_gaps(G, wts):
    # G: (N, p-1) gaps g(x,d); wts: probability weights of the N points
    out = []
    for j in range(G.shape[1]):
        mu = wts @ G[:, j]; d = G[:, j] - mu; m = -d.min(); s2 = wts @ d**2; S = wts @ d**3
        out.append((j+1, G[:, j].min(), mu, m, np.sqrt(s2), S/np.sqrt(s2)**3, m*(s2-m*m) <= S + 1e-12*s2**1.5))
    return out
def slotsym_gaps(U1, W, p, n):
    rows = []; wts = []
    for N in itertools.product(range(n+1), repeat=p-1):
        if sum(N) > n: continue
        cnt = list(N) + [n-sum(N)]; wgt = factorial(n)/np.prod([factorial(c) for c in cnt])
        y = sum(a*cnt[a] for a in range(p)) % p
        z = np.prod([U1[:, a]**cnt[a] for a in range(p)], axis=0); f = W.T @ z
        rows.append([f[y] - f[(y-d) % p] for d in range(1, p)]); wts.append(wgt)
    wts = np.array(wts); return np.array(rows), wts/wts.sum()
import exact_certificate as ec
for n in [11, 13]:
    dd = ec.NETWORKS[n]
    U1 = np.array([[float(x) for x in row] for row in dd['u']]); W = np.array([[float(x) for x in row] for row in dd['w']])
    G, w = slotsym_gaps(U1, W, 3, n); h = G.min()
    for (d, hd, mu, m, sig, sk, ok) in stats_from_gaps(G, w):
        print(f"p=3 n={n} winner, delta={d}: min gap {hd:.4e} (h={h:.4e}) mean {mu:.4e} depth {m:.4e} sigma {sig:.4e} depth/sigma {m/sig:.3f} skew {sk:+.3f} lemma ok {ok}")
d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz')
net = EW.Net(d['U0'], d['W0']); p, n = net.p, net.n
f = net.table(); X = np.indices((p,)*n).reshape(n, -1).T; y = X.sum(1) % p; ff = f.reshape(-1, p)
G = np.array([[ff[i, y[i]] - ff[i, (y[i]-dl) % p] for dl in range(1, p)] for i in range(len(X))]); w = np.ones(len(X))/len(X)
h = G.min()
for (dl, hd, mu, m, sig, sk, ok) in stats_from_gaps(G, w):
    print(f"(5,2,2) mixed violator, delta={dl}: min gap {hd:.4e} (h={h:.4e}) depth/sigma {m/sig:.3f} skew {sk:+.3f} lemma ok {ok}")
