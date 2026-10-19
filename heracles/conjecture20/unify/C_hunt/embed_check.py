# Check of Prop. embed: an n = 2 fluctuation design embedded at n = 3, 4 via T1 = {1}, T2 = {2..n}, T3 = [n] has the same
# pair-average detection / L2 ratio.
import itertools, numpy as np


def ratio(p, n, design):          # design: {(l, T): a}
    X = np.indices((p,) * n).reshape(n, -1).T
    D = {}
    for d in (1, p - 1):
        v = np.zeros(X.shape[0])
        for (l, T), a in design.items():
            v += np.real(a * (1 - np.exp(-2j * np.pi * l * d / p)) * np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p))
        D[d] = v
    det = 0.5 * ((-D[1]).max() + (-D[p - 1]).max())
    L2 = np.sqrt(sum(abs(a) ** 2 * (1 - np.cos(2 * np.pi * l / p)) for (l, T), a in design.items()))
    return det / L2


for p in [5, 7, 11]:
    z = np.load(f'out2/detl2_p{p}_n2_F_avg.npz') if p == 7 else np.load(f'out2/detl2_p{p}_n2_F.npz')
    a = z['a']; cols = z['cols']
    d2 = {(int(c[0]), (int(c[1]), int(c[2]))): a[k] for k, c in enumerate(cols)}
    out = [f"p={p} n=2 ratio={ratio(p, 2, d2):.12f}"]
    for n in [3, 4]:
        if p ** n > 20000: continue
        emb = {}
        for (l, T), av in d2.items():
            Tn = {(1, 0): (1,) + (0,) * (n - 1), (0, 1): (0,) + (1,) * (n - 1), (1, 1): (1,) * n}[T]
            emb[(l, Tn)] = av
        out.append(f"n={n} embedded ratio={ratio(p, n, emb):.12f}")
    print('  '.join(out), f" sqrt((p-1)/(p+1))={np.sqrt((p-1)/(p+1)):.12f}")
