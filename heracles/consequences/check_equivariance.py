"""Rough check of the dilation-equivariance claim for training (CONSEQUENCES.md, claim 9).

Product model z(a,b,c) = sum_i e1_i(a) e2_i(b) w_i(c), cross-entropy over c, weight decay lam, full-batch GD,
float64. Relabel every input and output token by x -> u x mod p (an automorphism of Z_p, so it maps the
task a+b=c to itself). Run A: original init. Run B: the relabelled init (and relabelled training mask).
In exact arithmetic run B is run A relabelled at every step; here we measure the float drift and check that
the learned frequencies map k -> fold(u^{-1} k). Control: run B with the ORIGINAL (unrelabelled) mask.
"""
import numpy as np

p, width, u, lam, lr, steps = 13, 12, 2, 1e-3, 0.3, 6000
h = (p - 1) // 2
X = np.arange(p)
phi = (u * X) % p                   # token relabelling x -> u x
S = (X[:, None] + X[None, :]) % p   # answers


def fold(k):
    k %= p
    return min(k, p - k)


def run(th, mask):
    e1, e2, w = [t.copy() for t in th]
    N = mask.sum()
    hist = []
    for t in range(steps + 1):
        z = np.einsum('ia,ib,ic->abc', e1, e2, w)
        z -= z.max(-1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(-1, keepdims=True)
        P[X[:, None], X[None, :], S] -= 1
        G = P * mask[:, :, None] / N
        if t % 500 == 0:
            hist.append((e1.copy(), e2.copy(), w.copy()))
        g1 = np.einsum('abc,ib,ic->ia', G, e2, w) + 2 * lam * e1
        g2 = np.einsum('abc,ia,ic->ib', G, e1, w) + 2 * lam * e2
        gw = np.einsum('abc,ia,ib->ic', G, e1, e2) + 2 * lam * w
        e1 -= lr * g1
        e2 -= lr * g2
        w -= lr * gw
    return hist


def relabel(th):
    # new table at token phi(x) equals old table at x
    out = []
    for t in th:
        n = np.empty_like(t)
        n[:, phi] = t
        out.append(n)
    return out


def acc(th):
    e1, e2, w = th
    z = np.einsum('ia,ib,ic->abc', e1, e2, w)
    return (z.argmax(-1) == S).mean()


def top_freqs(th, thr=0.25):
    F = np.abs(np.fft.fft(th[0], axis=1))[:, 1:h + 1]
    tot = F.sum(0)
    return sorted(int(k) + 1 for k in np.nonzero(tot > thr * tot.max())[0])


rng = np.random.default_rng(0)
th0 = [rng.normal(0, 0.6, (width, p)) for _ in range(3)]
for frac in (1.0, 0.5):
    mask = (rng.random((p, p)) < frac).astype(float) if frac < 1 else np.ones((p, p))
    maskB = np.empty_like(mask)
    maskB[np.ix_(phi, phi)] = mask     # relabelled training set
    hA = run(th0, mask)
    hB = run(relabel(th0), maskB)
    hC = run(relabel(th0), mask)       # control: relabelled init, original mask
    drift = [max(np.abs(b - r).max() for b, r in zip(B, relabel(Aa))) / max(np.abs(r).max() for r in relabel(Aa))
             for Aa, B in zip(hA, hB)]
    dC = max(np.abs(c - r).max() for c, r in zip(hC[-1], relabel(hA[-1]))) / max(np.abs(r).max() for r in relabel(hA[-1]))
    fA, fB = top_freqs(hA[-1]), top_freqs(hB[-1])
    uinv = pow(u, -1, p)
    print(f'frac={frac}: acc A {acc(hA[-1]):.3f} B {acc(hB[-1]):.3f} C(control) {acc(hC[-1]):.3f}')
    print(f'  relative drift B vs relabelled A every 500 steps: ' + ' '.join(f'{d:.0e}' for d in drift))
    print(f'  control (relabelled init, ORIGINAL mask) final distance: {dC:.3f}')
    print(f'  frequencies A {fA} -> predicted B {sorted(fold(uinv * k) for k in fA)}; observed B {fB}')
