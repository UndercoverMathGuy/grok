# Integrated temperature identity on the certified p=3 networks: for a FIXED network with equal neuron costs,
#   log(gamma/gamma_IF)(s) = log(gamma/gamma_IF)(s_cross) + (log nu - H) (s - s_cross),  H = log(width)  (exact, linear).
import math, numpy as np
from entropy_check import p3net, counts
from lib import Bp
base = "/home/user/grok/heracles/conjecture20/oned_certified/"
for n in [11, 14, 18]:
    p, n, U, W = p3net(base + f"p3_n{n}.json"); nu = n + 1
    h = min(min(f[y] - f[c] for c in range(p) if c != y) for N in counts(n, p)
            for (y, f) in [(int((np.arange(p) * np.array(N)).sum() % p), np.prod(U ** np.array(N)[None, :], axis=1) @ W)])
    S = nu * (U ** 2).sum(1); km = 1 - math.cos(2 * math.pi / 3)
    lr = lambda s: math.log(h / np.sum(S ** (nu / s / 2)) ** s / (nu ** (1 - s) * Bp(3, n) * km))
    w = S / S.sum(); H = -(w * np.log(w)).sum()
    sc = [s for s in np.linspace(1, nu / 2, 20001) if lr(s) >= 0][0]
    print(f"n={n}: log ratio at b=2 {lr(nu/2):.5f};  (log nu - H)(nu/2 - s_cross) = {(math.log(nu)-H)*(nu/2-sc):.5f};  s_cross={sc:.4f} (b={nu/sc:.4f})")
