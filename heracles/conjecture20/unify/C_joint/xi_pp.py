# p-point version of the single-T field-row constant:
#   Xi*_{n,p}(s) = sup_{d in [0,1], th} (1/p) sum_k (1 + d cos(th + 2 pi k/p))_+^r / F_n(d)^r
import numpy as np
def Phi(n, r, d):
    nu = n+1
    return 0.5*((1+d)**r*(nu+(n-1)*d)**(1-r) + (1-d)**r*(nu-(n-1)*d)**(1-r))
ds = np.linspace(0, 1, 801); ths = np.linspace(0, 2*np.pi, 721)
for (n, s) in [(2, 1.5), (2, 1.2), (3, 2.0), (3, 1.1), (4, 1.5), (4, 2.5), (5, 2.0), (5, 3.0), (8, 4.5), (8, 2.0), (12, 6.5), (20, 3.0)]:
    nu = n+1; r = 1/s; F = nu**(1-s)*Phi(n, r, ds)**s
    line = f"n={n:2d} s={s:4.2f}:"
    for p in [3, 5, 7, 11, 13, 31, 101]:
        k = np.arange(p)
        A = np.maximum(1 + ds[:, None, None]*np.cos(ths[None, :, None] + 2*np.pi*k[None, None, :]/p), 0)**r
        E = A.mean(2).max(1)                     # sup over theta
        X = E/F**r
        line += f" p={p}: {X.max():.5f}"
    print(line, flush=True)
