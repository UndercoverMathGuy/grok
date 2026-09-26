import numpy as np
from approg_lib import *
G23 = gamma_star_exact(23)
for L, dl, eps in ((3, 1.2, 0.02), (6, 1.4, 0.02), (9, 1.6, 0.012)):
    T = list(range(1, L + 1))
    w, u = W_exact(23, T, dl, eps, G23, return_u=True)
    fl = eps ** 1.5
    print(f"AP{L} d={dl}: W={w:.5f}; u>floor on T: {[k for k in T if u[k-1] > fl * 1.0001]}; outside support: "
          f"{[(k, round(u[k-1] ** (2/3), 3)) for k in range(1, 12) if k not in T and u[k-1] > 1e-12]}; T shares {[round(u[k-1] ** (2/3), 3) for k in T]}")
w0, u0 = W_exact(23, [], 1.0, 0.0, G23, return_u=True)
print("unfloored optimum support/shares:", [(k, round(u0[k-1] ** (2/3), 3)) for k in range(1, 12) if u0[k-1] > 1e-12])
