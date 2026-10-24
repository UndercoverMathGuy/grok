# Sufficient condition for concavity of F = nu^{1-s} Phi^s (single-T aligned payment):
#   N d - 2r <= (N+2r) tanh((1-r) artanh d + r artanh(eps d)) <= N d + 2r,  N=n-1, eps=N/(N+2).
import numpy as np
d = np.linspace(0, 1-1e-9, 200001)
worst_up = (1e9, None); worst_lo = (1e9, None)
for n in range(1, 401):
    N = n-1; eps = N/(N+2)
    for r in np.linspace(1e-4, 1, 101):
        T = np.tanh((1-r)*np.arctanh(d) + r*np.arctanh(eps*d))
        up = (N*d + 2*r) - (N+2*r)*T; lo = (N+2*r)*T - (N*d - 2*r)
        if up.min() < worst_up[0]: worst_up = (up.min(), (n, r, d[up.argmin()]))
        if lo.min() < worst_lo[0]: worst_lo = (lo.min(), (n, r, d[lo.argmin()]))
print("upper slack min:", worst_up); print("lower slack min:", worst_lo)
