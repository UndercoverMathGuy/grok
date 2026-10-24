# sup over cone data of  (c/|A|) * exp( rho^2 (cos 2phi/cos(2pi/p) - 1) / |A|^2 ),  A = c + rho e^{i phi}, |phi| <= pi/p
# (the n -> infinity, b = 2 envelope of the aligned single-feature bound).  Must be <= 1 for the theorem.
import numpy as np
from scipy.optimize import minimize
for p in [5, 7, 11, 13, 17, 101]:
    cp = np.cos(2 * np.pi / p)
    def negF(z):
        rho = np.exp(z[0]); phi = (np.pi / p) * np.tanh(z[1])
        A2 = 1 + 2 * rho * np.cos(phi) + rho ** 2
        return -(-0.5 * np.log(A2) + rho ** 2 * (np.cos(2 * phi) / cp - 1) / A2)
    best = -np.inf
    for r0 in np.linspace(-6, 4, 41):
        for f0 in np.linspace(-3, 3, 13):
            res = minimize(negF, [r0, f0], method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 4000})
            best = max(best, -res.fun)
    # grid as well
    R = np.exp(np.linspace(-8, 5, 3000))[:, None]; PH = np.linspace(-np.pi / p, np.pi / p, 401)[None, :]
    A2 = 1 + 2 * R * np.cos(PH) + R ** 2
    G = -0.5 * np.log(A2) + R ** 2 * (np.cos(2 * PH) / cp - 1) / A2
    print(f"p={p}: sup log F = {max(best, G.max()):.6f}  (attained at rho->0 if 0); interior local max on grid excluding rho<1e-3: {G[R[:,0]>1e-3].max():.6f}")
