# Sanity check of the exact pointwise identity (any network, any n, p):
#   g(x,delta) = sum_l Re(Phi_l(x)(1 - rho^{-l delta}))  with  Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-l c},
#   min(g(x,d), g(x,-d)) = <Re Phi(x), kappa(d)> - |<Im Phi(x), sin(2 pi l d/p)>|,
# hence  h = min_x min_d [ <Re Phi(x),kappa(d)> - |<Im Phi(x),sigma(d)>| ]  <=  min_x m(Re Phi(x)).
# (The constant (DC) part of f(x)[.] drops out of every gap.)
import numpy as np, itertools
rng = np.random.default_rng(7)
for (p, n, width) in [(5, 2, 4), (7, 2, 6), (11, 2, 5), (7, 3, 4), (13, 2, 7)]:
    rho = np.exp(2j*np.pi/p); F = (p-1)//2; ls = np.arange(1, F+1)
    U = rng.normal(size=(width, n, p)); W = rng.normal(size=(width, p))
    worst = 0.0; h_direct = np.inf; h_formula = np.inf
    for x in itertools.product(range(p), repeat=n):
        fx = np.zeros(p)
        for i in range(width):
            fx += W[i]*np.prod([U[i, t, x[t]] for t in range(n)])
        y = sum(x) % p
        Phi = np.array([(2/p)*rho**(l*y)*np.sum(fx*rho**(-l*np.arange(p))) for l in ls])
        for d in range(1, p):
            g_direct = fx[y] - fx[(y-d) % p]
            g_formula = np.sum(np.real(Phi*(1 - rho**(-ls*d))))
            worst = max(worst, abs(g_direct - g_formula))
            h_direct = min(h_direct, g_direct)
        for d in range(1, F+1):
            kap = 1 - np.cos(2*np.pi*ls*d/p); sig = np.sin(2*np.pi*ls*d/p)
            val = np.dot(Phi.real, kap) - abs(np.dot(Phi.imag, sig))
            h_formula = min(h_formula, val)
    print(f"p={p} n={n} width={width}: max|g - formula| = {worst:.2e},  h(direct) = {h_direct:.6f},  h(identity) = {h_formula:.6f}")
