import numpy as np, itertools
psi = lambda t: (np.sqrt(t*t+4) - t)/2
rng = np.random.default_rng(11); worst = 1e9; worst2p = 0
for k in range(50000):
    N = rng.integers(2, 10); x = rng.normal(size=N)*rng.exponential(size=N)**2; w = rng.dirichlet(np.ones(N)*rng.uniform(0.2, 3))
    x -= w @ x; sig = np.sqrt(w @ x**2); t = (w @ x**3)/sig**3
    worst = min(worst, (-x.min())/(sig*psi(t)))
for k in range(2000):                      # two-point laws: equality
    q = rng.uniform(0.01, 0.99); a = rng.exponential(); b = q*a/(1-q); x = np.array([a, -b]); w = np.array([q, 1-q])
    sig = np.sqrt(w @ x**2); t = (w @ x**3)/sig**3; worst2p = max(worst2p, abs(b/(sig*psi(t)) - 1))
print(f"min over 50000 random laws of (-min F)/(sigma psi(t)) = {worst:.6f} (>= 1);  two-point laws |ratio-1| <= {worst2p:.1e}")
# psi(t)+psi(-t) = sqrt(t^2+4) >= 2
ts = np.linspace(-50, 50, 1001); print("min (psi(t)+psi(-t))/2 =", np.min((psi(ts)+psi(-ts))/2))
