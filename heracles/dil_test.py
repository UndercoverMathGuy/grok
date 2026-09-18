"""Exact test of the per-frequency dilution inequality for ReLU neurons on Z_p x Z_p (p = 113).
For a neuron x(a,b) = u(a) + v(b): clean coefficients h(k,k) (2D DFT of relu(x)), input amplitudes rho_k(u), rho_k(v),
sigma = rms of x. Questions:
  Q1 leakage: clean amplitude at frequencies where rho_k(u) rho_k(v) = 0 (harmonics + intermodulation), as a share of total clean.
  Q2 direct constant: |h(k,k)| / (rho_k(u) rho_k(v) / sigma) at supported k, vs the single-frequency value.
  Q3 the quantity the programme needs per frequency: |h(k,k)| <= C * sqrt(m_k(u) m_k(v)) / sigma  (m = energy at k)."""
import numpy as np, itertools, math
p = 113; nf = (p-1)//2; th = 2*np.pi*np.arange(p)/p
rng = np.random.default_rng(1)
def neuron(fu, au, pu, fv, av, pv):
    u = sum(a*np.cos(k*th+f) for k,a,f in zip(fu,au,pu)); v = sum(a*np.cos(k*th+f) for k,a,f in zip(fv,av,pv))
    return u, v
def clean(u, v):
    h = np.maximum(u[:,None]+v[None,:], 0.0); H = np.fft.fft2(h)/p**2
    return np.array([abs(H[k,k]) for k in range(1, nf+1)])
def amps(u):
    U = np.fft.fft(u)/p; return 2*np.abs(U[1:nf+1])      # rho_k: u = sum rho_k cos(k a + phi)
# single-frequency reference constant
u,v = neuron([1],[1],[0],[1],[1],[0]); c = clean(u,v); sig = math.sqrt(np.var(u[:,None]+v[None,:]))
C1 = c[0]/(1*1/sig); print(f"single freq: |h(1,1)| {c[0]:.5f}  sigma {sig:.4f}  C1 = |h|/(rho rho/sigma) = {C1:.5f}; harmonics share {(c.sum()-c[0])/c.sum():.3f} (h2/h1 {c[1]/c[0]:.3f})")
rows = []
for trial in range(400):
    K = rng.integers(1, 9)
    fu = sorted(rng.choice(np.arange(1, 25), size=K, replace=False)); fv = fu if rng.uniform() < 0.7 else sorted(rng.choice(np.arange(1,25), size=K, replace=False))
    au = rng.uniform(0.2, 1, K); av = rng.uniform(0.2, 1, K); pu = rng.uniform(0, 2*np.pi, K); pv = rng.uniform(0, 2*np.pi, K)
    u, v = neuron(fu, au, pu, fv, av, pv); c = clean(u, v); ru, rv = amps(u), amps(v)
    sig = math.sqrt(np.var(u[:,None]+v[None,:]))
    supp = (ru*rv) > 1e-9
    if not supp.any(): continue
    leak = c[~supp].sum()/c.sum(); leak_max = c[~supp].max()/c[supp].max() if supp.any() else np.inf
    ratio = c[supp]/(ru[supp]*rv[supp]/sig)
    rows.append((K, len(set(fu)&set(fv)), leak, leak_max, ratio.max()/C1, ratio.min()/C1))
rows = np.array(rows)
print("\nrandom multi-frequency neurons (K freqs in u, K in v, 70% same support), 400 trials:")
for K in range(1, 9):
    r = rows[rows[:,0]==K]
    if len(r): print(f" K={K}: n={len(r):3d}  leak share of clean: med {np.median(r[:,2]):.3f} max {r[:,2].max():.3f} | max single leaked freq / max supported: med {np.median(r[:,3]):.3f} max {r[:,3].max():.3f} | direct const / C1: max {r[:,4].max():.3f} min {r[:,5].min():.3f}")
# adversarial: u,v on {1,2} feeding k=3; and comb-locked
for fu in ([1,2],[1,2,3],[1,3],[2,3]):
    u,v = neuron(fu,[1]*len(fu),[0]*len(fu),fu,[1]*len(fu),[0]*len(fu)); c = clean(u,v); ru,rv=amps(u),amps(v); sig=math.sqrt(np.var(u[:,None]+v[None,:]))
    supp=(ru*rv)>1e-9
    print(f" locked comb {fu}: clean at supported {np.round(c[supp],4)}  leaked top3 {np.round(np.sort(c[~supp])[-3:][::-1],4)}  leak share {c[~supp].sum()/c.sum():.3f}  direct/C1 {np.round(c[supp]/(ru[supp]*rv[supp]/sig)/C1,3)}")

# --- check of the exact separable representation (THEORY_relu Lemma L5):
#   h(k,k) = -(1/pi) int_0^inf [ c_u(t,k) c_v(t,k) - s_u(t,k) s_v(t,k) ] dt / t^2,
#   c_u(t,k) = k-th Fourier coefficient of cos(t u(.)), s_u of sin(t u(.));  from |x| = (2/pi) int (1 - cos tx)/t^2 dt.
u, v = neuron([1, 2], [1, 0.7], [0.3, 1.1], [1, 3], [0.8, 0.6], [2.0, 0.4])
H = np.fft.fft2(np.maximum(u[:, None] + v[None, :], 0.0)) / p**2
t = np.linspace(1e-4, 400, 400000); dt = t[1] - t[0]
def coef(f, k): return (np.fft.fft(f, axis=1) / p)[:, k]          # f (T, p) -> k-th coefficient per t
cu, su = coef(np.cos(np.outer(t, u)), slice(1, 6)), coef(np.sin(np.outer(t, u)), slice(1, 6))
cv, sv = coef(np.cos(np.outer(t, v)), slice(1, 6)), coef(np.sin(np.outer(t, v)), slice(1, 6))
rep = -(1 / np.pi) * np.sum((cu * cv - su * sv) / t[:, None]**2, axis=0) * dt
print("\nL5 check, k = 1..5:  exact h(k,k)", np.round([H[k, k] for k in range(1, 6)], 5), "\n                     integral   ", np.round(rep, 5))
