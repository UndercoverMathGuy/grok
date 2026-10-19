import numpy as np, hlib as H, itertools, time
z = np.load('out/VIOL_a_p29_b2.5_Sstar_s12_10.npz', allow_pickle=True)
p = int(z['p']); n = int(z['n']); b = float(z['b']); s = 3 / b; S = [int(v) for v in z['S']]
print('S', S, 'stored ratio', float(z['ratio']))
G0, y0, B0 = H.programme(p, s, S=S, restarts=60)
print('reweighted-LP Gamma^S', G0, 'support', [l for l in S if y0[l-1] > 0])
t = time.time(); Ge, ye = H.programme_exact(p, s, S); print('exact vertex Gamma^S', Ge, 'support', [l for l in S if ye[l-1] > 1e-12], f'{time.time()-t:.0f}s')
G1, y1, B1 = H.programme(p, s, restarts=200, seed=5)
print('full-F reweighted LP (200 restarts)', G1, [l for l in range(1, 15) if y1[l-1] > 0])
cc = H.CC(p, n, list(z['freqs']), s); x = z['x']
fd = cc.feature_data(x)
print('gains', {l: round(d['gain'], 6) for l, d in fd.items()})
print('ratio vs exact Gamma^S:', cc.gnorm(x) / (3 ** (1 - s) * Ge))
