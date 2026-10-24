# Investigate VIOL_b_p11_n2_b2.5_pen_s51_9: network with feature 1 and a tiny foreign clean coefficient at frequency 4.
import numpy as np, hlib as H, glib as Gb, mpmath as mp
GL = H.GL
z = np.load('out/VIOL_b_p11_n2_b2.5_pen_s51_9.npz')
U, W = z['U'], z['W']; p, n, b = int(z['p']), int(z['n']), float(z['b']); s = 3 / b; F = (p - 1) // 2
D = 3 ** (1 - s) * H.Bp(p, n) * (1 - np.cos(2 * np.pi / p))
net = GL.Net(U, W, b)
print('float ratio', net.gamma() / D)
hm, ct = H.margin_mp(U, W, dps=30)
cm = mp.fsum([c ** (mp.mpf(b) / 2) for c in ct])
print('mp ratio', mp.nstr(hm / cm ** mp.mpf(s) / mp.mpf(D), 15))
# frequency content per neuron
for i in range(net.m):
    def fe(v):
        e = [float(np.sum((v @ GL.freq_proj(p, k)) ** 2)) for k in range(F + 1)]
        return np.round(np.array(e) / max(sum(e), 1e-300), 4)
    print('neuron', i, 'norm^2 %.4g' % ((U[i] ** 2).sum() + (W[i] ** 2).sum()), 'u1', fe(U[i, 0]), 'u2', fe(U[i, 1]), 'w', fe(W[i]))
at = net.atoms()
print('atoms beta_ij (rows neurons):'); print(np.round(np.abs(at), 10))
