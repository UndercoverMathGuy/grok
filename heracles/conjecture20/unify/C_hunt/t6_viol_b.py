import numpy as np, hlib as H
GL = H.GL
z = np.load('out/VIOL_b_p11_n2_b2.5_pen_s51_9.npz')
U, W = z['U'], z['W']; p, n, b = int(z['p']), int(z['n']), float(z['b']); s = 3 / b
D = 3 ** (1 - s) * H.Bp(p, n) * (1 - np.cos(2 * np.pi / p))
full = GL.Net(U, W, b)
blk = GL.Net(U[:3], W[:3], b)
f4 = GL.Net(U[3:4], W[3:4], b)
print('ratio full', full.gamma() / D, ' block only', blk.gamma() / D)
print('h full', full.margin(), ' h block', blk.margin(), ' cost full', full.cost(), ' cost block', blk.cost())
gb = blk.gaps(); g4 = f4.gaps(); gf = full.gaps()
print('block gaps: min per delta', np.round(gb.min(0), 8))
print('block gaps: max per delta', np.round(gb.max(0), 8))
print('feat4 gaps: min per delta', np.round(g4.min(0), 10))
print('full gaps: min per delta', np.round(gf.min(0), 8))
# is the block itself exactly IF? fluctuations of feature 1
print('block clean', np.abs(blk.clean())[:2], 'block gain', (lambda A, Q, a: A / a)(*blk.cheapness())[0])
