import sys; sys.argv = ['dict_lp.py', '7', '2', '2.0', 'cS', '3', '0', '1']
exec(open('dict_lp.py').read().split('t0 = time.time(); recs = []')[0])
import numpy as np
rng = np.random.default_rng(3)
for trial in range(3):
    cc0 = H.CC(p, n, [1, 1, 2, 3, 3], s)
    x0 = rng.standard_normal(2 * cc0.m + cc0.m * n)
    U, W = cc0.to_net(x0, bexp)
    cc1, x1 = cc_from(U, W)
    nt = GL.Net(U, W, bexp)
    print('model gnorm orig', cc0.gnorm(x0), 'converted', cc1.gnorm(x1), 'table gamma/Bp', nt.margin() / nt.cost() ** s / H.Bp(p, n))
