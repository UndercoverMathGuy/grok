# sanity: blocks give ratio 1 (u/s/f, diag and P); junky local maxima from claude0923 and xcharge.
import sys, glob, numpy as np
sys.path.insert(0, '..')
from common import programme, block_net
from lib import evaluate, unpack
for p in (5, 7, 11, 13):
    Gam, y = programme(p, 1.5)
    r = evaluate(p, block_net(p, y))
    print(f"block p={p}: T={r['T']:.6f} " + " ".join(f"{k}={r[k]:.6f}" for k in ('u_diag', 's_diag', 'f_diag', 'u_P')))
for p, N, s in [(5, 9, 3), (7, 12, 1), (7, 12, 9)]:
    try:
        v = np.load(f'../../claude0923/endpoint_p{p}_s{s}.npy')
    except FileNotFoundError:
        continue
    r = evaluate(p, unpack(v, N, p))
    print(f"lmax p={p} s{s}: T={r['T']:.4f} " + " ".join(f"{k}={r[k]:.4f}" for k in ('u_diag', 's_diag', 'f_diag', 'u_P', 's_P', 'f_P')))
