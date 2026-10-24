import sys; import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts'); sys.path.insert(0, '.')
import prog, vlib as V
for (p, s) in [(5, 2.0), (5, 4.0), (7, 3.0), (11, 4.0), (13, 2.0), (13, 3.0), (17, 3.0), (19, 3.0), (11,1.5)]:
    G, y, cov = prog.solve(p, s, restarts=4)
    G2, y2 = V.programme(p, s)
    S = list(np.nonzero(y > 1e-10)[0] + 1)
    Bd = list(np.nonzero(cov <= G * (1 + 1e-6))[0] + 1)
    print(f'p={p} s={s}: prog.solve G={G:.10f} y={np.round(y,6)} supp(>1e-10)={S} bind={Bd} | vertex G={G2:.10f} y={np.round(y2,6)}')
    # recheck value of y from prog
    r = 1/s; yy = np.clip(y,0,None); val = (V.kap(p) @ yy).min() / (np.sum(yy[yy>0]**r))**s
    print('   recomputed prog value', val, ' sum y^r', np.sum(yy[yy>0]**r), ' min y in supp', yy[yy>1e-10].min())
