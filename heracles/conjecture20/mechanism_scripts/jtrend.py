import sys, glob, re, time, numpy as np
sys.path.insert(0, '.')
from joint import inputs, lp_joint
p = 5
best = {}
for f in glob.glob('cs_p5_n*_m*.npy'):
    n, m = map(int, re.findall(r'n(\d+)_m(\d+)', f)[0])
    best.setdefault(n, []).append((m, f))
for n in sorted(best):
    if n < int(sys.argv[1]) or n > int(sys.argv[2]): continue
    t0 = time.time()
    for m, f in sorted(best[n]):
        x = np.load(f); lam = np.sort(x[:m])
        U = inputs(lam, p, 1)
        r1 = lp_joint(U, p, n, freqs=(1,)); r2 = lp_joint(U, p, n, freqs=(1, 2))
        if r1 and r2:
            print("n=%d m=%d  single %.5f  joint %.5f  gain x%.3f  (freq-2 energy %.4f) (%.0fs)" % (n, m, r1[0], r2[0], r2[0] / r1[0], r2[2], time.time() - t0), flush=True)
