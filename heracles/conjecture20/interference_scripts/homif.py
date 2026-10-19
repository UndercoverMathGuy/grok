import sys, time, math, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from cone import *
torch.set_num_threads(1)
p, n, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
alphas = [float(a) for a in sys.argv[4].split(',')]
noise = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
rng = np.random.default_rng(seed); nu = n + 1; t0 = time.time()
U, Wr, Wi = if_gadget(p, n)
U = U + noise * rng.standard_normal(U.shape); Wr = Wr + noise * rng.standard_normal(nu); Wi = Wi + noise * rng.standard_normal(nu)
for al in alphas:
    c = Cone(p, n, math.radians(al))
    r, (U, Wr, Wi) = optimise(c, U, Wr, Wi, stages=(0.01, 0.003, 0.001, 3e-4, 1e-4), iters=500)
    amp = np.abs(Wr + 1j * Wi); amp = amp / amp.max()
    print("p=%d n=%d seed=%d alpha=%.1f eff=%.5f  #amp>1e-3: %d (%.0fs)" % (p, n, seed, al, r, (amp > 1e-3).sum(), time.time() - t0), flush=True)
    np.save('homif_p%d_n%d_s%d_a%d.npy' % (p, n, seed, round(al)), np.concatenate([U, Wr[:, None], Wi[:, None]], 1))
