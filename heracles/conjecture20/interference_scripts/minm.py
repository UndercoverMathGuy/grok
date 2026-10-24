# best normalised margin (ratio to IF) with exactly m neurons, slot-symmetric, random restarts
import sys, time, numpy as np, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from famp import Net, optimise
torch.set_num_threads(1)
p, n, m, R = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(100 * p + 10 * n + m); net = Net(p, n); best = -1e9; t0 = time.time()
for k in range(R):
    r, U, W, la = optimise(net, rng.standard_normal((m, p)), rng.standard_normal((m, p)), np.zeros(m),
                           stages=(0.3, 0.1, 0.03, 0.01, 0.003, 1e-3), iters=300)
    best = max(best, r)
print("p=%d n=%d m=%d (nu=%d): best ratio over %d restarts = %.5f  (%.0fs)" % (p, n, m, n + 1, R, best, time.time() - t0), flush=True)
