# Track the sub-nu-width branch in n: seed by random restarts at n0 with m0 neurons, then continue n -> n+1,
# trying (same m) and (m+1 by splitting a neuron); keep the best few per m.  Fine-temperature continuation.
import sys, time, numpy as np, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from famp import Net, optimise
torch.set_num_threads(1)
p, n0, m0, n1, R, seed = [int(a) for a in sys.argv[1:7]]
rng = np.random.default_rng(seed); t0 = time.time()
net = Net(p, n0)
seeds = []
for k in range(R):
    r, U, W, la = optimise(net, rng.standard_normal((m0, p)), rng.standard_normal((m0, p)), np.zeros(m0),
                           stages=(0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 3e-4, 1e-4), iters=300)
    seeds.append((r, U, W, la))
seeds.sort(key=lambda t: -t[0]); pool = seeds[:3]
print("p=%d n=%d seeds: %s (%.0fs)" % (p, n0, [round(s[0], 4) for s in pool], time.time() - t0), flush=True)
fine = (3e-3, 1e-3, 3e-4, 1e-4)
for n in range(n0 + 1, n1 + 1):
    net = Net(p, n); nu = n + 1; cands = []
    for r, U, W, la in pool:
        m = len(la)
        cands.append((U, W, la))
        if m + 1 < nu:
            for _ in range(2):
                j = rng.integers(m)
                cands.append((np.vstack([U, U[j] + 0.05 * rng.standard_normal(p)]), np.vstack([W, W[j] + 0.05 * rng.standard_normal(p)]),
                              np.append(la, la[j] - 0.7)))
    res = []
    for U, W, la in cands:
        r, U2, W2, la2 = optimise(net, U, W, la, stages=fine, iters=500)
        res.append((r, U2, W2, la2))
    res.sort(key=lambda t: -t[0])
    # keep best per m (up to 3 entries)
    pool, seen = [], set()
    for t in res:
        if len(t[3]) not in seen and t[0] > 0:
            pool.append(t); seen.add(len(t[3]))
        if len(pool) == 3: break
    if not pool: pool = res[:1]
    print("p=%d n=%d best %.5f  by m %s (%.0fs)" % (p, n, res[0][0], [(len(t[3]), round(t[0], 4)) for t in pool], time.time() - t0), flush=True)
    np.save("branch_p%d_n%d_s%d.npy" % (p, n, seed), np.concatenate([pool[0][1], pool[0][2], pool[0][3][:, None]], 1))
