# X-free bound B_XB for any network: min over (i) clean row-uniform bounds at each offset D, (ii) X-blind half-offset
# lines of type {1}, {2}, {1,2} (x_T = s + D/2, D uniform, other coordinate uniform), s = any start point.
# Valid for mixing networks (the lines are X-blind). Reported as B_XB / (gamma_sym P^{3/2}):  <= 1 means (T) holds X-free.
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
def bxb(p, net):
    z = table(p, net); g = gaps(z); i2 = pow(2, -1, p); Ds = range(1, p)
    clean = min(g[:, :, D].mean() for D in Ds)
    lines = []
    for s in range(p):
        lines.append(np.mean([g[(s + D * i2) % p, :, D].mean() for D in Ds]))                  # type {1}
        lines.append(np.mean([g[:, (s + D * i2) % p, D].mean() for D in Ds]))                  # type {2}
        lines.append(np.mean([np.mean([g[a, (s + D * i2 - a) % p, D] for a in range(p)]) for D in Ds]))   # type {1,2}
    Gam, _ = programme(p, 1.5)
    return min(clean, min(lines)) / (gamma_sym(p, Gam) * cost2(net) ** 1.5), margin(z) / (gamma_sym(p, Gam) * cost2(net) ** 1.5)
if __name__ == '__main__':
    for p, N, s in ((5, 9, 3), (7, 12, 9)):
        v = np.load(f'../../claude0923/endpoint_p{p}_s{s}.npy').reshape(3, N, p)
        b, t = bxb(p, [(v[0, i], v[1, i], v[2, i]) for i in range(N)])
        print(f"p={p} junky local max s{s}: true T {t:.4f}   X-free bound B_XB {b:.4f}")
    rng = np.random.default_rng(5); worst = 0
    for t in range(300):   # blocks + mixing perturbations: how close does the X-free bound get to failing?
        p = 5; Gam, ys = programme(p, 1.5)
        net = block_net(p, ys * rng.uniform(0.3, 1.7, 2)) + [tuple(rng.uniform(0.05, 0.6) * v for v in n) for n in random_net(p, int(rng.integers(1, 5)), rng, pure=False)]
        b, tt = bxb(p, net); worst = max(worst, b)
    print("max X-free bound over 300 block+mixing nets at p=5:", round(worst, 4))
