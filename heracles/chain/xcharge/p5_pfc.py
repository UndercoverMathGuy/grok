# Per-frequency ceilings on the p = 5 local max.
# PFC   : margin <= (2/p^3) min_D sum_k cov_k(D) Y_k,  Y_k = (cost_k / 3^{1/3})^{3/2},  cost_k = sum_i |c_ik|^{2/3} (diagonal atoms)
# PFC-b : same with cost_k replaced by (p/6) b_k, b_k = frequency-k share of the weight norm (includes imbalance slack)
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
def pfc(p, net):
    z = table(p, net); m = margin(z); h = (p - 1) // 2
    cost = np.zeros(h + 1); b = np.zeros(h + 1)
    for e1, e2, w in net:
        E1, E2, W = np.fft.fft(e1), np.fft.fft(e2), np.fft.fft(w)
        for k in range(1, h + 1):
            cost[k] += (abs(E1[k] * E2[k] * W[k])) ** (2 / 3)
            b[k] += 2 / p * (abs(E1[k]) ** 2 + abs(E2[k]) ** 2 + abs(W[k]) ** 2)
    A = cover_matrix(p)
    Y = (cost[1:] / 3 ** (1 / 3)) ** 1.5; Yb = ((p / 6) * b[1:] / 3 ** (1 / 3)) ** 1.5
    return m, 2 / p ** 3 * (A @ Y).min(), 2 / p ** 3 * (A @ Yb).min(), Y, Yb
if __name__ == '__main__':
    p, N = 5, 9
    v = np.load('../../claude0923/endpoint_p5_s3.npy').reshape(3, N, p)
    net = [(v[0, i], v[1, i], v[2, i]) for i in range(N)]
    m, c1, c2, Y, Yb = pfc(p, net)
    print(f"margin {m:.6f} | PFC bound {c1:.6f} (ratio {m / c1:.4f}) | PFC-b bound {c2:.6f} (ratio {m / c2:.4f})")
