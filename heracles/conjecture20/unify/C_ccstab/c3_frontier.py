# Fixed-Renyi-concentration frontier: is the convex-end efficiency at given Renyi entropy maximised by decoupled networks?
# (5,2), s = 3/2 (b = 2).  Decoupled (IF two-block) networks: gamma_1/B_p = M(yhat), H_{2/3}(m) = log 3 + H_{2/3}(yhat).
import numpy as np, math
from sclib import *

p, n, b = 5, 2, 2.0; s = 1.5; r = 1 / s
B = Bp(p, n); K = ER.kap(p)
def M(yh): return float((K @ yh).min())
ts = np.linspace(0, 1, 200001)
dec = []
for t in ts:
    yh = np.array([1.0, t]) / (1 + t)
    dec.append((math.log(3) + renyi(yh, r), M(yh)))
dec = np.array(dec)
d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz')
G1 = p / (p - 1); Gs, _ = programme_exact(p, s); gIF = 3 ** (1 - s) * B * Gs
print('Gamma_1.5(5) = %.6f; gamma_IF = D_F at n=2' % Gs)
for kk in range(4):
    net = Net(d['U%d' % kk], d['W%d' % kk], 2.0)
    a = masses(net); m = a / a.sum(); h = net.margin(); g1 = h / a.sum() / B
    eta = renyi(m, r); Hsh = renyi(m, 1.0)
    # best decoupled efficiency M at Renyi entropy <= eta
    ok = dec[:, 0] <= eta + 1e-12
    best = dec[ok, 1].max() if ok.any() else float('nan')
    print('viol%d: gamma/gamma_IF=%.6f  H_{2/3}(m)=%.6f  Shannon H_1(m)=%.6f  gamma_1/B_p=%.6f ; best decoupled IF-block M at H_{2/3}<=eta: %s'
          % (kk, net.gamma() / gIF, eta, Hsh, g1, ('%.6f' % best) if ok.any() else 'none (eta < log 3)'))
    # the IF projection of theta: same clean profile
    beta = net.clean(); yh = np.abs(beta) / np.abs(beta).sum()
    print('        own IF projection: H=%.6f, M(yhat)=%.6f ; rho(gamma) = gamma_1/B_p / M(yhat) = %.4f'
          % (math.log(3) + renyi(yh, r), M(yh), g1 / M(yh)))
print('log 3 = %.6f' % math.log(3))
