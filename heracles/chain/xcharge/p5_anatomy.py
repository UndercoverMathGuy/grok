# Anatomy of the p = 5 junky local max: budget slack, per-frequency populations, eta-form slack, chain pure bound vs margin.
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
p, N = 5, 9
v = np.load('../../claude0923/endpoint_p5_s3.npy').reshape(3, N, p)
net = [(v[0, i], v[1, i], v[2, i]) for i in range(N)]
z = table(p, net); m = margin(z); P = cost2(net); Gam, ystar = programme(p, 1.5); gs = gamma_sym(p, Gam)
print(f"margin {m:.6f}  P {P:.6f}  gamma/gamma_sym {m / P ** 1.5 / gs:.6f}")
F = lambda u: np.fft.fft(u)
tot_diag = 0; slackB = 0
pops = {1: [], 2: []}
for i, (e1, e2, w) in enumerate(net):
    E1, E2, W = F(e1), F(e2), F(w)
    th2 = e1 @ e1 + e2 @ e2 + w @ w
    dc = (abs(E1[0]) ** 2 + abs(E2[0]) ** 2 + abs(W[0]) ** 2) / p
    amps = {k: (abs(E1[k]), abs(E2[k]), abs(W[k])) for k in (1, 2)}
    diag = sum((amps[k][0] * amps[k][1] * amps[k][2]) ** (2 / 3) for k in (1, 2))
    tot_diag += diag; slackB += th2 - (6 / p) * diag
    for k in (1, 2):
        c = E1[k] * E2[k] * np.conj(W[k])
        if abs(c) > 1e-12:
            pops[k].append((c, np.conj(E1[k]) / E1[k], np.conj(E2[k]) / E2[k]))
    print(f"neuron {i}: |theta|^2 {th2:.4f}  DC {dc:.4f}  |e1^,e2^,w^|(k=1) {np.round(amps[1], 3)}  (k=2) {np.round(amps[2], 3)}")
print(f"budget: P = {P:.5f}, (6/p) sum|c|^(2/3) = {(6 / p) * tot_diag:.5f}, slack {slackB:.5f} ({slackB / P:.3f} of P)")
G, J = spectrum(z)
for k in (1, 2):
    c = np.array([t[0] for t in pops[k]]); eta = np.sqrt(sum(abs(J[T][k]) ** 2 for T in J))
    cost_k = (np.abs(c) ** (2 / 3)).sum(); yk = max(0, abs(G[k]) - 0.272 * eta)
    print(f"k={k}: atoms {len(c)}, |G| {abs(G[k]):.4f}, eta {eta:.4f}, |J1,J2,J12| {[round(abs(J[T][k]), 4) for T in ('1', '2', '12')]}, "
          f"cost_k {cost_k:.4f}, eta-form RHS {3 ** (1 / 3) * yk ** (2 / 3):.4f}, clean-JF RHS {3 ** (1 / 3) * abs(G[k]) ** (2 / 3):.4f}")
A = cover_matrix(p)
for Aim in (0.272, 0.316):
    y = np.array([max(0, abs(G[k]) - Aim * np.sqrt(sum(abs(J[T][k]) ** 2 for T in J))) for k in (1, 2)])
    print(f"chain pure bound at aim {Aim}: (2/p^3) min_D sum cov y = {2 / p ** 3 * (A @ y).min():.6f}  vs true margin {m:.6f}")
