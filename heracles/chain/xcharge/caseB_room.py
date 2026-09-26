# Room left by the PROVED single-frequency chain (flip aim, depth 0.29, eta-form) on case-B networks:
#   pure chain bound PB = (2/p^3) min_D sum_k cov_k(D) (|G_k| - 0.29 eta_k)_+   (valid for the diagonal part only)
#   room for X = gamma_sym P^{3/2} - PB   (the X leak under the chain's aim must fit in this)
import sys, glob, numpy as np
sys.path.insert(0, '..')
from common import *
from xblind_bound import bxb
p = 5; Gam, _ = programme(p, 1.5); A = cover_matrix(p)
def room(net):
    z = table(p, net); G, J = spectrum(z); P = cost2(net); tgt = gamma_sym(p, Gam) * P ** 1.5
    eta = np.array([np.sqrt(sum(abs(J[T][k]) ** 2 for T in J)) for k in (1, 2)])
    y = np.clip(np.abs(G[1:3]) - 0.29 * eta, 0, None)
    PB = 2 / p ** 3 * (A @ y).min()
    return margin(z) / tgt, PB / tgt, bxb(p, net)[0], eta / np.maximum(np.abs(G[1:3]), 1e-12)
files = sorted(glob.glob('caseB3_d*_s*.npy'))
for f in files:
    v = np.load(f).reshape(3, 9, p); net = [(v[0, i], v[1, i], v[2, i]) for i in range(9)]
    T, PB, F, rel = room(net)
    print(f"{f}: true T {T:+.3f} | F (X-free) {F:.3f} | pure chain bound PB/target {PB:.3f} -> room for X {1 - PB:.3f} | eta/|G| {np.round(rel, 2)}")
v = np.load('../../claude0923/endpoint_p5_s3.npy').reshape(3, 9, p)
T, PB, F, rel = room([(v[0, i], v[1, i], v[2, i]) for i in range(9)])
print(f"p=5 junky local max: true T {T:.3f} | F {F:.3f} | PB/target {PB:.3f} | eta/|G| {np.round(rel, 2)}")
