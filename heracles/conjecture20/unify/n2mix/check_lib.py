"""Checks for n2lib and the channel-resolved mode budget.
(1) field coefficients = FFT of the fields computed from the table; gaps = sum_j Re(Phi_j c_j).
(2) own data read off fields equals atom formulas (paper Lemma 6 form).
(3) mode budget: total <= 4 e^2;  channel-resolved:  paid_L <= 4 (e^2 - b_L^2)  for every neuron and every L.
(4) free-mode reach at p = 7, L = {1,2}: which own characters +-chi_{l,T} carry free modes.
"""
import itertools
import numpy as np
from n2lib import *

rng = np.random.default_rng(1)
worst = {}
for p in (5, 7, 11):
    G = Grid(p)
    for trial in range(30):
        kind = ['gauss', 'freq', 'freqdc'][trial % 3]
        m = rng.integers(1, 5)
        if kind == 'gauss':
            U, W = random_net(G, m, rng)
        else:
            fr = list(rng.choice(G.F, size=rng.integers(1, len(G.F) + 1), replace=False))
            U, W = random_net(G, m, rng, kind='freq', freqs=fr, dc=(1.0 if kind == 'freqdc' else 0.0))
        # (1)
        phi = fieldcoef(G, U, W)
        Phi = fields_from_table(G, U, W)
        for ji in range(len(G.F)):
            ff = np.fft.fft2(Phi[ji]) / p ** 2      # E_x Phi rho^{-<chi,x>}
            worst['field'] = max(worst.get('field', 0), np.max(np.abs(ff - phi[ji])))
        g = gaps(G, U, W)
        g2 = np.zeros_like(g)
        for di, d in enumerate(G.DELS):
            g2[:, :, di] = sum(np.real(Phi[ji] * G.cfac(j, d)) for ji, j in enumerate(G.F))
        worst['gaps'] = max(worst.get('gaps', 0), np.max(np.abs(g - g2)))
        # (2)
        A, BT = own_data(G, U, W)
        bet = atoms(G, U, W)
        worst['clean'] = max(worst.get('clean', 0), np.max(np.abs(A - bet.sum(axis=0))))
        # (3)
        for L in [list(G.F)] + [list(c) for k in range(1, len(G.F)) for c in itertools.combinations(G.F, k)]:
            own, free, paid, bL, e = mode_masses(G, U, W, L)
            tot = own + free + paid
            worst['total<=4e^2'] = max(worst.get('total<=4e^2', -1), np.max((tot - 4 * e ** 2) / e ** 2))
            worst['paid<=4(e^2-bL^2)'] = max(worst.get('paid<=4(e^2-bL^2)', -1), np.max((paid - 4 * (e ** 2 - bL ** 2)) / e ** 2))
            worst['bL<=e'] = max(worst.get('bL<=e', -1), np.max(bL / e - 1))
print("max deviations / violations (should be ~0 or <=0):")
for k, v in worst.items():
    print(f"  {k:22s} {v:.3e}")

# equality cases of the channel-resolved budget at p=7, L={1,2}
G = Grid(7); a = np.arange(7)
def blk(k, amp, ph):
    return amp * np.cos(2 * np.pi * k * a / 7 + ph)
# balanced shared neuron (R17): pieces at 1 and 2, equal norms
U = np.zeros((1, 2, 7)); W = np.zeros((1, 7))
for t in range(2):
    U[0, t] = blk(1, 1, 0.3 * t) + blk(2, 1, 0.7 * t)
W[0] = blk(1, 1, 0.1) + blk(2, 1, 0.5)
own, free, paid, bL, e = mode_masses(G, U, W, [1, 2])
print(f"R17 shared neuron: bL/e = {bL[0]/e[0]:.6f}, free/e^2 = {free[0]/e[0]**2:.4f}, paid/e^2 = {paid[0]/e[0]**2:.2e}")
# frequency-1 neuron with output frequency-3 stray of energy eps (paid = 4(e^2-bL^2) at the balanced point?)
for eps in (0.05, 1.0):
    U = np.zeros((1, 2, 7)); W = np.zeros((1, 7))
    U[0, 0] = blk(1, 1, 0); U[0, 1] = blk(1, 1, 0.4)
    W[0] = blk(1, 1, 0.2) + blk(3, np.sqrt(eps), 0.9)
    own, free, paid, bL, e = mode_masses(G, U, W, [1, 2])
    print(f"stray output 3, eps={eps}: paid/(4(e^2-bL^2)) = {paid[0]/(4*(e[0]**2-bL[0]**2)):.4f}")
# equality: output only at 3 with inputs at frequency 1 balanced (c = A1 = A2)
U = np.zeros((1, 2, 7)); W = np.zeros((1, 7))
U[0, 0] = blk(1, 1, 0); U[0, 1] = blk(1, 1, 0.4); W[0] = blk(3, 1, 0.2)
own, free, paid, bL, e = mode_masses(G, U, W, [1, 2])
print(f"dedicated paid neuron (balanced): paid/(4(e^2-bL^2)) = {paid[0]/(4*(e[0]**2-bL[0]**2)):.6f}")

# (4) free reach at p=7, L={1,2}
lab = classify(G, [1, 2])
print("\nfree-mode reach at p=7, L={1,2}: (output j, character) labels at +-chi_{l,T}")
for l in (1, 2):
    for T in G.TS:
        chi = ((2 * l * T[0]) % 7, (2 * l * T[1]) % 7)
        mchi = ((-chi[0]) % 7, (-chi[1]) % 7)
        s1 = [lab[(ji, chi[0], chi[1])] for ji in range(3)]
        s2 = [lab[(ji, mchi[0], mchi[1])] for ji in range(3)]
        print(f"  l={l} T={T}: at chi={chi}: {s1};  at -chi={mchi}: {s2}")
