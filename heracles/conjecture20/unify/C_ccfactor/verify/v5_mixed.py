# Referee check v5 (CC13): canonical CD for mixed networks with (a) proportional allocation of each neuron's full
# b-cost over its frequency pieces, (b) raw split costs.  Recomputed with vlib on SPLIT violators and E_kkt local maxima.
import sys
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V

R = '/home/user/grok/heracles/conjecture20/unify/'


def pieces(U, W, p):
    F = (p - 1) // 2
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    out = np.zeros((U.shape[0], F + 1))
    out[:, 0] = ((np.abs(Uh[:, :, 0]) ** 2).sum(1) + np.abs(Wh[:, 0]) ** 2) / p
    for k in range(1, F + 1):
        out[:, k] = 2 * ((np.abs(Uh[:, :, k]) ** 2).sum(1) + np.abs(Wh[:, k]) ** 2) / p
    return out


def cds(U, W, b):
    m, n, p = U.shape; nu = n + 1; s = nu / b; B = V.Bp(p, n); pi = 3 ** (1 - s)
    h = V.gaps(U, W).min(); sqn = V.sq(U, W); pc = pieces(U, W, p)
    assert np.allclose(pc.sum(1), sqn)
    full = sqn ** (b / 2)
    share = pc[:, 1:] / pc[:, 1:].sum(1, keepdims=True)
    Pprop = (share * full[:, None]).sum(0)
    Pspl = (pc[:, 1:] ** (b / 2)).sum(0)
    G, _ = V.programme(p, s)
    return dict(rd=h / full.sum() ** s / (pi * B * G), cd_prop=h / (pi * B * V.Mcov(Pprop ** s, p)),
                cd_split=h / (pi * B * V.Mcov(Pspl ** s, p)), dc=pc[:, 0].sum() / sqn.sum())


for f in ['viol_p5_n2_b2', 'viol_p7_n2_b2', 'viol_p11_n2_b2']:
    d = np.load(R + 'split/' + f + '.npz')
    for k in range(4):
        if 'U%d' % k not in d or d['U%d' % k].shape[0] == 0: continue
        o = cds(d['U%d' % k], d['W%d' % k], 2.0)
        print(f'{f}[{k}]: RD ratio {o["rd"]:.5f}  CD_prop {o["cd_prop"]:.5f}  CD_split {o["cd_split"]:.5f}  DC share {o["dc"]:.2e}')
for f in ['lm_p5_n2_b2.0_k4', 'lm_p7_n2_b2.0_k6']:
    d = np.load(R + 'E_kkt/' + f + '.npz')
    o = cds(d['U'], d['W'], 2.0)
    print(f'{f}: RD ratio {o["rd"]:.5f}  CD_prop {o["cd_prop"]:.5f}  CD_split {o["cd_split"]:.5f}  DC share {o["dc"]:.2e}')
