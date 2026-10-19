# C_ccfactor c4: every candidate inequality of the factorisation route against BRIEF_CORE section 3.
#   CD_prop   : h <= pi B_p M(Pt^s), Pt = feature costs with each neuron's FULL b-cost split in proportion to its
#               frequency pieces (= SPLIT R10's D_split; for one feature it is the definition of pi)
#   CD_IF     : the same with nu^{1-s} in place of pi (must FAIL at p=3)
#   JCM       : pair cone moment (Rbar-h_d)^2 >= Ibar^2 + sum_l kappa_l cos(2 pi l d/p) V_l   (class C)
#   DU_min/avg: directional single-feature prices
import sys, json, glob, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_kkt')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
import ccf, cclib as C
import klib as K
import lib as GL
E = C.E
CJ = '/home/user/grok/heracles/conjecture20/'


def pieces(U, W, p):
    """frequency pieces' squared norms per neuron: (m, F+1), column 0 = DC."""
    F = (p - 1) // 2
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    out = np.zeros((U.shape[0], F + 1))
    out[:, 0] = (np.abs(Uh[:, :, 0]) ** 2).sum(1) / p + np.abs(Wh[:, 0]) ** 2 / p
    for k in range(1, F + 1):
        out[:, k] = 2 * ((np.abs(Uh[:, :, k]) ** 2).sum(1) + np.abs(Wh[:, k]) ** 2) / p
    return out


def cd_general(U, W, b, pi):
    """returns (CD_prop ratio, CD_split ratio, split-network CD ratio, h, gamma, achievable P_1 range for F=2)."""
    net = E.Net(U, W); p, n = net.p, net.n; nu = n + 1; s = nu / b; F = (p - 1) // 2; B = E.Bp(p, n)
    h = net.margin(); sq = pieces(U, W, p)                                     # (m, F+1)
    full = net.sq() ** (b / 2)
    piece_cost = sq[:, 1:] ** (b / 2)                                           # split costs ||omega^{(k)}||^b
    Pspl = piece_cost.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(sq[:, 1:].sum(1, keepdims=True) > 0, sq[:, 1:] / sq[:, 1:].sum(1, keepdims=True), 0)
    Pprop = (share * full[:, None]).sum(0)                                      # full cost, split in proportion
    kap = E.kappa(p)
    M = lambda P: (kap @ P ** s).min()
    out = dict(h=h, gamma=h / full.sum() ** s, CDprop=h / (pi * B * M(Pprop)), CDsplit=h / (pi * B * M(Pspl)))
    # optimal per-neuron reallocation (F=2 only): neuron i can give any fraction of its full cost to a frequency it carries
    if F == 2:
        lo = full[(sq[:, 2] < 1e-12 * sq.sum(1))].sum(); hi = full.sum() - full[(sq[:, 1] < 1e-12 * sq.sum(1))].sum()
        P1 = np.linspace(lo, hi, 2001); best = max(M(np.array([x, full.sum() - x])) for x in P1)
        out['CDrealloc'] = h / (pi * B * best); out['P1range'] = (lo / full.sum(), hi / full.sum())
    return out


print('=== (1) p = 3 winners (one frequency: CD_prop with pi IS the definition of pi; CD_IF must fail; JCM vacuous)')
sys.path.insert(0, CJ)
import exact_certificate as EC
for nn, d in EC.NETWORKS.items():
    U1 = np.array([[float(a) for a in row] for row in d['u']]); W1 = np.array([[float(a) for a in row] for row in d['w']])
    U = np.repeat(U1[:, None, :], nn, axis=1)
    net = K.Net(U, W1, 2.0); nu = nn + 1; s = nu / 2.0
    g = net.gamma(); B = E.Bp(3, nn); kmin = 1.5
    print(f'  n={nn}: gamma/gamma_IF = {g / (nu ** (1 - s) * B * kmin):.6f} = CD_IF ratio (>1: CD_IF fails);'
          f' pi >= {g / (B * kmin):.4e} = {g / (B * kmin) / nu ** (1 - s):.4f} nu^(1-s);'
          f' acute offsets at p=3: cos(2pi/3) = {np.cos(2 * np.pi / 3):.2f} < 0 -> JCM/acute statements vacuous')
for f in sorted(glob.glob(CJ + 'oned_certified/p3_n1[1-4].json')):
    net, p, nn = K.build1d(f); print(f'  {f.split("/")[-1]}: gamma/gamma_IF = {net.gamma() / K.gamma_if(3, nn, 2.0, 1.5):.6f} (CD_IF fails, single feature)')

print('=== (2) SPLIT R10 mixed violators (b=2, pi = 3^{1-s} proved for all networks at n=2)')
for f in ['viol_p5_n2_b2', 'viol_p7_n2_b2', 'viol_p11_n2_b2']:
    d = np.load(CJ + 'unify/split/' + f + '.npz')
    for kk in range(4):
        U, W = d['U%d' % kk], d['W%d' % kk]
        if U.shape[0] == 0: continue
        p = U.shape[2]; s = 1.5; pi = 3 ** (1 - s)
        o = cd_general(U, W, 2.0, pi)
        G, ys = E.programme_vertices(p, s) if p <= 11 else (None, None)
        Gm = G if G else GL.programme(p, s)[0]
        sp = GL.split(GL.Net(U, W, 2.0)); spn = E.Net(sp.U, sp.W)
        o2 = cd_general(sp.U, sp.W, 2.0, pi) if spn.margin() > -1e300 else None
        print(f'  {f}[{kk}]: gamma/(pi B Gamma) = {o["gamma"] / (pi * E.Bp(p, 2) * Gm):.5f}; CD_prop = {o["CDprop"]:.5f};'
              f' CD_split = {o["CDsplit"]:.5f}; split network: CD = {o2["CDprop"]:.5f}'
              + (f'; CD with optimal neuron reallocation = {o["CDrealloc"]:.5f} (P_1 share range {np.round(o["P1range"], 3)})' if 'CDrealloc' in o else ''))

print('=== (3) E_width X1/X2: IF networks break pointwise-in-s entropy forms but satisfy CD with EQUALITY at any allocation')
for (p, s) in [(11, 1.35), (11, 1.45), (11, 1.49), (13, 1.45), (5, 1.45)]:
    n = 2; b = 3 / s; F = (p - 1) // 2
    y = np.ones(F) if p > 5 or s < 1.4 else np.array([1.0, 0.0])
    U, W = E.if_network(p, n, y / F)
    fn = ccf.FeatNet(E.Net(U, W), s)
    Gm = E.programme_vertices(p, s)[0]; pi = 3 ** (1 - s)
    print(f'  (p,s)=({p},{s}) {"uniform IF" if y.sum() > 1 else "single IF block"}: CD = {fn.cd_ratio(pi):.12f}; RD = {fn.rd_ratio(pi, Gm):.6f}')

print('=== (4) (5,6) width-6 single feature (LARGEN R10b): directional prices with nu^{1-s}')
net1, p, nn = K.build1d(CJ + 'mechanism_scripts/kkt/oned_p5_n6.json')
net = E.Net(net1.U, net1.W); nu = nn + 1
g = net.gaps(); P = net.cost(2.0); s = nu / 2; B = E.Bp(p, nn)
beta = net.clean(); fl = np.nonzero(np.abs(beta) > 1e-12 * np.abs(beta).max())[0] + 1
print(f'  features (clean coefficients) at frequencies {fl}; |beta| = {np.round(np.abs(beta), 8)}')
j = fl[0]; jinv = pow(int(j), -1, p)
m = g.min(0)                                   # m(delta) unnormalised
for l in range(1, (p - 1) // 2 + 1):          # direction l of the feature = offset delta with j*delta = l
    dpos = (l * jinv) % p; dneg = (-l * jinv) % p
    kap = 1 - np.cos(2 * np.pi * l / p); unit = nu ** (1 - s) * B * P ** s * kap
    print(f'  direction {l}: min(m(l),m(-l))/(nu^(1-s) B P^s kappa) = {min(m[dpos - 1], m[dneg - 1]) / unit:.5f};'
          f' average = {(m[dpos - 1] + m[dneg - 1]) / 2 / unit:.5f}')

print('=== (5) Legendre flat field (arith2): fraction of the variance kept by the pair cone moment at delta=1')
for p in [61, 101, 401, 1009]:
    if p % 4 != 1: continue
    F = (p - 1) // 2; w = np.arange(p)
    chi = np.array([0] + [1 if pow(int(x), (p - 1) // 2, p) == 1 else -1 for x in range(1, p)], float)
    ch = np.fft.fft(chi) / p; l = np.arange(1, F + 1)
    ph = ch[l] / (1 - np.exp(-2j * np.pi * l / p)); v = 2 * np.abs(ph)
    kap = 1 - np.cos(2 * np.pi * l / p); V = v ** 2                     # field energies of the flat design
    frac = (kap * np.cos(2 * np.pi * l / p) * V).sum() / (kap * V).sum()
    print(f'  p={p}: sum_l kappa_l cos(phi_l) V_l / sum_l kappa_l V_l = {frac:+.4f}  (<= 0: the JCM bound is vacuous)')

print('=== (6) E_kkt local maxima at (5,2,2) and (7,2,2) (mixed, non-IF ones included): CD_prop')
for f in sorted(glob.glob(CJ + 'unify/E_kkt/lm_p5_n2_b2.0_k*.npz')) + sorted(glob.glob(CJ + 'unify/E_kkt/lm_p7_n2_b2.0_k*.npz')):
    d = np.load(f); p = d['U'].shape[2]; s = 1.5; pi = 3 ** (1 - s)
    o = cd_general(d['U'], d['W'], 2.0, pi)
    Gm = E.programme_vertices(p, s)[0]
    print(f'  {f.split("/")[-1]:24s} gamma/(pi B Gamma) = {o["gamma"] / (pi * E.Bp(p, 2) * Gm):.5f}  CD_prop = {o["CDprop"]:.5f}'
          + (f'  CD_realloc = {o["CDrealloc"]:.5f}' if 'CDrealloc' in o else ''))
