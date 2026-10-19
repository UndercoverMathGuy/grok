# C_ccfactor c2: (A) field normalisations used by the pair cone moment; (B) KKT charge identity
# E_q*[g(T_i)] = h a_i^r / P_b at KKT points; (C) charge rules (test-measure, Shapley, isolated) versus CD;
# (D) the pair cone moment (JCM) on random class-C networks.
import sys, itertools, glob
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_kkt')
import ccf, cclib as C
import klib as K
E = C.E
rng = np.random.default_rng(11)

print('=== (A) class-C field normalisations: V_l = E|Phi_l - E Phi_l|^2 = sum_T |beta_{l,T}|^2; circularity; orthogonality')
worst = [0, 0, 0]
for (p, n) in [(5, 2), (7, 2), (5, 3), (11, 2), (7, 3)]:
    F = (p - 1) // 2
    for t in range(4):
        widths = {l: int(rng.integers(1, 4)) for l in range(1, F + 1)}
        net = C.rand_classC(p, n, widths, rng)
        fn = ccf.FeatNet(net, 1.5)
        Phi = C.phi_x(net); f = Phi - Phi.mean(1, keepdims=True)
        V_direct = (np.abs(f) ** 2).mean(1)
        worst[0] = max(worst[0], np.abs(V_direct - fn.V).max() / fn.V.max())
        worst[1] = max(worst[1], np.abs((f ** 2).mean(1)).max() / fn.V.max())
        G = (f @ f.conj().T) / f.shape[1]; G2 = (f @ f.T) / f.shape[1]
        off = np.abs(G - np.diag(np.diag(G))).max() + np.abs(G2).max()
        worst[2] = max(worst[2], off / fn.V.max())
print(f'  rel err V_l: {worst[0]:.2e}; |E f_l^2|/V: {worst[1]:.2e}; cross terms: {worst[2]:.2e}')

print('=== (B) KKT charge identity: at a KKT point with unit b-cost, E_q*[g(T_i)] = h a_i^r (each neuron earns h per unit cost)')
# (B1) decoupled IF optimum with q* = uniform(x) x mu*, mu* = programme dual on binding offsets (E_dual D3c)
from scipy.optimize import nnls
for (p, n, b) in [(7, 2, 2.6), (11, 2, 2.0), (13, 2, 2.3), (7, 3, 2.0)]:
    nu = n + 1; s = nu / b; r = 1 / s
    Gm, ys = E.programme_vertices(p, s); y = ys[0]
    U, W = E.if_network(p, n, y)
    net = K.Net(U, W, b)
    act, q, res, h = K.kkt_measure(net)
    Q = K.q_full(net, act, q)
    ng = net.neuron_gaps()                              # (m, N, p-1)
    earn = np.einsum('ixd,xd->i', ng, Q)                # E_q g(T_i)
    a = net.a(); P = net.cost()
    print(f'  IF optimum (p,n,b)=({p},{n},{b}): KKT residual {res:.1e}; max |E_q g(T_i)/(h a_i^r/P) - 1| = '
          f'{np.abs(earn / (h * a ** r / P) - 1).max():.2e}  (sum E_q g = h: {abs(earn.sum()-h)/h:.1e})')
# (B2) local maxima saved by E_kkt (IF and non-IF), any widths
for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p5_n2_b2.0_k*.npz'))[:6] + \
         sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p7_n2_b2.0_k*.npz'))[:4]:
    d = np.load(f); net = K.Net(d['U'], d['W'], float(d['b'])); r = 1 / net.s
    act, q, res, h = K.kkt_measure(net, rtol=1e-5)
    Q = K.q_full(net, act, q); earn = np.einsum('ixd,xd->i', net.neuron_gaps(), Q)
    a = net.a(); P = net.cost(); g = net.gamma()
    ratio = earn / (h * a ** r / P)
    print(f'  {f.split("/")[-1]:24s} gamma={g:.5f} KKT res {res:.1e}  earnings/(h a^r/P) in [{ratio.min():.4f},{ratio.max():.4f}]')

print('=== (C) charge rules vs CD on perturbed IF / random class-C networks (n=2: pi = 3^{1-s})')
summary = dict(n=0, cd_ok=0, tm_viol=0, sh_viol=0, iso_viol=0, cdavg_viol=0, tm_viol_cdok=0, sh_viol_cdok=0)
examples = []
for (p, n, b) in [(7, 2, 2.0), (11, 2, 2.0), (11, 2, 2.5), (13, 2, 2.2), (7, 3, 2.0)]:
    nu = n + 1; s = nu / b; pi = ccf.price(n, s)
    Gm, ys = E.programme_vertices(p, s)
    F = (p - 1) // 2
    for t in range(30):
        # allocation: programme optimum or random support/allocation
        if t % 3 == 0:
            y = ys[0]
        else:
            supp = rng.choice(np.arange(1, F + 1), size=int(rng.integers(2, F + 1)), replace=False)
            y = np.zeros(F); y[supp - 1] = rng.uniform(0.3, 1.0, len(supp))
        eps = [0.03, 0.1, 0.25][t % 3]
        net, lab = ccf.perturbed_if(p, n, y, rng, eps, extra=int(rng.integers(0, 2)))
        fn = ccf.FeatNet(net, s, lab)
        if fn.h <= 0: continue
        summary['n'] += 1
        cd = fn.cd_ratio(pi); summary['cd_ok'] += cd <= 1 + 1e-9
        Ex = fn.excess(pi)
        Bl = pi * fn.B * fn.Pl ** fn.s                   # price of each feature
        bind = fn.binding()
        D = fn.joint_depth()
        cdavg = (D[bind] - (Ex[:, None] * fn.kap[:, bind]).sum(0)).max()   # >=0 at some binding delta suffices
        summary['cdavg_viol'] += cdavg < -1e-12
        Ctm = fn.charge_tm(); Csh = fn.charge_shapley(); Diso = fn.iso_depth()
        feats = [l - 1 for l in fn.feat]
        lim = Bl[feats, None] * fn.kap[feats][:, bind]
        tm = (Ctm[feats][:, bind] / lim).max(); sh = (Csh[feats][:, bind] / lim).max()
        iso = ((fn.Rbar_l[feats][:, bind] - Diso[feats][:, bind]) / lim).max()
        summary['tm_viol'] += tm > 1 + 1e-9; summary['sh_viol'] += sh > 1 + 1e-9; summary['iso_viol'] += iso > 1 + 1e-9
        if cd <= 1 + 1e-9 and tm > 1 + 1e-9:
            summary['tm_viol_cdok'] += 1
            examples.append((p, n, b, eps, cd, tm, sh, fn.gamma() / (pi * fn.B * Gm)))
        if cd <= 1 + 1e-9 and sh > 1 + 1e-9: summary['sh_viol_cdok'] += 1
print('  ', summary)
print('  examples (p,n,b,eps, CD ratio, max test-measure charge/price, max Shapley charge/price, gamma/(pi B Gamma)):')
for e in examples[:8]:
    print('   ', tuple(round(float(v), 4) if isinstance(v, float) else v for v in e))

print('=== (D) pair cone moment: (Rbar(d) - h_d)^2 >= Ibar(d)^2 + sum_l kappa_l(d) cos(2 pi l d/p) V_l, every d')
worst = -1e9; ncase = 0; tight = []
for (p, n) in [(5, 2), (7, 2), (11, 2), (5, 3), (7, 3), (13, 2)]:
    F = (p - 1) // 2; s = 1.5
    for t in range(25):
        y = rng.uniform(0.2, 1, F) * (rng.random(F) < 0.8); y[0] = max(y[0], 0.3)
        net, lab = ccf.perturbed_if(p, n, y, rng, [0.05, 0.2, 0.5][t % 3], extra=int(rng.integers(0, 3)))
        fn = ccf.FeatNet(net, s, lab)
        for d in range(p - 1):
            G = fn.G(); hd = min(G[d], G[fn.neg(d)])
            if hd <= 0: continue
            lhs, rhs, Rb, _ = fn.jcm(d); ncase += 1
            worst = max(worst, (rhs - lhs) / Rb ** 2); tight.append(rhs / lhs if lhs > 0 else 0)
print(f'  {ncase} (network, offset) pairs with h_d > 0: max (rhs-lhs)/Rbar^2 = {worst:.2e}  (<= 0 required); '
      f'max rhs/lhs = {max(tight):.4f}')
# single feature at its binding direction: JCM == W8(i)
for (p, n) in [(5, 2), (7, 3), (11, 2)]:
    y = np.zeros((p - 1) // 2); y[0] = 1.0
    net, lab = ccf.perturbed_if(p, n, y, rng, 0.15, extra=1); fn = ccf.FeatNet(net, 1.5, lab)
    lhs, rhs, Rb, hd = fn.jcm(0)
    A = abs(fn.beta[0]); kap = 1 - np.cos(2 * np.pi / p); x = 1 - hd / (kap * A) if hd > 0 else 1
    kp = 1 / np.cos(2 * np.pi / p) - 1
    print(f'  single feature ({p},{n}): V = {fn.V[0]:.4e} <= k_p x^2 |A|^2 = {kp * x * x * A * A:.4e} (W8(i)); JCM lhs-rhs = {lhs - rhs:.3e}')
