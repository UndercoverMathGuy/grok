# Check F1 (exact Renyi chain rule with tilted feature marginal), F2 (exact factorisation
# gamma = B_p Pi_feat Pi_within), F3 (deficit split), F4 (general networks: split + W5 + mixed gain).
import numpy as np, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
import cclib as C
E = C.E
rng = np.random.default_rng(7)
worst = dict(chain=0, arimoto_lo=-1, arimoto_hi=-1, fact=0, feat=-1, split=0, mean_id=0)
print('--- random class-C networks')
for (p, n) in [(5, 2), (7, 2), (5, 3), (11, 2), (7, 3), (13, 2)]:
    F = (p - 1) // 2
    for t in range(6):
        widths = {l: int(rng.integers(1, 5)) for l in range(1, F + 1) if rng.random() < 0.8 or l == 1}
        net = C.rand_classC(p, n, widths, rng, scale={l: rng.uniform(0.5, 2) for l in widths})
        ok, lab = C.neuron_freq(net)
        assert ok
        for b in [2.0, 2.5, (n + 1) - 0.3]:
            s = (n + 1) / b; r = 1 / s
            d = C.accounting(net, s, 1.0, lab)
            a = net.mass(); m = a / a.sum()
            # F1: K_r(m)^{s-1} = E_M[K_l^{s-1}] K_r(zhat)^{s-1}  <=>  H_r(m) = H_A + H_r(zhat)
            worst['chain'] = max(worst['chain'], abs(d['Hm'] - d['HA'] - d['Hz']))
            HM = C.renyi(d['M'][d['M'] > 0], r); Hl = np.log(d['K'][d['M'] > 0])
            worst['arimoto_lo'] = max(worst['arimoto_lo'], HM + Hl.min() - d['Hm'])
            worst['arimoto_hi'] = max(worst['arimoto_hi'], d['Hm'] - HM - Hl.max())
            # zhat is the distribution of feature s-costs P_l^s
            Ps = d['Pl'] ** s; worst['mean_id'] = max(worst['mean_id'], np.abs(d['zhat'] - Ps / Ps.sum()).max())
            # F2: gamma = B_p Pi_feat Pi_within ; Pi_feat <= Gamma^{(S)}_s
            worst['fact'] = max(worst['fact'], abs(d['gamma'] - d['B'] * d['Pi_feat'] * d['Pi_within']) / abs(d['gamma']))
            S = [l for l in range(1, F + 1) if d['M'][l - 1] > 0]
            GS, _ = C.programme_vertices(p, s, S)
            worst['feat'] = max(worst['feat'], d['Pi_feat'] - GS)
            # F3: e = M(M o eps) - Gamma_1 d_int ; Pi_within = e / M(M o K^{s-1})
            G1 = p / (p - 1)
            lhs = d['e']; rhs = C.Mcov(d['M'] * d['eps'], p) - G1 * d['dint']
            worst['split'] = max(worst['split'], abs(lhs - rhs) / (abs(lhs) + 1e-30))
            # E_m sigma = sum_l M_l E_{m^(l)} sigma, and 1 - E_{m^(l)} sigma = eps_l
            assert abs(d['Esig'] - (d['M'] * d['Esig_l']).sum()) < 1e-10
            assert np.allclose((1 - d['Esig_l'])[d['M'] > 0], d['eps'][d['M'] > 0], atol=1e-10)
for k, v in worst.items():
    print(f'  {k:12s} {v:.3e}')

print('--- IF / decoupled optimum: Pi_feat = Gamma_s, Pi_within = nu^{1-s} (n=2: = pi)')
for (p, n, b) in [(5, 2, 2.0), (7, 2, 2.5), (11, 2, 2.0), (13, 2, 2.2), (7, 3, 2.0)]:
    s = (n + 1) / b
    G, ys = C.programme_vertices(p, s)
    U, W = E.if_network(p, n, ys[0]); net = E.Net(U, W)
    d = C.accounting(net, s, (n + 1) ** (1 - s))
    print(f'  (p,n,b)=({p},{n},{b}) Pi_feat/Gamma={d["Pi_feat"]/G:.12f}  Pi_within/nu^(1-s)={d["Pi_within"]/(n+1)**(1-s):.12f}'
          f'  CD ratio={d["CD"]:.12f}  dint={d["dint"]:.2e} Esig={d["Esig"]:.2e}')

print('--- general (shared-neuron) networks: split + W5 accounting')
wW5 = 0; wmix = 0
for (p, n) in [(5, 2), (7, 2), (5, 3)]:
    for t in range(5):
        m_ = 6
        U = rng.normal(size=(m_, n, p)); W = rng.normal(size=(m_, p))
        U -= U.mean(2, keepdims=True); W -= W.mean(1, keepdims=True)        # no DC
        net = E.Net(U, W)
        sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global'); import lib as G
        gn = G.split(G.Net(U, W, 2.0)); spl = E.Net(gn.U, gn.W)
        for b in [2.0, 2.6]:
            s = (n + 1) / b; r = 1 / s
            lhs = s * np.log(net.cost(b)); rhs = s * np.log(spl.cost(b))
            Hn = C.renyi(net.mass(), r); Hs = C.renyi(spl.mass()[spl.mass() > 1e-300], r)
            id1 = np.log(net.mass().sum()) + (s - 1) * Hn - lhs
            id2 = np.log(spl.mass().sum()) + (s - 1) * Hs - rhs
            wW5 = max(wW5, abs(id1), abs(id2))
            gap = lhs - rhs   # >= 0, = 0 at b=2 without DC
            if b == 2.0: wmix = max(wmix, abs(gap))
            assert gap >= -1e-12
print(f'  W5 identity err {wW5:.2e};  b=2 no-DC cost gap (should be 0) {wmix:.2e}')
