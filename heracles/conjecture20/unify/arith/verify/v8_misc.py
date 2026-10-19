# Referee: (R3) water-filling formula vs brute-force box maximisation + concavity/homogeneity of E0;
#          (R7) KKT multipliers on the binding set at programme maximisers (own LP);
#          (R8/R9b) compression lemma on EXACT networks with a cheap progression block.
import numpy as np, sys, json
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/arith2'); sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/arith/verify')
import lib
from scipy.optimize import minimize, linprog
from fractions import Fraction
rng = np.random.default_rng(7)
# ---- R3
worst = 0; wc = 0; wh = 0
for trial in range(300):
    n = rng.integers(3, 25); c = rng.uniform(0.05, 0.4)
    a = rng.exponential(size=n)*(rng.random(n) < 0.8); kap = rng.uniform(0, 2, n)
    E = lib.waterfill_excess(a, kap, c)
    f = lambda u: -(c*(u*kap).sum()-np.sqrt((u*u*kap).sum()+1e-300))
    best = 0
    for st in range(6):
        u0 = a*rng.uniform(0, 1, n)
        res = minimize(f, u0, bounds=[(0, ai) for ai in a], method='L-BFGS-B')
        best = max(best, -res.fun)
    worst = max(worst, best-E)
    a2 = rng.exponential(size=n); E2 = lib.waterfill_excess(a2, kap, c); Em = lib.waterfill_excess((a+a2)/2, kap, c)
    wc = max(wc, (E+E2)/2-Em); wh = max(wh, abs(lib.waterfill_excess(3*a, kap, c)-3*E))
print(f"R3: max[brute - waterfill] = {worst:.2e} (<= ~0),  concavity violation {wc:.2e},  homogeneity err {wh:.2e}")
# ---- R7
for p in [31, 61, 101, 211]:
    G, y = lib.load_support(p); s = 1.5; r = 2/3
    K = 1-np.cos(2*np.pi*np.outer(np.arange(1, p), np.arange(1, (p-1)//2+1))/p)   # all delta = 1..p-1
    cov = K@y; Gm = cov.min(); B = np.nonzero(cov < Gm*(1+1e-7))[0]; S = np.nonzero(y)[0]
    yn = y/np.sum(y[S]**r)**s
    target = Gm/np.sum(y[S]**r)**s*yn[S]**(r-1)
    # find mu >= 0 on B, sum mu = 1, K[B,S]^T mu = target  (LP feasibility, minimise max residual)
    nb = len(B); ns = len(S)
    Aeq = np.vstack([K[np.ix_(B, S)].T, np.ones((1, nb))]); beq = np.concatenate([target, [1]])
    res = linprog(np.zeros(nb), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    resid = np.abs(Aeq@res.x-beq).max() if res.status == 0 else np.nan
    print(f"R7 p={p}: |S|={ns}, |B| (delta in Z_p^*)={nb}, LP status={res.status}, residual={resid:.1e}, min mu={res.x.min() if res.status==0 else np.nan:.3f}")
# ---- R8 / R9(b) on exact networks
import v2_network as V
D = '/home/user/grok/heracles/conjecture20/unify/arith2/'
certs = {}
for fn in ['compress_cert_uniform_s1.5.json', 'compress_cert_warm_s1.5.json']:
    for q, e in json.load(open(D+fn)).items(): certs[int(q)] = (e['freqs'], [float(Fraction(x)) for x in e['bprime']])
s = 1.5; c = 1-np.sqrt(2/3); tp = np.sqrt(1.5); r = 2/3
for p in [43, 61]:
    G, y = lib.prog(p, s, restarts=3); F = (p-1)//2; L = np.arange(1, F+1)
    Kd = 1-np.cos(2*np.pi*np.outer(np.arange(1, p), L)/p)
    for trial in range(4):
        q = int(rng.integers(17, F+1)); q = q if q in certs else 17; d = int(rng.integers(1, p))
        Bset = sorted({min(d*j % p, p-d*j % p) for j in range(1, q+1)})
        abar = y.max()*rng.uniform(0.3, 1.0)
        feats = []
        for l in L:
            if l in Bset: feats.append((l, V.rand_feature(rng, 'two', abar, s)))
            elif y[l-1] > 1e-12: feats.append((l, V.rand_feature(rng, 'IF', y[l-1], s)))
        net = V.build(p, feats); f = V.table(p, net)
        x = np.arange(p); X1, X2 = np.meshgrid(x, x, indexing='ij'); Y = (X1+X2) % p
        fy = np.take_along_axis(f, Y[:, :, None], 2)[:, :, 0]
        h = min((fy-np.take_along_axis(f, ((Y-dd) % p)[:, :, None], 2)[:, :, 0]).min() for dd in range(1, p))
        a = np.zeros(F)
        for l, atoms in feats: a[l-1] = 3**(1-s)*sum(m**r for (m, _, _, _) in atoms)**s
        aB = np.zeros(F); aB[np.array(Bset)-1] = a[np.array(Bset)-1]; aBc = a-aB
        fr, bp = certs[q]; bpp = np.zeros(F)
        for ll, bl in zip(fr, bp):
            k = d*ll % p
            if k == 0: continue
            k = min(k, p-k); bpp[k-1] += tp*abar*bl
        lhs1 = (Kd@(aBc+tp*aB)).min(); lhs2 = (Kd@(aBc+bpp)).min()
        dom = (Kd@bpp-tp*(Kd@aB)).min()
        cost_old = np.sum(a[a > 0]**r); cost_new = np.sum(aBc[aBc > 0]**r)+np.sum(bpp[bpp > 0]**r)
        print(f"R8 p={p} q={q} d={d}: h={h:.4e}  m(a_Bc+tau a_B)={lhs1:.4e}  m(a_Bc+b'')={lhs2:.4e}  Kb''-tau K a_B >= {dom:.2e}  "
              f"cost new/old={cost_new/cost_old:.4f}  h/(G||a||)={h/(G*cost_old**s):.4f}  ok={h <= lhs1+1e-12 and lhs1 <= lhs2+1e-12 and dom >= -1e-12 and cost_new < cost_old}")
