# Efficiency-aware residual PWE on stored/adversarial networks:
#   FRE := min_x min_delta sum_l Re Phi_l(x) kappa_l(delta)   (>= h),   ratio  (FRE / P_b^s) / gamma_IF.
import sys, glob, json, itertools, numpy as np
from math import factorial
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint'); sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import EW, fields_x, Gam
def kap(p):
    F = (p-1)//2; return 1 - np.cos(2*np.pi*np.outer(np.arange(1, p), np.arange(1, F+1))/p)
def fre_general(net, s):
    p, n = net.p, net.n; nu = n+1; Phi = fields_x(net); F = Phi.shape[0]
    Y = Phi.real.reshape(F, -1); FRE = (kap(p) @ Y).min()
    P = net.cost(nu/s); gIF = nu**(1-s)*EW.Bp(p, n)*Gam(p, s)
    return FRE/P**s/gIF, net.gamma(s)/gIF
def fre_slotsym(U1, W, p, n, s):
    F = (p-1)//2; m = len(U1); rho = np.exp(2j*np.pi/p); K = kap(p); FRE = np.inf; h = np.inf
    for N in itertools.product(range(n+1), repeat=p-1):
        if sum(N) > n: continue
        cnt = list(N) + [n-sum(N)]; y = sum(a*cnt[a] for a in range(p)) % p
        z = np.prod([U1[:, a]**cnt[a] for a in range(p)], axis=0); f = W.T @ z
        h = min(h, min(f[y]-f[c] for c in range(p) if c != y))
        Phi = np.array([(2.0/p)*rho**(l*y)*np.sum(f*rho**(-l*np.arange(p))) for l in range(1, F+1)])
        FRE = min(FRE, (K @ Phi.real).min())
    nu = n+1; sq = n*(U1**2).sum(1) + (W**2).sum(1); P = np.sum(sq**(nu/(2*s)))
    gIF = nu**(1-s)*EW.Bp(p, n)*Gam(p, s)
    return FRE/P**s/gIF, h/P**s/gIF
rows = []
for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p*_k*.npz')):
    d = np.load(f); net = EW.Net(d['U'], d['W']); s = (net.n+1)/float(d['b'])
    if net.margin() <= 0: continue
    a, g = fre_general(net, s); rows.append((f.split('/')[-1], g, a))
for p in [5, 7, 11]:
    d = np.load(f'/home/user/grok/heracles/conjecture20/unify/split/viol_p{p}_n2_b2.npz')
    for k in range(8):
        if f'U{k}' not in d: break
        net = EW.Net(d[f'U{k}'], d[f'W{k}'])
        if net.margin() <= 0: continue
        a, g = fre_general(net, 1.5); rows.append((f'viol_p{p}:{k} (mixed)', g, a))
for name, g, a in rows: print(f"  {name}: gamma/gamma_IF = {g:.4f}   FRE/IF = {a:.4f}")
print(f"  max FRE/IF over {len(rows)} stored networks: {max(a for _,_,a in rows):.5f}")
import exact_certificate as ec
for n in [11, 13]:
    dd = ec.NETWORKS[n]
    U1 = np.array([[float(x) for x in row] for row in dd['u']]); W = np.array([[float(x) for x in row] for row in dd['w']])
    a, g = fre_slotsym(U1, W, 3, n, (n+1)/2)
    print(f"  p=3 n={n} DC winner (not class C): gamma/gamma_IF = {g:.4f}  FRE/IF = {a:.4f}")
d = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/kkt/oned_p5_n6.json'))
p, n = d['p'], d['n']; lam, lv, av = np.array(d['lam']), np.array(d['logV']), np.array(d['argV']); lv = lv - lv.max()
eps = np.array([1.0 if a < p/2 else -1.0 for a in range(p)]); e = np.array([2*a if a < p/2 else 2*a-p for a in range(p)], float)
rho = np.exp(2j*np.pi*np.arange(p)/p); U1 = []; W = []
for i in range(len(lam)):
    u = eps*np.exp(lam[i]*e); w = np.real(np.exp(lv[i]+1j*av[i])*rho)
    nu_, nw = np.linalg.norm(u), np.linalg.norm(w); t = (nw*nu_**n)**(1.0/(n+1)); U1.append(u/nu_*t); W.append(w/nw*t)
a, g = fre_slotsym(np.array(U1), np.array(W), p, n, (n+1)/2)
print(f"  (5,6) width-6 single feature (DC inputs): gamma/gamma_IF = {g:.4f}  FRE/IF = {a:.4f}")
