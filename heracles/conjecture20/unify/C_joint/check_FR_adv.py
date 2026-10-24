# (FR) on the remaining adversarial facts: all stored SPLIT/E_dual violators (p=5,7,11), the (5,6) width-6 network.
import sys, json, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import *
from check_FR_mixed import FRratio
for p in [5, 7, 11]:
    d = np.load(f'/home/user/grok/heracles/conjecture20/unify/split/viol_p{p}_n2_b2.npz')
    for k in range(8):
        if f'U{k}' not in d: break
        net = EW.Net(d[f'U{k}'], d[f'W{k}'])
        if net.margin() <= 0: continue
        G = Gam(p, 1.5); gIF = 3**(-0.5)*EW.Bp(p, 2)*G
        a, b = FRratio(net, 1.5)
        print(f"viol p={p} net{k}: gamma/gamma_IF={net.gamma(1.5)/gIF:.4f}  gamma/(FR bound)={a:.4f}")
d = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/kkt/oned_p5_n6.json'))
p, n = d['p'], d['n']; lam, lv, av = np.array(d['lam']), np.array(d['logV']), np.array(d['argV']); lv = lv - lv.max()
eps = np.array([1.0 if a < p/2 else -1.0 for a in range(p)]); e = np.array([2*a if a < p/2 else 2*a-p for a in range(p)], float)
rho = np.exp(2j*np.pi*np.arange(p)/p); U1 = []; W = []
for i in range(len(lam)):
    u = eps*np.exp(lam[i]*e); w = np.real(np.exp(lv[i]+1j*av[i])*rho)
    nu_, nw = np.linalg.norm(u), np.linalg.norm(w); t = (nw*nu_**n)**(1.0/(n+1)); U1.append(u/nu_*t); W.append(w/nw*t)
U1 = np.array(U1); W = np.array(W); s = (n+1)/2
h, mn, M, Q, P = slotsym_FR(U1, W, p, n, s); G = Gam(p, s); g = h/P**s; gIF = (n+1)**(1-s)*EW.Bp(p, n)*G
print(f"(5,6) width-6 network: gamma/gamma_IF={g/gIF:.5f} (stored 0.17723); gamma/(FR bound)={g/(EW.Bp(p,n)*G*(M.sum()/Q.sum())**s):.4f}; h/Gam(min_x)^s={h/(G*mn**s):.4f}")
