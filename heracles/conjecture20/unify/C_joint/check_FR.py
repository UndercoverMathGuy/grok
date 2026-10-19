# Check of the field-row theorem (FR):
#  (FR0) every network:  h <= Gamma_s * min_x (sum_l (Re Phi_l(x))_+^r)^s <= Gamma_s * (sum_l M_l)^s,  M_l = E_x (Re Phi_l)_+^r
#  (FR)  class C:        gamma <= gamma_IF * (sum_l Q_l Xi_l / sum_l Q_l)^s,   Xi_l = nu^{1-r} M_l / Q_l
import sys, glob, json, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import *
rng = np.random.default_rng(17)
if __name__ == '__main__':
    worst0 = -1e9; worstFR = -1e9; nC = 0; nB = 0; worstB = -1e9
    for trial in range(120):
        p, n = [(3, 2), (5, 2), (5, 3), (7, 2), (7, 3), (11, 2)][trial % 6]
        s = rng.uniform(1, (n+1)/2); r = 1/s; nu = n+1
        if trial % 2 == 0:   # class C, positive margin: perturb an IF network at the programme allocation
            G0, ys = gamma_exact(p, s) if p > 3 else (Gam(3, s), [np.array([1.0])])
            U, W = EW.if_network(p, n, ys[0]); m = len(W); a = np.arange(p)
            eps = rng.choice([0.02, 0.1, 0.3])
            for i in range(m):   # keep frequency purity: perturb amplitudes and phases of each class-C factor
                j = np.argmax(np.abs(np.fft.fft(W[i]))[1:(p+1)//2]) + 1
                for t in range(n):
                    amp = np.linalg.norm(U[i, t])*np.sqrt(2/p)*(1+eps*rng.normal()); U[i, t] = amp*np.cos(2*np.pi*j*a/p + np.angle(np.fft.fft(U[i, t])[j]) + eps*rng.normal())
                amp = np.linalg.norm(W[i])*np.sqrt(2/p)*(1+eps*rng.normal()); W[i] = amp*np.cos(2*np.pi*j*a/p + np.angle(np.fft.fft(W[i])[j]) + eps*rng.normal())
            classC = True
        else:
            m = rng.integers(2, 9); U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p)); classC = False
        net = EW.Net(U, W); h = net.margin(); G = Gam(p, s)
        mn, M = FR_terms(net, s); b0 = G*max(mn, 0)**s; b1 = G*M.sum()**s
        worst0 = max(worst0, (h - b0)/abs(b1), (b0 - b1)/abs(b1)); nB += 1
        if classC and h > 0:
            Q = clean_Q(net, s); P = net.cost(nu/s); Bp = EW.Bp(p, n); live = Q > 1e-14
            worstB = max(worstB, Q.sum()/(Bp**r*P) - 1)
            Xi = nu**(1-r)*M[live]/Q[live]; avg = np.sum(Q[live]*Xi)/np.sum(Q[live])
            worstFR = max(worstFR, (h/P**s)/(nu**(1-s)*Bp*G*avg**s) - 1); nC += 1
    print(f"(FR0) on {nB} networks: max relative violation {worst0:.2e} (<=0 expected)")
    print(f"(FR) on {nC} positive-margin class-C networks: max gamma/(gamma_IF avgXi^s) - 1 = {worstFR:.2e} (<=0); budget sum Q/(B_p^r P_b) - 1 <= {worstB:.2e}")
    print("IF networks at programme allocations (equality expected):")
    for (p, n, s) in [(5, 2, 1.5), (7, 2, 1.3), (7, 3, 1.7), (11, 2, 1.4), (5, 3, 2.0)]:
        G, ys = gamma_exact(p, s); y = ys[0]; U, W = EW.if_network(p, n, y); net = EW.Net(U, W)
        r = 1/s; nu = n+1; mn, M = FR_terms(net, s); Q = clean_Q(net, s); live = Q > 1e-14
        Xi = nu**(1-r)*M[live]/Q[live]; avg = np.sum(Q[live]*Xi)/np.sum(Q[live]); gIF = nu**(1-s)*EW.Bp(p, n)*G
        print(f"   p={p} n={n} s={s}: gamma/gamma_IF={net.gamma(s)/gIF:.8f}  avgXi^s={avg**s:.8f}")
    print("stored local maxima (E_kkt), FR0 tightness:")
    for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p*_k*.npz')):
        d = np.load(f); net = EW.Net(d['U'], d['W']); p, n = net.p, net.n; nu = n+1; b = float(d['b']); s = nu/b; G = Gam(p, s)
        h = net.margin()
        if h <= 0: continue
        mn, M = FR_terms(net, s); gIF = nu**(1-s)*EW.Bp(p, n)*G
        print(f"   {f.split('/')[-1]}: gamma/gamma_IF={net.gamma(s)/gIF:.5f}  h/Gam(min_x)^s={h/(G*mn**s):.4f}  h/Gam(sumM)^s={h/(G*M.sum()**s):.4f}")
    d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz')
    net = EW.Net(d['U0'], d['W0']); s = 1.5; h = net.margin(); G = Gam(5, s); mn, M = FR_terms(net, s)
    print(f"(5,2,2) mixed violator: gamma/gamma_IF={net.gamma(s)/(3**(1-s)*EW.Bp(5,2)*G):.4f}; h/Gam(min_x)^s={h/(G*mn**s):.4f}; h/Gam(sumM)^s={h/(G*M.sum()**s):.4f}")
    import exact_certificate as ec
    for n in [11, 13]:
        dd = ec.NETWORKS[n]
        U1 = np.array([[float(x) for x in row] for row in dd['u']]); W = np.array([[float(x) for x in row] for row in dd['w']])
        s = (n+1)/2; r = 1/s; nu = n+1; p = 3
        h, mn, M, Q, P = slotsym_FR(U1, W, p, n, s); G = Gam(3, s)
        g = h/P**s; gIF = nu**(1-s)*EW.Bp(p, n)*G; Xi = nu**(1-r)*M/Q
        print(f"p=3 n={n} winner: gamma/gamma_IF={g/gIF:.4f}; h/Gam(min_x)^s={h/(G*mn**s):.4f}; h/Gam(sumM)^s={h/(G*M.sum()**s):.4f}; "
              f"clean-budget Xi^s={Xi[0]**s:.4f}; clean budget sum Q/(B_p^r P_b)={Q.sum()/(EW.Bp(p,n)**r*P):.4f} (<1: DC cost not in clean budget)")
