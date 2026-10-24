# J4 check: explicit class-C networks realising the single-T family.
#  feature l: IF block (nu atoms, xi_it = e^{2 pi i k/nu}) of clean amplitude alpha_l, and an n-atom single-T block
#  on T=[n] (xi_it = e^{2 pi i k/n} e^{i phi_l [t=0]}) of clean amplitude beta_l.
# Compare (i) the margin from the explicit tables (E_width Net, independent code) with the 1D formula,
#         (ii) gamma = h / P_b^s with balanced neurons, against B_p * h_1D / (nu^{1-r} sum alpha^r + n^{1-r} sum beta^r)^s.
import sys, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_width')
import elib as EW
from cjlib import TL
rng = np.random.default_rng(3)

def neuron_tables(p, n, j, c, xi):
    """balanced class-C neuron at frequency j with field c prod_t (1 + xi_t rho^{2 j x_t}) (in units where the field
    of the table equals B-normalised atoms); returns U (n,p), W (p,).  Atom |c| = B_p ||omega||^nu exactly."""
    nu = n+1; a = np.arange(p)
    th = np.angle(xi)/2.0
    # amplitude of each balanced factor: P = 2^n |c| / B_p-normalisation handled by scaling below
    U = np.array([np.cos(2*np.pi*j*a/p + th[t]) for t in range(n)])
    psi = np.angle(c) + th.sum()
    W = np.cos(2*np.pi*j*a/p + psi)
    # raw atom magnitude of (U,W): c_raw = a_w prod(a_t/2) with all amplitudes 1 -> 2^{-n}
    # scale every factor by lam so that lam^nu 2^{-n} = |c|
    lam = (abs(c)*2**n)**(1.0/nu)
    return lam*U, lam*W

def build(p, n, alpha, beta, phi):
    F = (p-1)//2; nu = n+1; Us = []; Ws = []
    for l in range(1, F+1):
        if alpha[l-1] > 0:
            for k in range(nu):
                xi = np.exp(2j*np.pi*k/nu)*np.ones(n)
                U, W = neuron_tables(p, n, l, alpha[l-1]/nu, xi); Us.append(U); Ws.append(W)
        if beta[l-1] > 0:
            for k in range(n):
                xi = np.exp(2j*np.pi*k/n)*np.ones(n); xi[0] *= np.exp(1j*phi[l-1])
                U, W = neuron_tables(p, n, l, beta[l-1]/n, xi); Us.append(U); Ws.append(W)
    return EW.Net(np.array(Us), np.array(Ws))

worst = 0; worstg = 0
for (p, n) in [(5, 2), (7, 2), (7, 3), (11, 2), (5, 4), (11, 3), (13, 2)]:
    F = (p-1)//2
    for trial in range(4):
        s = rng.uniform(1, (n+1)/2); r = 1/s
        alpha = rng.exponential(1, F)*(rng.random(F) < 0.7); beta = rng.exponential(1, F)*(rng.random(F) < 0.7); phi = rng.uniform(-np.pi, np.pi, F)
        if alpha.sum()+beta.sum() == 0: continue
        net = build(p, n, alpha, beta, phi)
        h_tab = net.margin()
        tl = TL(p, n, s, list(range(1, F+1)))
        # field scale: table field Phi_l has clean coefficient conj(beta_l) with beta = 2 p^{-nu} fhat; our atoms are
        # B_p-normalised (|c| = B_p ||omega||^nu) -> the table margin equals (field scale) * h_1D. Determine the scale
        # from the clean coefficient of one feature and compare.
        cl = net.clean()
        l0 = int(np.argmax(alpha+beta)); scale = abs(cl[l0])/(alpha[l0]+beta[l0])
        h_1d = tl.gaps(alpha, beta, phi).min()*scale
        g_tab = net.gamma(s); Bp = EW.Bp(p, n)
        g_1d = Bp*tl.gaps(alpha, beta, phi).min()/(nu_r := (n+1)**(1-r)*np.sum(alpha**r) + n**(1-r)*np.sum(beta**r))**s
        e1 = abs(h_tab-h_1d)/abs(h_tab) if h_tab != 0 else abs(h_1d); e2 = abs(g_tab-g_1d)/abs(g_tab)
        worst = max(worst, e1); worstg = max(worstg, e2)
        print(f"p={p} n={n} s={s:.3f}: h_table={h_tab:.6e} h_1D*scale={h_1d:.6e} (scale {scale:.4e}, B_p {Bp:.4e}) gamma_table={g_tab:.6e} gamma_1D={g_1d:.6e}")
print(f"max rel err margin {worst:.2e}, gamma {worstg:.2e}")
