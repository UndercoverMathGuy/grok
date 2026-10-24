"""Independent referee checks at (p,n)=(5,2) for R2 (mode budget), R3 (Pythagorean cost), R4 (margin ledger),
R5 (R-elimination), R6 (HS family: probability, equivariance).  Own implementation (no p52lib imports).
Networks: arbitrary real neurons (DC + all frequencies), beta_2 forced to 0 exactly by solving for one neuron's
output frequency-2 component."""
import numpy as np, itertools
rng = np.random.default_rng(12345)
p = 5; rho = np.exp(2j*np.pi/p)
B5 = 2*(30.0)**-1.5
X = np.array(list(itertools.product(range(p), repeat=2)))
def hat(v):  # vhat(k) = sum_a v(a) rho^{-k a}
    a = np.arange(p); return np.array([np.sum(v*rho**(-k*a)) for k in range(p)])
def table(U1, U2, W):
    return np.einsum('ia,ib,ic->abc', U1, U2, W)
def gaps(f):
    G = np.zeros((p, p, 4))
    for x1 in range(p):
        for x2 in range(p):
            y = (x1+x2) % p
            for di, d in enumerate([1, 2, 3, 4]):
                G[x1, x2, di] = f[x1, x2, y] - f[x1, x2, (y-d) % p]
    return G
def fields(f):
    """Phi_l(x) = (2/p) rho^{l y} sum_c f[c] rho^{-lc}"""
    out = {}
    for l in (1, 2):
        Ph = np.zeros((p, p), complex)
        for x1 in range(p):
            for x2 in range(p):
                y = (x1+x2) % p
                Ph[x1, x2] = (2/p)*rho**(l*y)*np.sum(f[x1, x2, :]*rho**(-l*np.arange(p)))
        out[l] = Ph
    return out
def coef(Ph, chi):
    return np.mean([Ph[x1, x2]*rho**(-(chi[0]*x1+chi[1]*x2)) for x1, x2 in X])
def rand_net(m, scale_dc=1.0):
    U1 = rng.normal(size=(m, p)); U2 = rng.normal(size=(m, p)); W = rng.normal(size=(m, p))
    U1 += scale_dc*rng.normal(size=(m, 1)); U2 += scale_dc*rng.normal(size=(m, 1))
    return U1, U2, W
def beta_list(U1, U2, W, j, T=()):
    """per-neuron beta_{ij,T}: input index -j on slots in T"""
    out = []
    for i in range(len(W)):
        k1 = -j if 0 in T else j; k2 = -j if 1 in T else j
        out.append(2*p**-3*np.conj(hat(W[i])[j])*hat(U1[i])[k1 % p]*hat(U2[i])[k2 % p])
    return np.array(out)
def kill_beta2(U1, U2, W):
    """modify W[0]'s frequency-2 component so that beta_2 = 0 exactly (in exact arithmetic)"""
    b = beta_list(U1, U2, W, 2)
    rest = b[1:].sum()
    c0 = 2*p**-3*hat(U1[0])[2]*hat(U2[0])[2]
    # need conj(what0(2)) c0 = -rest  -> what0(2) = conj(-rest/c0)
    target = np.conj(-rest/c0)
    cur = hat(W[0])[2]
    dv = target - cur
    a = np.arange(p)
    # add real vector with hat at 2 equal dv: v(a) = (2/p) Re(dv rho^{2a})
    W = W.copy(); W[0] = W[0] + (2/p)*np.real(dv*rho**(2*a))
    return W
# ---------------------------------------------------------------- R2 mode budget
worst2 = -np.inf
for t in range(3000):
    u1, u2, w = rng.normal(size=p), rng.normal(size=p), rng.normal(size=p)
    if t % 3 == 0: u1 += 3*rng.normal(); w += 2*rng.normal()
    if t % 5 == 0:  # near-budget-tight neurons: pure frequency-1 content
        a = np.arange(p); ph = rng.uniform(0, 2*np.pi, 3)
        u1 = np.cos(2*np.pi*a/p+ph[0]); u2 = np.cos(2*np.pi*a/p+ph[1]); w = np.cos(2*np.pi*a/p+ph[2]) + 1e-3*rng.normal(size=p)
    nrm2 = np.sum(u1**2)+np.sum(u2**2)+np.sum(w**2); e = B5*nrm2**1.5
    H1, H2, Hw = hat(u1), hat(u2), hat(w)
    beta = 2*p**-3*np.conj(Hw[1])*H1[1]*H2[1]
    tot = 0.0; own = 0.0
    for j in (1, 2):
        for k1 in range(p):
            for k2 in range(p):
                mm = abs(2*p**-3*Hw[j]*H1[k1]*H2[k2])**2
                if j == 1 and k1 in (1, 4) and k2 in (1, 4): own += mm
                else: tot += mm
    assert abs(own - 4*abs(beta)**2) < 1e-12*max(1, own)
    worst2 = max(worst2, (tot - 4*(e**2-abs(beta)**2))/e**2)
print(f"R2: max over 3000 neurons of [non-own - 4(e^2-|beta|^2)]/e^2 = {worst2:.3e}  (<=0 required)")
# ---------------------------------------------------------------- R3 Pythagorean cost
worst3 = -np.inf
for t in range(600):
    m = rng.integers(1, 7)
    U1, U2, W = rand_net(m, scale_dc=rng.uniform(0, 2))
    if t % 4 == 0:
        a = np.arange(p)
        for i in range(m):
            ph = rng.uniform(0, 2*np.pi, 3); sc = rng.uniform(0.2, 2)
            U1[i] = sc*np.cos(2*np.pi*a/p+ph[0]); U2[i] = sc*np.cos(2*np.pi*a/p+ph[1]); W[i] = sc*np.cos(2*np.pi*a/p+ph[2])
    b = beta_list(U1, U2, W, 1)
    nr = np.sum(U1**2, 1)+np.sum(U2**2, 1)+np.sum(W**2, 1)
    e = B5*nr**1.5; rh = np.sqrt(np.maximum(e**2-abs(b)**2, 0)); R = 2*rh.sum()
    for s in (1.0, 1.25, 1.5):
        r = 1/s; bb = 3*r; P = np.sum(nr**(bb/2))
        P1 = 3**(1-s)*np.sum(abs(b)**r)**s
        lhs = 3**(1-s)*B5*P**s; rhs = np.sqrt(P1**2 + 3**(2-2*s)*R**2/4)
        worst3 = max(worst3, (rhs-lhs)/lhs)
print(f"R3: max relative [sqrt(P1^2+cR^2) - 3^(1-s)B5 P^s] = {worst3:.3e}  (<=0 required)")
# ---------------------------------------------------------------- R4/R5/R6 ledger with HS certificates
E = [(a, b) for a in range(3) for b in range(3)]
def hs_q(Gs):
    """q(x,delta) from 9x9 PSD Gram matrices (4 of them), normalised by total trace"""
    M = sum(np.trace(G).real for G in Gs)
    q = np.zeros((p, p, 4))
    for x1, x2 in X:
        th = 2*np.pi*2*np.array([x1, x2])/p
        v = np.array([np.exp(1j*(e[0]*th[0]+e[1]*th[1])) for e in E])
        for d in range(4):
            q[x1, x2, d] = np.real(np.conj(v) @ Gs[d] @ v)/25/M
    return q
def rand_G():
    Gs = []
    for d in range(4):
        Z = rng.normal(size=(9, rng.integers(1, 10))) + 1j*rng.normal(size=(9, 1))
        Gs.append(Z @ Z.conj().T)
    return Gs
c = {j: np.array([1 - rho**(-j*d) for d in (1, 2, 3, 4)]) for j in (1, 2)}
worst_h = -np.inf; worst_id = 0; worst_b = -np.inf; worst_q = np.inf; worst_sum = 0; worst_ratio = -np.inf
for t in range(150):
    m = rng.integers(2, 7)
    U1, U2, W = rand_net(m, scale_dc=rng.uniform(0, 2))
    W = kill_beta2(U1, U2, W)
    f = table(U1, U2, W); G = gaps(f); h = G.min()
    Ph = fields(f)
    b2 = coef(Ph[2], (0, 0))
    assert abs(b2) < 1e-12*np.max(abs(f)), b2
    A = np.conj(coef(Ph[1], (0, 0)))
    bT = {T: np.conj(coef(Ph[1], chi)) for T, chi in [((0,), (2, 0)), ((1,), (0, 2)), ((0, 1), (2, 2))]}
    # check against direct sign-flip formula (paper Lemma 6 convention: input index -j on T)
    for T in bT:
        assert abs(bT[T] - beta_list(U1, U2, W, 1, T).sum()) < 1e-12
    assert abs(A - beta_list(U1, U2, W, 1).sum()) < 1e-12
    nr = np.sum(U1**2, 1)+np.sum(U2**2, 1)+np.sum(W**2, 1)
    bi = beta_list(U1, U2, W, 1); e = B5*nr**1.5; R = 2*np.sum(np.sqrt(np.maximum(e**2-abs(bi)**2, 0)))
    Gs = rand_G(); q = hs_q(Gs)
    worst_q = min(worst_q, q.min()); worst_sum = max(worst_sum, abs(q.sum()-1))
    sqg = np.sum(q*G)
    worst_h = max(worst_h, h - sqg)
    # Fourier data of C_j
    Cj = {j: np.einsum('abd,d->ab', q, c[j]) for j in (1, 2)}
    def Chat(j, psi): return np.mean([Cj[j][x1, x2]*rho**(-(psi[0]*x1+psi[1]*x2)) for x1, x2 in X])
    a0 = 25*Chat(1, (0, 0)); aT = {T: 25*Chat(1, (-chi[0] % p, -chi[1] % p)) for T, chi in [((0,), (2, 0)), ((1,), (0, 2)), ((0, 1), (2, 2))]}
    own_chis = [(0, 0), (2, 0), (0, 2), (2, 2)]
    nonown = 0; n2 = 0
    for j in (1, 2):
        for chi in itertools.product(range(p), repeat=2):
            if (j == 1 and chi in own_chis) or (j == 2 and chi == (0, 0)): continue
            ch = Chat(j, ((-chi[0]) % p, (-chi[1]) % p))
            nonown += 25*np.real(coef(Ph[j], chi)*ch); n2 += abs(25*ch)**2
    a = np.real(np.conj(A)*a0) + sum(np.real(np.conj(bT[T])*aT[T]) for T in bT)
    worst_id = max(worst_id, abs(sqg - (a + nonown)))
    n = np.sqrt(n2)
    worst_b = max(worst_b, sqg - (a + R*n))
    # R5 + R3 corollary at s = 1.5 with F = P1 (exact payment): ratio <= sqrt(a+^2/P1^2 + 12 n^2)/Gamma
    s = 1.5; P = np.sum(nr); P1 = 3**(1-s)*np.sum(abs(bi)**(1/s))**s
    gam = h/P**s; gIF = 3**(1-s)*B5*2.5*2**-s
    bound = np.sqrt(max(a, 0)**2/P1**2 + 12*n**2)/(2.5*2**-s)
    worst_ratio = max(worst_ratio, gam/gIF - bound)
print(f"R6: min q = {worst_q:.3e} (>=0), max |sum q - 1| = {worst_sum:.2e}")
print(f"R4: max[h - sum q g] = {worst_h:.3e} (<=0); max |sum q g - (a + nonown pairing)| = {worst_id:.2e}; max[sum q g - (a + R n)] = {worst_b:.3e} (<=0)")
print(f"R5: max[gamma/gamma_IF - sqrt(a+^2/P1^2+12n^2)/Gamma] at s=1.5 = {worst_ratio:.3e} (<=0)")
# ---------------------------------------------------------------- R6 equivariance under torus action
Gs = rand_G()
def cert_coeffs(Gs):
    M = sum(np.trace(G).real for G in Gs)
    wh = []
    for G in Gs:
        d = {}
        for i, e in enumerate(E):
            for k, e2 in enumerate(E):
                f_ = (e2[0]-e[0], e2[1]-e[1]); d[f_] = d.get(f_, 0) + G[i, k]
        wh.append(d)
    ds = lambda cv, f_: sum(cv[k]*wh[k].get(f_, 0) for k in range(4))/M
    a0 = ds(c[1], (0, 0)); aT = [ds(c[1], (-1, 0)), ds(c[1], (0, -1)), ds(c[1], (-1, -1))]
    n2 = 0
    for f1 in range(-2, 3):
        for f2 in range(-2, 3):
            f_ = (f1, f2)
            if f_ not in [(0, 0), (-1, 0), (0, -1), (-1, -1)]: n2 += abs(ds(c[1], f_))**2
            if f_ != (0, 0): n2 += abs(ds(c[2], f_))**2
    return a0, aT, np.sqrt(n2)
a0, aT, n = cert_coeffs(Gs)
t1, t2 = 0.7, -2.1
Dt = np.diag([np.exp(-1j*(e[0]*t1+e[1]*t2)) for e in E])
Gr = [Dt.conj().T @ G @ Dt for G in Gs]
a0r, aTr, nr_ = cert_coeffs(Gr)
print("R6 torus: |a0'-a0| =", abs(a0r-a0), " |n'-n| =", abs(nr_-n), " a_T'/a_T phases:", [np.round(np.angle(aTr[k]/aT[k]), 6) for k in range(3)], " expected", [round(t1, 6), round(t2, 6), round(t1+t2, 6)])
