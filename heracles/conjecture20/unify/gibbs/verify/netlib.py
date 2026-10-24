# Independent class-C n=2 network code for the referee (does not import the agent's lib).
import numpy as np
def rho(p): return np.exp(2j*np.pi/p)
def table(U1, U2, W):            # f[x1,x2,c] = sum_i W[i,c] U1[i,x1] U2[i,x2]
    return np.einsum('ia,ib,ic->abc', U1, U2, W)
def gaps(f):                     # g[x1,x2,d] = f[x][y] - f[x][y-d]
    p = f.shape[0]; x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); y = (x1 + x2) % p
    g = np.empty((p, p, p))
    for d in range(p): g[:, :, d] = f[x1, x2, y] - f[x1, x2, (y - d) % p]
    return g
def fhat(f, k1, k2, k3):
    p = f.shape[0]; a = np.arange(p); r = rho(p)
    return np.einsum('abc,a,b,c->', f, r**(-k1*a), r**(-k2*a), r**(-k3*a))
def coeffs(f):                   # beta_l, beta_{l,T}; T in {1},{2},{12}; beta_{j,T}: input index -j on T
    p = f.shape[0]; Fn = (p - 1)//2; out = {}
    for l in range(1, Fn + 1):
        out[(l, ())] = 2*p**-3*fhat(f, l, l, -l)
        out[(l, (1,))] = 2*p**-3*fhat(f, -l, l, -l)
        out[(l, (2,))] = 2*p**-3*fhat(f, l, -l, -l)
        out[(l, (1, 2))] = 2*p**-3*fhat(f, -l, -l, -l)
    return out
def kap(l, d, p): return 1 - np.cos(2*np.pi*l*d/p)
def neuron_vec(p, l, amp, ph):   # amp*cos(2 pi l a/p + ph)
    a = np.arange(p); return amp*np.cos(2*np.pi*l*a/p + ph)
def random_classC(p, rng, per=(1, 3), freqs=None):
    U1, U2, W, fr = [], [], [], []
    for l in (freqs if freqs is not None else range(1, (p-1)//2 + 1)):
        for _ in range(int(rng.integers(per[0], per[1] + 1))):
            amps = rng.exponential(size=3) + 0.05
            ph = rng.uniform(0, 2*np.pi, size=3)
            U1.append(neuron_vec(p, l, amps[0], ph[0])); U2.append(neuron_vec(p, l, amps[1], ph[1]))
            W.append(neuron_vec(p, l, amps[2], ph[2])); fr.append(l)
    return np.array(U1), np.array(U2), np.array(W), np.array(fr)
def atoms(U1, U2, W, fr, p):     # per-neuron atom at own frequency: 2 p^-3 fhat_i(l,l;-l)
    out = []
    for i in range(len(W)):
        l = fr[i]; a = np.arange(p); r = rho(p)
        u1 = (U1[i]*r**(-l*a)).sum(); u2 = (U2[i]*r**(-l*a)).sum(); w = (W[i]*r**(l*a)).sum()   # what(-l) = conj(what(l))
        out.append(2*p**-3*w*u1*u2)
    return np.array(out)
def cost(U1, U2, W, b):
    return sum((np.sum(U1[i]**2) + np.sum(U2[i]**2) + np.sum(W[i]**2))**(b/2) for i in range(len(W)))
def Bp(p): return 2*(6*p)**-1.5
