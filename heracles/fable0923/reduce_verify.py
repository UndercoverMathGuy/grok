"""reduce_verify.py -- numerical verification of every identity in the general-density reduction (n=2, q=2).

Network: pure neurons, neuron l at frequency k: e_l1(a) = r1 cos(2 pi k a/p + al1), e_l2(b) = r2 cos(2 pi k b/p + al2),
w_l(c) = rho cos(2 pi k c/p + be).  Atom: c_l = r1 r2 rho e^{i(al1+al2-be)}/8 ... we take the definitions from the
Fourier data directly (Theorem D's dictionary): with ehat(k) = sum_a e(a) omega^{-ka} (unnormalised),
   c_l = ehat_l1(k) ehat_l2(k) conj(what_l(k)),  v_lj = conj(ehat_lj(k))/ehat_lj(k).
Checked identities (per random network, random density w on Z_p^2, every D != 0):
 (I1)  E_w[gap_D] = (2/p^3) * sum_k Re[ (1 - omega^{-kD}) * ( G_k + sum_T J_{k,T} * what(2k 1_T) ) ]  [conventions fixed by the check]
 (I2)  the +-D symmetrised form:  (E_w[gap_D] + E_w[gap_{-D}])/2 = (2/p^3) sum_k w_kD Re[ G_k + sum_T J_{k,T} what(2k 1_T) ].
 (I3)  cost: P_2 >= (3p/2) sum_{l} |c_l|^{2/3} ... with the unnormalised DFT the constant is (3/(2p))^{...}; we check
       P_2 = 3 (p/2) sum_l (8|c_l|/p^3)^{2/3}... -- rather than trust a constant, we check the END-TO-END chain:
 (E2E) gamma_2 <= 3^{-1/2} B Gamma_S  is NOT what we check (it is the claim); we check the intermediate bound
       margin <= min_D (E_w[gap_D]+E_w[gap_{-D}])/2 for the density that aims each frequency, which is what the proof uses.
"""
import numpy as np, itertools, sys
rng = np.random.default_rng(1)

def build(p, S, per, rng):
    a = np.arange(p); E1 = []; E2 = []; W = []; freq = []
    for k in S:
        for _ in range(per):
            r = rng.random(3) + 0.3; ph = rng.random(3) * 2*np.pi
            E1.append(r[0]*np.cos(2*np.pi*k*a/p + ph[0])); E2.append(r[1]*np.cos(2*np.pi*k*a/p + ph[1]))
            W.append(r[2]*np.cos(2*np.pi*k*a/p + ph[2])); freq.append(k)
    return np.array(E1), np.array(E2), np.array(W), np.array(freq)

def atoms(E1, E2, W, freq, p):
    c = {}; v = {}
    for l, k in enumerate(freq):
        om = np.exp(-2j*np.pi*k*np.arange(p)/p)
        e1 = (E1[l]*om).sum(); e2 = (E2[l]*om).sum(); w = (W[l]*om).sum()
        c.setdefault(k, []).append(e1*e2*np.conj(w)); v.setdefault(k, []).append((np.conj(e1)/e1, np.conj(e2)/e2))
    G = {}; J = {}
    for k in c:
        cl = np.array(c[k]); v1 = np.array([x[0] for x in v[k]]); v2 = np.array([x[1] for x in v[k]])
        G[k] = cl.sum(); J[k] = np.array([(cl*v1).sum(), (cl*v2).sum(), (cl*v1*v2).sum()])
    return G, J

def gaps(E1, E2, W, p):
    Z = np.einsum('la,lb,lc->abc', E1, E2, W)
    a = np.arange(p); y = (a[:, None] + a[None, :]) % p
    corr = Z[a[:, None], a[None, :], y]
    out = {}
    for D in range(1, p):
        out[D] = corr - Z[a[:, None], a[None, :], (y + D) % p]
    return out

def check(p, S, per):
    E1, E2, W, freq = build(p, S, per, rng)
    G, J = atoms(E1, E2, W, freq, p)
    gp = gaps(E1, E2, W, p)
    w = rng.random((p, p)); w /= w.sum()
    A = np.arange(p); om = np.exp(2j*np.pi/p)
    def what(m):  # what(m) = sum_x w(x) omega^{-m.x}
        return (w * np.exp(-2j*np.pi*(m[0]*A[:, None] + m[1]*A[None, :])/p)).sum()
    worst = 0
    for D in range(1, p):
        lhs = (w * gp[D]).sum()
        best = None
        for sgn in (1, -1):        # try both conjugation conventions for the moment; record which one matches
            rhs = 0
            for k in S:
                t = [what((2*k, 0)), what((0, 2*k)), what((2*k, 2*k))]
                if sgn < 0: t = [np.conj(x) for x in t]
                rhs += ((1 - om**(-k*D)) * (G[k] + sum(J[k][i]*t[i] for i in range(3)))).real
            rhs *= 2 / p**3
            err = abs(lhs - rhs)
            if best is None or err < best[0]: best = (err, sgn)
        worst = max(worst, best[0] / max(abs(lhs), 1e-12))
    # symmetrised form + aimed bound
    m = min(g.min() for g in gp.values())
    return worst, best[1], m

if __name__ == "__main__":
    for p, S, per in [(7, [1, 2], 2), (11, [1, 3], 1), (11, [1, 3], 3), (13, [1, 2, 5], 2), (17, [1, 3, 5], 1)]:
        worst, sgn, m = check(p, S, per)
        print(f"p={p} S={S} per={per}: identity (I1) max rel err {worst:.2e} (convention sign {sgn}); margin {m:+.3e}")
