# Lemma 1(b),(c) sanity: decoupled networks Theta(theta_1, A) (relabelled copies of ONE single feature scaled by A_l^{1/nu})
# satisfy gamma_lam(Theta; sigma) >= psi(sigma) = c(theta_1) m_lam(A) / (sum_{l,i} (A_l a_i)^{1/sigma})^sigma for all sigma,
# with entropy H(z)+H(theta_1); and the weighted bound is attained at s = 1 by IF networks (W^lam(1) = D^lam(1)).
import numpy as np
from elib import Net, Bp, programme_vertices, kap, wmargin, entropy, zshares, Hent, if_block, freq_proj, if_network
rng = np.random.default_rng(3)

def relabel(U, W, l, p):
    idx = (l * np.arange(p)) % p
    return U[:, :, idx], W[:, idx]

def cm_min(net):
    # c(theta) = min_x cm(Phi_1(x)) for a single feature (outputs in span{1,cos,sin} at frequency 1)
    p, n = net.p, net.n; f = net.table().reshape(-1, p); Y = net.Y
    Phi = (2 / p) * np.exp(2j * np.pi * Y / p) * (f @ np.exp(-2j * np.pi * np.arange(p) / p))
    return np.min(Phi.real - np.abs(Phi.imag) / np.tan(np.pi / p))

worst = 1e9; worstH = 0
for p, n in [(5, 2), (7, 2), (7, 3), (11, 2)]:
    F = (p - 1) // 2; nu = n + 1
    for trial in range(6):
        # single feature: IF block at frequency 1 + small noise, outputs projected to span{1,cos,sin}
        U1, W1 = if_block(p, n, 1); m1 = U1.shape[0]
        U1 = U1 + 0.1 * rng.normal(size=U1.shape); W1 = (W1 + 0.1 * rng.normal(size=W1.shape)) @ (freq_proj(p, 0) + freq_proj(p, 1))
        th1 = Net(U1, W1, 2.0); c1 = cm_min(th1)
        if c1 <= 0: continue
        S = sorted(rng.choice(np.arange(1, F + 1), size=rng.integers(1, F + 1), replace=False).tolist())
        A = np.zeros(F); A[np.array(S) - 1] = rng.uniform(0.2, 1, len(S))
        lam = rng.uniform(0.7, 1.4, F)
        Us, Ws = [], []
        for l in S:
            Ul, Wl = relabel(U1, W1, l, p); k = A[l - 1] ** (1 / nu)
            Us.append(Ul * k); Ws.append(Wl * k)
        a = ((U1 ** 2).sum((1, 2)) + (W1 ** 2).sum(1)) ** (nu / 2)
        mA = np.min(kap(p) @ (lam * A))
        for sig in [1.0, 1.3, nu / 2 * 0.9, nu / 2]:
            b = nu / sig
            Th = Net(np.concatenate(Us), np.concatenate(Ws), b)
            gl = wmargin(Th, lam) / Th.cost() ** sig
            Z = np.sum([(A[l - 1] * ai) ** (1 / sig) for l in S for ai in a])
            psi = c1 * mA / Z ** sig
            worst = min(worst, gl / psi)
            zA = np.array([A[l - 1] ** (1 / sig) for l in S]); zA /= zA.sum()
            worstH = max(worstH, abs(entropy(Th) - (Hent(zA) + entropy(Net(U1, W1, b)))))
print(f"Lemma 1(b): min gamma_lam(Theta)/psi - 1 = {worst-1:+.2e} (>= 1);  max |H(Theta) - H(z) - H(theta_1)| = {worstH:.1e}")

# Lemma 1(c): at s = 1 (b = nu) the weighted IF network with amplitudes y^lam attains B_p Gamma^lam_1
mx = 0
for p, n in [(5, 2), (7, 2), (7, 3), (11, 2)]:
    F = (p - 1) // 2; nu = n + 1
    for t in range(3):
        lam = rng.uniform(0.7, 1.4, F)
        Gm, maxs, _ = programme_vertices(p, 1.0, lam)
        net = if_network(p, n, maxs[0], b=float(nu))
        r = wmargin(net, lam) / (Bp(p, n) * Gm * net.cost())
        mx = max(mx, abs(r - 1))
print(f"Lemma 1(c): weighted IF networks at b = nu: max |h_lam/(B_p Gamma^lam_1 P) - 1| = {mx:.1e}")
