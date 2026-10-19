# N2: weighted restricted universal bound  h_lam(theta) <= B_p Gamma^{lam,(S)}_s P^s  for theta in N_S (all s),
#     hence W^lam_S(1) = D^lam_S(1) (pi(1)=1).  N3: tax derivative d/dmu_l log gamma_mu = -(nu/2) E_l.
#     Also: temperature identity d/ds log gamma = -H on the same random nets.
import numpy as np
from elib import Net, Bp, programme_vertices, wmargin, Evec, entropy, freq_proj, neuron_sq
rng = np.random.default_rng(1)
worst = -1e9; cnt = 0
for p, n in [(5, 2), (7, 2), (5, 3), (7, 3), (11, 2)]:
    F = (p - 1) // 2; nu = n + 1
    for trial in range(40):
        m = rng.integers(2, 9)
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        # restrict to a random feature class S: remove output content at frequencies outside S
        S = sorted(rng.choice(np.arange(1, F + 1), size=rng.integers(1, F + 1), replace=False).tolist())
        Pm = freq_proj(p, 0) + sum(freq_proj(p, l) for l in S)
        if trial % 3 == 0:   # IF-like start + noise so that h > 0 often
            from elib import if_network
            y = np.zeros(F); y[np.array(S) - 1] = rng.uniform(0.3, 1, len(S))
            net0 = if_network(p, n, y); U = net0.U + 0.05 * rng.normal(size=net0.U.shape); W = net0.W + 0.05 * rng.normal(size=net0.W.shape)
        W = W @ Pm
        lam = rng.uniform(0.6, 1.6, F)
        for b in [2.0, 2.0 + rng.uniform(0, nu - 2), float(nu)]:
            net = Net(U, W, b); s = net.s
            hl = wmargin(net, lam)
            Gm = programme_vertices(p, s, lam, S)[0]
            ratio = hl / (Bp(p, n) * Gm * net.cost() ** s)
            worst = max(worst, ratio); cnt += 1
print(f"N2 weighted restricted universal bound: max h_lam/(B_p Gamma^lam_S P^s) = {worst:.6f} over {cnt} cases (must be <= 1)")

# N3 tax derivative and temperature identity
err_tax = 0; err_temp = 0
for p, n in [(5, 2), (7, 3), (11, 2)]:
    F = (p - 1) // 2; nu = n + 1
    for trial in range(10):
        m = 5; U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p)); b = rng.uniform(2, nu)
        net = Net(U, W, b); s = net.s; h = abs(net.margin()) + 1.0   # any fixed s-free numerator
        sq = neuron_sq(net)
        E = Evec(net)
        for l in range(1, F + 1):
            Pl = freq_proj(p, l)
            def lc(mu):
                sqm = sq + (mu - 1) * ((W @ Pl) ** 2).sum(1)
                return np.log(h) - s * np.log(np.sum(sqm ** (b / 2)))
            t = 1e-6; fd = (lc(1 + t) - lc(1 - t)) / (2 * t)
            err_tax = max(err_tax, abs(fd + nu / 2 * E[l - 1]))
        a = sq ** (nu / 2)
        def lg(ss): return np.log(h) - ss * np.log(np.sum(a ** (1 / ss)))
        t = 1e-6; fd = (lg(s + t) - lg(s - t)) / (2 * t)
        err_temp = max(err_temp, abs(fd + entropy(net)))
print(f"N3 tax derivative: max |d/dmu_l log gamma + (nu/2) E_l| = {err_tax:.2e};  temperature identity max err {err_temp:.2e}")
