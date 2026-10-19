# C_ccfactor c8: sanity checks of the remaining PROVED statements.
# (a) factorised identity  gamma = B_p (1 - delta_int) M(e o zhat) K_r(zhat)^{1-s}, per-feature efficiency identity
#     e_l = [1 - E_{m^(l)} sigma] exp(-(s-1) H_r(m^(l))), and the Arimoto identification of the chain-rule term;
# (b) sum_{l in F} Psi^(l) = (p/2)(I + J) on every offset set; equal-energy profiles make the pair cone moment vacuous;
# (c) the construction behind CD => DU_min (gaps add; binding pair; limit of the CD ratio);
# (d) is the E_kkt (5,2,2) network k4 (canonical CD violated) a local maximiser of gamma?
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
import ccf, cclib as C
import lib as GL
E = C.E
rng = np.random.default_rng(23)

print('=== (a) gamma = B_p (1-delta_int) M(e o zhat) K_r(zhat)^{1-s}; e_l identity; Arimoto')
w = [0, 0, 0]
for (p, n) in [(5, 2), (7, 2), (11, 2), (5, 3), (7, 3), (13, 2), (3, 2)]:
    F = (p - 1) // 2
    for t in range(8):
        widths = {l: int(rng.integers(1, 5)) for l in range(1, F + 1) if l == 1 or rng.random() < 0.8}
        net = C.rand_classC(p, n, widths, rng, scale={l: rng.uniform(0.4, 2.0) for l in widths})
        ok, lab = C.neuron_freq(net)
        for b in [2.0, 2.4, n + 1 - 0.2]:
            s = (n + 1) / b; r = 1 / s; B = E.Bp(p, n)
            a = net.mass(); h = net.margin(); beta = np.real(net.clean())
            feats = sorted(set(lab)); Pl = np.array([(a[lab == l] ** r).sum() if l in feats else 0 for l in range(1, F + 1)])
            zh = Pl ** s / (Pl ** s).sum(); e = np.where(Pl > 0, beta / (B * np.where(Pl > 0, Pl, 1) ** s), 0)
            Mv = lambda v: (E.kappa(p) @ v).min()
            dint = 1 - h / Mv(beta)
            Kr = np.exp(C.renyi(zh[zh > 0], r))
            pred = B * (1 - dint) * Mv(e * zh) * Kr ** (1 - s)
            w[0] = max(w[0], abs(pred - net.gamma(s)) / abs(net.gamma(s)))
            at, _ = net.atoms(); sig = 1 - np.real(at).sum(1) / (B * a)
            for l in feats:
                ml = a[lab == l] / a[lab == l].sum()
                el = (1 - (ml * sig[lab == l]).sum()) * np.exp(-(s - 1) * C.renyi(ml, r))
                w[1] = max(w[1], abs(el - e[l - 1]) / max(abs(e[l - 1]), 1e-300))
            # Arimoto conditional Renyi entropy of order r of neuron given feature
            m = a / a.sum(); Ml = np.array([m[lab == l].sum() for l in feats])
            arim = (r / (1 - r)) * np.log(sum(Ml[k] * ((m[lab == l] / Ml[k]) ** r).sum() ** (1 / r) for k, l in enumerate(feats)))
            w[2] = max(w[2], abs(C.renyi(m, r) - C.renyi(zh[zh > 0], r) - arim))
print(f'  rel. err factorised identity {w[0]:.2e}; per-feature efficiency identity {w[1]:.2e}; H_r(m) - H_r(zhat) - H^A_r {w[2]:.2e}')

print('=== (b) sum_l Psi^(l) = (p/2)(I+J); equal energies -> pair cone moment vacuous')
err = 0; worstfrac = -9
for p in [5, 7, 11, 13, 31, 101]:
    F = (p - 1) // 2
    for t in range(5):
        offs = rng.choice(np.arange(1, p), size=min(p - 1, int(rng.integers(2, 8))), replace=False)
        M = sum(np.real(np.outer(1 - np.exp(-2j * np.pi * l * offs / p), np.conj(1 - np.exp(-2j * np.pi * l * offs / p))))
                for l in range(1, F + 1))
        err = max(err, np.abs(M - (p / 2) * (np.eye(len(offs)) + 1)).max())
    for d in range(1, p):
        l = np.arange(1, F + 1); kap = 1 - np.cos(2 * np.pi * l * d / p)
        worstfrac = max(worstfrac, (kap * np.cos(2 * np.pi * l * d / p)).sum())       # V_l = 1 for all l
print(f'  max |sum_l Psi^(l) - (p/2)(I+J)| = {err:.2e};  max_d sum_l kappa_l cos(phi_l) (equal V) = {worstfrac:.4f} (< 0: vacuous)')

print('=== (c) CD => DU_min construction: small feature theta_1 at frequency j + optimal feature at k = j l^{-1}')
for (p, n, b) in [(7, 2, 2.0), (11, 2, 2.5), (7, 3, 2.0)]:
    nu = n + 1; s = nu / b; pi = nu ** (1 - s); B = E.Bp(p, n); F = (p - 1) // 2
    # theta_1: perturbed IF single feature at frequency 1 (non-IF)
    y = np.zeros(F); y[0] = 1.0
    net1, lab1 = ccf.perturbed_if(p, n, y, rng, 0.3, extra=1)
    g1 = net1.gaps(); P1 = net1.cost(b); prof = g1.min(0) / (B * P1 ** s)          # normalised profile m(l), l = delta
    for (j, l) in [(1, 2), (2, 3)]:
        if l >= p - 1: continue
        k = (j * pow(l, -1, p)) % p; k = min(k, p - k)
        if k == j: continue
        d0 = (pow(j, -1, p) * l) % p                                                # offset with j d0 = l
        for eps in [0.3, 0.1, 0.03]:
            # relabel theta_1 to frequency j with cost eps; optimal block at k with cost 1
            U1, W1 = C.relabel_copy(net1.U, net1.W, j, p)
            c1 = (eps / P1) ** (1 / b); U1 = U1 * c1; W1 = W1 * c1
            Uk, Wk = E.if_block(p, n, k, 1.0); ck = (1.0 / E.Net(Uk, Wk).cost(b)) ** (1 / b); Uk *= ck; Wk *= ck
            net = E.Net(np.concatenate([U1, Uk]), np.concatenate([W1, Wk]))
            h = net.margin(); Pl = np.zeros(F); Pl[j - 1] = eps; Pl[k - 1] = 1.0
            Mps = (E.kappa(p) @ Pl ** s).min(); cd = h / (pi * B * Mps)
            gk = E.Net(Uk, Wk).gaps().min(0); bound = (B * eps ** s * prof[(np.arange(1, p) * j) % p - 1] + gk).min()
            pred_lim = (pi * (1 - np.cos(2 * np.pi / p)) + eps ** s * min(prof[l - 1], prof[p - l - 1])) / \
                       (pi * (1 - np.cos(2 * np.pi / p)) + eps ** s * pi * (1 - np.cos(2 * np.pi * l / p)))
            print(f'  ({p},{n},{b}) j={j} l={l} k={k} eps={eps}: h - sum-of-profiles bound = {h - bound:.2e} (>=0);'
                  f' CD ratio {cd:.6f}; pair-term ratio {pred_lim:.6f}; min(m(l),m(-l))/(pi kappa) = '
                  f'{min(prof[l - 1], prof[p - l - 1]) / (pi * (1 - np.cos(2 * np.pi * l / p))):.4f}')

print('=== (d) E_kkt lm_p5_n2_b2.0_k4 (canonical CD_prop = 1.0146): re-optimise from perturbations')
d = np.load('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p5_n2_b2.0_k4.npz')
U0, W0 = d['U'], d['W']; base = GL.Net(U0, W0, 2.0).gamma()
for t in range(4):
    sc = [1e-3, 1e-2, 3e-2, 1e-1][t]
    U = U0 + sc * rng.normal(size=U0.shape) * np.abs(U0).mean(); W = W0 + sc * rng.normal(size=W0.shape) * np.abs(W0).mean()
    net = GL.optimise(U, W, 2.0)
    print(f'  perturbation {sc:.0e}: gamma after re-optimisation / gamma(k4) = {net.gamma() / base:.6f}')
