# Referee check v1: CC1 (chain rule/Arimoto/cost forms), CC2 (factorisation + IDENTITY translation), CC3(iii),
# CC4 any-allocation identity (mixed networks too), CC6 algebra (WF <-> bracket form, rho in [0,1], chain).
# Independent implementation (vlib), random p in {3,5,7,11}, n in {2,3,4}, b uniform in [2, nu].
import sys
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V

rng = np.random.default_rng(20261019)
err = dict(chain=0, arim=0, bounds=0, cost1=0, cost2=0, fact=0, el=0, piw=0, Esig=0, dprog=0, dint=0, pifeat=0,
           anyalloc=0, wf=0)
viol = dict(nocheap_cd=0, rho=0, chain_red=0)
cnt = 0
cases = [(3, 2), (3, 3), (5, 2), (5, 3), (7, 2), (7, 3), (11, 2), (5, 4)]
for (p, n) in cases:
    F = (p - 1) // 2; nu = n + 1
    for t in range(6):
        widths = {l: int(rng.integers(1, 5)) for l in range(1, F + 1) if l == 1 or rng.random() < 0.7}
        U, W, lab = V.rand_classC(p, n, widths, rng, scale={l: rng.uniform(0.3, 2.5) for l in widths})
        f = V.table(U, W); g = V.gaps(U, W, f); h = g.min()
        beta = V.clean(f, p, n)
        # per-neuron atoms
        atoms = np.array([V.clean(V.table(U[i:i + 1], W[i:i + 1]), p, n) for i in range(len(lab))])
        for b in [2.0, rng.uniform(2, nu), float(nu) - 1e-3]:
            s = nu / b; r = 1 / s; B = V.Bp(p, n)
            a = V.sq(U, W) ** (nu / 2); m = a / a.sum()
            feats = sorted(set(lab))
            A = np.array([a[lab == l].sum() for l in range(1, F + 1)])
            Ml = A / a.sum()
            Pl = np.array([(a[lab == l] ** r).sum() for l in range(1, F + 1)])
            K = np.array([np.exp(V.renyi(a[lab == l], r)) if A[l - 1] > 0 else 1.0 for l in range(1, F + 1)])
            zh = Pl ** s / (Pl ** s).sum()
            P = Pl.sum(); gamma = h / P ** s
            # CC1
            Z = (Ml * K ** (s - 1)).sum(); HA = np.log(Z) / (s - 1)
            err['chain'] = max(err['chain'], abs(V.renyi(m, r) - V.renyi(zh, r) - HA))
            arim = (r / (1 - r)) * np.log(sum(Ml[l - 1] * ((a[lab == l] / A[l - 1]) ** r).sum() ** (1 / r) for l in feats))
            err['arim'] = max(err['arim'], abs(arim - HA))
            Hs = [V.renyi(a[lab == l], r) for l in feats]
            if not (min(Hs) - 1e-12 <= HA <= max(Hs) + 1e-12): err['bounds'] += 1
            err['cost1'] = max(err['cost1'], abs(P ** s - (Pl ** s).sum() * np.exp(V.renyi(zh, r)) ** (s - 1)) / P ** s)
            err['cost2'] = max(err['cost2'], abs((Pl ** s).sum() - a.sum() * np.exp((s - 1) * HA)) / (Pl ** s).sum())
            # CC2 (only where M(Re beta) != 0)
            MRb = V.Mcov(beta.real, p)
            e = np.where(Pl > 0, beta.real / (B * np.where(Pl > 0, Pl, 1) ** s), 0)
            if abs(MRb) > 1e-14:
                dint = 1 - h / MRb
                Kz = np.exp(V.renyi(zh, r))
                rhs = B * (1 - dint) * V.Mcov(e * zh, p) * Kz ** (1 - s)
                err['fact'] = max(err['fact'], abs(rhs - gamma) / max(abs(gamma), 1e-300))
                piw = h / (B * V.Mcov(Pl ** s, p)); piw2 = (1 - dint) * V.Mcov(e * zh, p) / V.Mcov(zh, p)
                pif = V.Mcov(zh, p) * Kz ** (1 - s)
                err['piw'] = max(err['piw'], abs(B * pif * piw - gamma) / max(abs(gamma), 1e-300) + abs(piw - piw2) / max(abs(piw), 1e-300))
                G_S, _ = V.programme(p, s, feats) if F <= 5 else (None, None)
                if G_S is not None: err['pifeat'] = max(err['pifeat'], pif - G_S)
                # IDENTITY translations
                sig = 1 - atoms[np.arange(len(lab)), lab - 1].real / (B * a)
                Esig = (m * sig).sum()
                err['Esig'] = max(err['Esig'], abs((1 - Esig) - (e * zh).sum() * np.exp((s - 1) * HA)))
                G1 = p / (p - 1)
                dprog_def = (G1 * beta.real.sum() - MRb) / (B * G1 * a.sum())
                dint_def = (MRb - h) / (B * G1 * a.sum())
                yhat = beta.real / beta.real.sum()
                if beta.real.sum() > 0: err['dprog'] = max(err['dprog'], abs(dprog_def - (1 - Esig) * (1 - V.Mcov(yhat, p) / G1)))
                else: err['neg_sum_cases'] = err.get('neg_sum_cases', 0) + 1
                if beta.real.sum() > 0: err['dint'] = max(err['dint'], abs(dint_def - (1 - Esig) * V.Mcov(yhat, p) / G1 * dint))
                # e_l formula
                for l in feats:
                    ml = a[lab == l] / A[l - 1]
                    el = (1 - (ml * sig[lab == l]).sum()) * np.exp(-(s - 1) * V.renyi(ml, r))
                    err['el'] = max(err['el'], abs(el - e[l - 1]) / max(abs(e[l - 1]), 1e-12))
                # CC3(iii): if e_l <= pi for all l then CD (h <= pi B M(P^s)); test with pi := max_l e_l (any pi works)
                pi_t = max(e.max(), 1e-12)
                if h > 0 and h > pi_t * B * V.Mcov(Pl ** s, p) * (1 + 1e-12): viol['nocheap_cd'] += 1
            # CC6: WF algebra and rho in [0,1]; chain WF => CD_pair => CD with an arbitrary pi
            pi_t = rng.uniform(0.2, 1.0)
            Ek = beta.real - pi_t * B * Pl ** s
            Kp = V.kap(p)
            cov = Kp @ Pl ** s; d0s = np.where(cov <= cov.min() * (1 + 1e-12))[0]
            Gd = g.min(0); Rbar = Kp @ beta.real
            X = Rbar - np.minimum(Gd, Gd[::-1]); D = Rbar - 0.5 * (Gd + Gd[::-1])
            # isolated depths
            Diso = np.zeros(p - 1)
            for l in feats:
                gl = V.gaps(U[lab == l], W[lab == l]).min(0)
                Diso += Kp[:, l - 1] * beta[l - 1].real - 0.5 * (gl + gl[::-1])
            ok = Diso > 1e-12
            rho = D[ok] / Diso[ok]
            if np.any(rho < -1e-10) or np.any(rho > 1 + 1e-10): viol['rho'] += 1
            for d0 in d0s:
                if X[d0] >= (Ek * Kp[d0]).sum():      # CD_pair(d0)
                    if h > pi_t * B * cov.min() * (1 + 1e-12) + 1e-15: viol['chain_red'] += 1
            # WF bracket equivalence for a random charge D_l
            for l in feats:
                d0 = d0s[0]; Dl = rng.normal() * abs(Ek[l - 1] * Kp[d0, l - 1]) + Ek[l - 1] * Kp[d0, l - 1]
                lhs = Dl >= Ek[l - 1] * Kp[d0, l - 1]
                sig = 1 - atoms[np.arange(len(lab)), lab - 1].real / (B * a)
                ml = a[lab == l] / A[l - 1]
                dl = Dl / (B * A[l - 1] * Kp[d0, l - 1])
                rhs = (1 - (ml * sig[lab == l]).sum() - dl) * np.exp(-(s - 1) * V.renyi(ml, r)) <= pi_t
                if lhs != rhs: err['wf'] += 1
            cnt += 1

# CC4 any-allocation identity for MIXED networks
for (p, n) in [(5, 2), (7, 2), (5, 3), (3, 2)]:
    nu = n + 1; F = (p - 1) // 2
    for t in range(10):
        m_ = int(rng.integers(2, 7))
        U = rng.normal(size=(m_, n, p)); W = rng.normal(size=(m_, p))
        h = V.gaps(U, W).min()
        for b in [2.0, rng.uniform(2, nu)]:
            s = nu / b; r = 1 / s; B = V.Bp(p, n); P = (V.sq(U, W) ** (b / 2)).sum()
            Pt = rng.dirichlet(np.ones(F)) * P
            zt = Pt ** s / (Pt ** s).sum()
            rhs = B * V.Mcov(zt, p) * np.exp(V.renyi(zt, r)) ** (1 - s) * h / (B * V.Mcov(Pt ** s, p))
            err['anyalloc'] = max(err['anyalloc'], abs(rhs - h / P ** s) / abs(h / P ** s))
print('cases:', cnt, ' (d_prog/d_int translation checked only where sum Re beta > 0; it FAILS when sum Re beta < 0, possible only for h < 0)')
for k, v in err.items():
    print(f'  {k:10s} {v:.3e}')
print('violations:', viol)
