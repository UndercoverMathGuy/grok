# v6: the alignment ledger and the failure of the single-feature subsidy inequality (SSI).
#   h = h_prof + AG  (Theorem R: h_prof = min over per-feature alignments),
#   RD  <=>  AG <= Ineff + PS,   Ineff = pi Gamma (sum Q)^s - pi M(Q^s) >= 0,  PS = pi M(Q^s) - h_prof.
#   CD  <=>  AG <= PS.
# Telescoping (floors replace features one at a time):  h_k = min_{x,d}[ sum_{l<=k} pi Q_l^s kappa_l(d)
#   + sum_{l>k} Re(Phi_l c_l(d)) ];  SSI at step k  <=>  h_{k-1} <= h_k;  h_F = pi M(Q^s).
import numpy as np
from fmlib import *
from cjlib import gamma_lp, gamma_exact

out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

def legendre(p):
    x = np.arange(p); qr = set((x[1:] ** 2) % p)
    return np.array([0] + [1 if t in qr else -1 for t in range(1, p)])

def chihat(p, l):
    return np.sum(legendre(p) * np.exp(-1j * TAU * l * np.arange(p) / p)) / p

def ledger_oned(p, n, s, Gam_lb):
    """single-set network: work on w in Z_p.  G_l(w,d) = beta k_l(d) + Re(c rho^{lw} c_l(d))."""
    F = (p - 1) // 2; r = 1 / s; pi = price_n2(s) if n == 2 else (n + 1) ** (1 - s)
    w = np.arange(p)[:, None]; d = np.arange(1, p)[None, :]
    Gs = []
    for l in range(1, F + 1):
        c = 2 * chihat(p, l); beta = abs(c)
        Gs.append(beta * kap(p, l, d) + np.real(c * np.exp(1j * TAU * l * w / p) * (1 - np.exp(-1j * TAU * l * d / p))))
    Gs = np.array(Gs)                                   # (F, p, p-1)
    beta = 2 / np.sqrt(p); Ql = n ** (1 - r) * beta ** r; Q = np.full(F, Ql)
    h = Gs.sum(0).min()
    hprof = Gs.min(1).sum(0).min()
    Ms = (kappa_mat(p) @ (Q ** s)).min()
    PS = pi * Ms - hprof; AG = h - hprof
    Ineff_lb = pi * Gam_lb * Q.sum() ** s - pi * Ms        # lower bound (Gamma >= Gamma_lb)
    # telescoping chain
    floors = pi * Q[:, None] ** s * kappa_mat(p).T     # (F, p-1)
    chain = [h]
    acc_floor = np.zeros(p - 1); acc_field = Gs.sum(0)
    first_fail = None
    for k in range(F):
        acc_field = acc_field - Gs[k]; acc_floor = acc_floor + floors[k]
        hk = (acc_field + acc_floor[None, :]).min()
        if first_fail is None and hk < chain[-1] - 1e-12: first_fail = k + 1
        chain.append(hk)
    return dict(h=h, hprof=hprof, AG=AG, PS=PS, Ineff_lb=Ineff_lb, CD=h / (pi * Ms), first_fail=first_fail,
                hF=chain[-1], piM=pi * Ms)

for p in [127, 199, 401]:
    s = 1.5
    G_lb, _ = gamma_lp(p, s, restarts=8, seed=1)
    L = ledger_oned(p, 2, s, G_lb)
    say(f'Legendre p={p} n=2 s={s}: h={L["h"]:.4f} h_prof={L["hprof"]:.4f} AG={L["AG"]:.4f} PS={L["PS"]:.4f} '
        f'Ineff>={L["Ineff_lb"]:.4f}  => AG-PS={L["AG"]-L["PS"]:+.4f} (CD ratio {L["CD"]:.4f}),  '
        f'AG <= Ineff+PS: {L["AG"] <= L["Ineff_lb"] + L["PS"]};  SSI first fails at step {L["first_fail"]}; '
        f'h_F={L["hF"]:.4f} = pi M(Q^s)={L["piM"]:.4f}')

# random class-C networks at small p: identities and signs
rng = np.random.default_rng(5); worst = 0; n_ok = 0; n_tot = 0
for (p, n) in [(5, 2), (7, 2), (11, 2), (7, 3)]:
    F = (p - 1) // 2
    for trial in range(30):
        s = rng.choice([1.2, 1.5]) if n == 2 else rng.choice([1.3, 2.0])
        S = list(range(1, F + 1))
        feats = {}
        for l in S:
            C0, xi0 = if_atoms(n, rng.uniform(0.3, 2))
            m = rng.integers(1, 3)
            C = np.concatenate([C0, 0.2 * (rng.normal(size=m) + 1j * rng.normal(size=m))])
            xi = np.concatenate([xi0, np.exp(1j * rng.uniform(0, TAU, (m, n)))])
            feats[l] = (C, xi)
        net = CNet(p, n, feats); r = 1 / s
        pi = price_n2(s) if n == 2 else (n + 1) ** (1 - s)
        Q = np.array([net.Q(r)[l] for l in S])
        h = net.h(); hp = net.hprof(); Ms = M_alloc(p, Q ** s)
        G, _ = gamma_exact(p, s)
        AG = h - hp; PS = pi * Ms - hp; Ineff = pi * G * Q.sum() ** s - pi * Ms
        worst = max(worst, abs((AG + hp) - h)); n_tot += 1
        n_ok += (AG >= -1e-12) and (Ineff >= -1e-12) and ((AG <= Ineff + PS) == (h <= pi * G * Q.sum() ** s))
say(f'random near-IF class-C networks: identity error {worst:.1e}; sign/equivalence checks passed {n_ok}/{n_tot}')
open('v6_ledger.out', 'w').write('\n'.join(out) + '\n')
