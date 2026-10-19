# Referee check v2: "ghost" frequencies. N_S (BRIEF_CORE) is defined by CLEAN support. A class-C network can carry
# neurons at a frequency k notin S whose atoms cancel (beta_k = 0) while beta_{k,T} != 0. Then P_k > 0, zhat is not
# in Delta_S, and the step "M(P^s) <= Gamma^{(S)} P^s" used for CD => RD_S (CC3(ii), CC4, CC6) fails.
import sys
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import vlib as V

p, n, b = 5, 2, 2.0; nu = 3; s = nu / b; r = 1 / s; B = V.Bp(p, n); pi = 3 ** (1 - s); F = 2
a = np.arange(p)
U1, W1 = V.if_block(p, n, 1, 1.0)


def ghost_pair(k, amp, phA, tau, rng=None):
    """two frequency-k neurons whose atoms cancel exactly: B has input phases shifted by tau, output phase chosen so
    that atom_B = -atom_A (same amplitudes)."""
    UA = np.array([[amp * np.cos(2 * np.pi * k * a / p + phA[t]) for t in range(n)]])
    WA = np.array([amp * np.cos(2 * np.pi * k * a / p + phA[n])])
    # atom ~ conj(what(k)) prod uhat_t(k); uhat_t(k) ~ e^{i ph_t}, what(k) ~ e^{i ph_w}; atom phase = sum ph_t - ph_w
    phB = [phA[t] + tau[t] for t in range(n)]
    phBw = phA[n] + sum(tau) + np.pi
    UB = np.array([[amp * np.cos(2 * np.pi * k * a / p + phB[t]) for t in range(n)]])
    WB = np.array([amp * np.cos(2 * np.pi * k * a / p + phBw)])
    return np.concatenate([UA, UB]), np.concatenate([WA, WB])


rng = np.random.default_rng(1)
print('=== (A) ghost networks in N_{1} at (5,2,2): IF block at 1 + ghost pair at 2 (beta_2 = 0)')
for amp in [0.3, 0.6, 1.0, 1.5]:
    UG, WG = ghost_pair(2, amp, rng.uniform(0, 2 * np.pi, 3), [0.9, -0.4])
    U = np.concatenate([U1, UG]); W = np.concatenate([W1, WG]); lab = np.array([1, 1, 1, 2, 2])
    f = V.table(U, W); beta = V.clean(f, p, n); sf = V.signflip(f, p, n)
    h = V.gaps(U, W, f).min()
    Pl = np.array([(V.sq(U, W)[lab == l] ** (b / 2)).sum() for l in (1, 2)]); P = Pl.sum()
    MPs = V.Mcov(Pl ** s, p); GS = 1 - np.cos(2 * np.pi / p)      # Gamma^{({1})} = kappa_min
    print(f'  amp={amp}: |beta_2|={abs(beta[1]):.1e}  max_T|beta_2,T|={max(abs(sf[(2, T)]) for T in [(0, 1), (1, 0), (1, 1)]):.3f}'
          f'  M(P^s)/(Gamma^(S) P^s) = {MPs / (GS * P ** s):.4f}  (>1 => CD does not imply RD_S)'
          f'  CD ratio {h / (pi * B * MPs):.4f}  RD_S ratio {h / P ** s / (pi * B * GS):.4f}')
