"""audit_chain.py -- every link of the THEORY_nsum chain on random and adversarial networks (exact, float64).
 L1  margin(z) <= margin(g)                       (junk lemma)
 L2  margin(g) <= min_D sum_k 2 a_k g_k(D)        (phase lemma), a_k = |g_hat(k)|
 L3  g_hat(k) = kappa sum_i prod_j e_hat_ij(k) conj w_hat_i(k)   (assembly; kappa = n! for x^n, 1 for prod)
 B_q sum_k a_k^{q/nu} <= c_n^{q/nu} (2p)^{-q/2} P_q   claimed for 2 <= q <= nu; shown to FAIL for q < 2 (one neuron, K freqs)
Networks: gaussian (all frequencies + DC), band-limited, tied operands, single multi-frequency balanced neuron.
uv run python heracles/audit_chain.py"""
import math, numpy as np, torch
from audit_common import rows, logits, margin, clean, Gmat, cn, Pq

torch.manual_seed(0)


def check(E, W, p, n, act, tag, qs):
    nu = n + 1; ops, y = rows(p, n); nf = (p - 1) // 2
    z = logits(E, W, ops, act); g = clean(z, y, p)
    mz = float(margin(z, y).min()); mg = float((g[0] - g[1:]).min())
    gh = np.fft.fft(g.numpy()) / p; a = np.abs(gh[1:nf + 1])
    ph = float((2 * a @ Gmat(p)).min())
    Eh = np.fft.fft(E.numpy(), axis=2) / p; Wh = np.fft.fft(W.numpy(), axis=1) / p
    kap = math.factorial(n) if act == "pow" else 1.0
    asm = kap * (np.prod(Eh, axis=0) * np.conj(Wh)).sum(0)
    l3 = float(np.abs(asm - gh)[1:].max() / (np.abs(gh)[1:].max() + 1e-300))
    out = [f"{tag:28s} L1 {mz:+.3e}<={mg:+.3e} {'ok' if mz <= mg + 1e-9 else 'VIOLATED'} | L2 <= {ph:.3e} {'ok' if mg <= ph + 1e-9 else 'VIOLATED'} | L3 relerr {l3:.1e}"]
    for q in qs:
        lhs = float((a ** (q / nu)).sum()); rhs = cn(n, act) ** (q / nu) * (2 * p) ** (-q / 2) * float(Pq(E, W, q))
        out.append(f"q={q:.2f}: {lhs / rhs:.3f}")
    print(" | B_q lhs/rhs ".join(out[:1]) + " | B_q lhs/rhs  " + "  ".join(out[1:]))
    return mz <= mg + 1e-9 and mg <= ph + 1e-9


def balanced_multifreq(p, n, K, m=1):
    a = torch.arange(p) * 2 * math.pi / p
    E = torch.zeros(n, m, p); W = torch.zeros(m, p)
    for k in range(1, K + 1):
        for j in range(n): E[j, 0] += torch.cos(k * a)
        W[0] += torch.cos(k * a)
    return E, W


if __name__ == "__main__":
    ok = True
    for (p, n) in ((11, 2), (13, 2), (7, 3), (5, 4)):
        for act in ("pow", "prod"):
            nu = n + 1; qs = (1.0, 1.5, 2.0, (2 + nu) / 2, float(nu))
            for t in range(3):
                E = torch.randn(n, 6, p) + 0.5; W = torch.randn(6, p) + 0.3
                ok &= check(E, W, p, n, act, f"p{p} n{n} {act} gauss+DC #{t}", qs)
            E = torch.randn(1, 6, p).repeat(n, 1, 1); W = torch.randn(6, p)
            ok &= check(E, W, p, n, act, f"p{p} n{n} {act} TIED operands", qs)
            E, W = balanced_multifreq(p, n, (p - 1) // 2)
            ok &= check(E, W, p, n, act, f"p{p} n{n} {act} 1 neuron, all freqs", qs)
    print("all L1/L2 ok:", ok)
