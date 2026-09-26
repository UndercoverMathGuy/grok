# Is per-offset domination too strong? Maximise the SINGLE-offset CAoff ratio at one offset class D0:
#   r1 = m^u_{D0} (p^3/2) / ( max_k cov_k(D0) * budget_diag^{3/2} )
# over pure networks with n_per neurons per frequency on a support S (free phases and amplitudes).
# r1 > 1 would falsify CAoff (any variant with the diagonal budget) while saying nothing about T.
import sys, math, numpy as np, torch
sys.path.insert(0, '..')
from common import cover_matrix
torch.set_default_dtype(torch.float64)
p, D0, L, n_per, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
mode = sys.argv[6] if len(sys.argv) > 6 else 'ap'
rng = np.random.default_rng(seed); h = (p - 1) // 2
inv = pow(D0, -1, p)
if mode == 'ap':     # frequencies whose coverage at D0 is near maximal: k D0 = +-(h - j)
    S = sorted({min((inv * (h - j)) % p, p - (inv * (h - j)) % p) for j in range(L)})
else:                # random support
    S = sorted(rng.choice(np.arange(1, h + 1), L, replace=False).tolist())
K = len(S); covD0 = np.array([1 - math.cos(2 * math.pi * k * D0 / p) for k in S]); cmax = max(1 - math.cos(2 * math.pi * k * D0 / p) for k in range(1, h + 1))
x = torch.arange(p, dtype=torch.float64)
ks = torch.tensor(S, dtype=torch.float64).repeat_interleave(n_per)
ph = torch.tensor(rng.uniform(0, 2 * np.pi, (3, K * n_per)), requires_grad=True)
la = torch.tensor(rng.normal(0, 0.3, K * n_per), requires_grad=True)
opt = torch.optim.Adam([ph, la], lr=0.02)
xi = torch.arange(p); Sg = (xi[:, None] + xi[None, :]) % p
def evaluate(tau):
    amp = torch.exp(la)
    E1 = amp[:, None] * torch.cos(2 * math.pi * ks[:, None] * x[None, :] / p + ph[0][:, None])
    E2 = amp[:, None] * torch.cos(2 * math.pi * ks[:, None] * x[None, :] / p + ph[1][:, None])
    W = amp[:, None] * torch.cos(2 * math.pi * ks[:, None] * x[None, :] / p + ph[2][:, None])
    Z = torch.einsum('ia,ib,ic->abc', E1, E2, W)
    corr = Z[xi[:, None], xi[None, :], Sg]
    vals = torch.cat([(corr - Z[xi[:, None], xi[None, :], (Sg - D0) % p]).reshape(-1), (corr - Z[xi[:, None], xi[None, :], (Sg + D0) % p]).reshape(-1)])
    m = -tau * torch.logsumexp(-vals / tau, 0) if tau > 0 else vals.min()
    F1, F2, FW = [torch.fft.fft(T.to(torch.complex128), dim=-1) for T in (E1, E2, W)]
    cost = torch.zeros(()); 
    for kk in S:
        c = (F1[:, kk] * F2[:, kk] * FW[:, kk]).abs()
        cost = cost + ((c + 1e-18) ** (2 / 3)).sum()     # per-neuron |c_ik|^{2/3}; neurons are single-frequency
    bud = cost / 3 ** (1 / 3)
    return m * p ** 3 / 2 / (cmax * bud ** 1.5)
best = -9
for it in range(3000):
    tau = 0.02 * (1 - it / 3000) + 1e-4
    r = evaluate(tau * float(torch.exp(la).max() ** 3))
    opt.zero_grad(); (-r).backward(); opt.step()
    if it % 100 == 0 or it == 2999:
        with torch.no_grad():
            best = max(best, float(evaluate(0)))
print(f"p={p} D0={D0} mode={mode} S={S} n_per={n_per} seed={seed}: best single-offset ratio r1 = {best:.4f}  (frequency coverage at D0: {np.round(covD0, 3)})", flush=True)
