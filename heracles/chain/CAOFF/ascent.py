# Adversarial ascent on the CAoff ratio (need(m)/budget)^{3/2} for unrestricted networks (mixing neurons, DC, junk).
# usage: ascent.py p N variant(u|s|f) budget(diag|P) seedtype(blockmix|lmax|random|single) seed steps [sigma]
import sys, math, numpy as np, torch
sys.path.insert(0, '..')
from common import programme, block_net, cover_matrix
from lib import need, evaluate, unpack
torch.set_default_dtype(torch.float64)
p, N, var, bud, st, seed, steps = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7])
sigma = float(sys.argv[8]) if len(sys.argv) > 8 else 0.3
xmin = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
jmin = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
rng = np.random.default_rng(seed); h = (p - 1) // 2
Gam, ystar = programme(p, 1.5)
from common import gamma_sym
gsym = gamma_sym(p, Gam)

def init():
    v = np.zeros((3, N, p))
    if st in ('blockmix', 'single'):
        y = ystar if st == 'blockmix' else np.eye(h)[rng.integers(h)] * 0.5
        blk = block_net(p, y)
        for i, (e1, e2, w) in enumerate(blk[:N]):
            v[0, i], v[1, i], v[2, i] = e1, e2, w
        v += sigma * rng.normal(size=v.shape) * np.abs(v).max()
    elif st == 'lmax':
        src = {5: ('../../claude0923/endpoint_p5_s3.npy', 9), 7: ('../../claude0923/endpoint_p7_s9.npy', 12)}[p]
        u = np.load(src[0]).reshape(3, src[1], p); n0 = min(N, src[1])
        v[:, :n0] = u[:, :n0]
        v += sigma * rng.normal(size=v.shape) * np.abs(v).max()
    else:
        v = rng.normal(size=v.shape)
    return v

x = torch.arange(p); S_ = (x[:, None] + x[None, :]) % p
i2 = pow(2, -1, p); A = cover_matrix(p)
def margins_t(th, tau):
    E1, E2, W = th[0], th[1], th[2]
    Z = torch.einsum('ia,ib,ic->abc', E1, E2, W)
    corr = Z[x[:, None], x[None, :], S_]
    ms = []
    for D in range(1, h + 1):
        gD = corr - Z[x[:, None], x[None, :], (S_ - D) % p]
        gm = corr - Z[x[:, None], x[None, :], (S_ + D) % p]
        if var == 'u':
            vals = torch.cat([gD.reshape(-1), gm.reshape(-1)])
        elif var == 's':
            vals = (0.5 * (gD + gm)).reshape(-1)
        else:
            sh = (D * i2) % p
            vals = (0.5 * (gD[(x[:, None] + sh) % p, x[None, :]] + gm[(x[:, None] - sh) % p, x[None, :]])).reshape(-1)
        ms.append(-tau * torch.logsumexp(-vals / tau, 0) + tau * math.log(1))  # soft-min (upper-ish bias removed by exact eval)
    return torch.stack(ms)

_a = torch.arange(p); _f = torch.minimum(_a, p - _a)
_same = (_f[:, None, None] == _f[None, :, None]) & (_f[None, :, None] == _f[None, None, :]) & (_f[None, None, :] > 0)
_clean = ((_a[:, None, None] + _a[None, None, :]) % p == 0) & ((_a[None, :, None] + _a[None, None, :]) % p == 0) & (_a[None, None, :] > 0)
def shares_t(th):
    Z = torch.einsum('ia,ib,ic->abc', th[0], th[1], th[2]); Z = Z - Z.mean(2, keepdim=True)
    P2 = torch.fft.fftn(Z).abs() ** 2; P2 = P2 * (_a[None, None, :] > 0)
    tot = P2.sum()
    return P2[~_same].sum() / tot, P2[_same & ~_clean].sum() / tot
def budget_t(th):
    if bud == 'P':
        return (p / 6) * (th ** 2).sum() / 3 ** (1 / 3)
    F = torch.fft.fft(th.to(torch.complex128), dim=-1)          # (3, N, p)
    c = (F[0, :, 1:h + 1] * F[1, :, 1:h + 1] * F[2, :, 1:h + 1]).abs()
    return ((c + 1e-18) ** (2 / 3)).sum() / 3 ** (1 / 3)

th = torch.tensor(init(), requires_grad=True)
opt = torch.optim.Adam([th], lr=0.01)
best = (-1, None)
for it in range(steps):
    with torch.no_grad():
        th /= math.sqrt((th ** 2).sum().item())
    tau = (0.05 * (1 - it / steps) + 0.002) * gsym
    m = margins_t(th, tau); M = m * p ** 3 / 2
    nd, arg = need(p, M.detach().numpy(), return_vertex=True)
    if arg is None:
        loss = -M.max()                                         # all margins <= 0: push the largest up first
    else:
        Sv, Tv, _ = arg
        Minv = torch.tensor(np.linalg.inv(A[np.ix_(Tv, Sv)]))
        yS = Minv @ M[torch.tensor(Tv)]
        ndt = (yS.clamp_min(1e-15) ** (2 / 3)).sum()
        loss = -(ndt / budget_t(th)) ** 1.5
    if xmin > 0 or jmin > 0:
        xs, js = shares_t(th)
        loss = loss + 20 * (torch.relu(xmin - xs) + torch.relu(jmin - js))
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 50 == 0 or it == steps - 1:
        r = evaluate(p, unpack(th.detach().numpy(), N, p))
        key = f'{var}_{bud}'
        if xmin > 0 or jmin > 0:
            xs, js = [float(t) for t in shares_t(th.detach())]
            r['xs'], r['js'] = xs, js
            if xs < 0.9 * xmin or js < 0.9 * jmin:
                continue
        if np.isfinite(r[key]) and r[key] > best[0]:
            best = (r[key], th.detach().numpy().copy(), r)
if best[1] is not None:
    np.save(f'asc_{p}_{var}_{bud}_{st}_s{seed}_x{xmin}_j{jmin}.npy', best[1])
    r = best[2]
    print(f"p={p} {var}_{bud} {st} s{seed}: best {best[0]:.6f} | T={r['T']:.4f} u_diag={r['u_diag']:.4f} s_diag={r['s_diag']:.4f} "
          f"f_diag={r['f_diag']:.4f} u_P={r['u_P']:.4f} s_P={r['s_P']:.4f} f_P={r['f_P']:.4f} xs={r.get('xs', -1):.3f} js={r.get('js', -1):.3f} sig={sigma} xmin={xmin} jmin={jmin}", flush=True)
else:
    print(f"p={p} {var}_{bud} {st} s{seed}: no positive-margin point found", flush=True)
