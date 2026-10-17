# Phase-free relaxation of the relations model: osc_w D_delta / 2  ->  lam * sqrt(sum_l v_l^2 kappa_l(delta))
# (lam = 1 is the Bhatia-Davis/Parseval floor; any phases give osc/2 >= sqrt(E D^2)).
# c*_lam(a,v) = max_delta (Gamma ||a|| - (K a)_delta + lam sqrt((K v^2)_delta)) / (K v)_delta ; minimise over a, 0<=v<=cap a.
# Also: sup ratio at payment constant c with all-cheap allowed:  max min_delta [(Ka)+c(Kv)-lam sqrt(K v^2)] / (Gamma ||a||).
import numpy as np, torch, sys, math, json
from gget import gamma_table
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
s = 1.5; r = 1 / s; cap = 3 ** 0.5

def run(p, G, lam, seeds, mode='crit', c=None, stages=(0.03, 0.01, 0.003, 0.001, 3e-4), iters=400):
    F = (p - 1) // 2; l = np.arange(1, F + 1)
    K = torch.tensor(1 - np.cos(2 * np.pi * np.outer(l, l) / p))
    best = (np.inf if mode == 'crit' else -np.inf, None)
    for (name, A0) in seeds:
        S = np.nonzero(A0)[0]; KS = K[:, S]
        x = torch.tensor(np.log(A0[S])).requires_grad_(); z = torch.full((len(S),), 2.0).requires_grad_()
        def vals(x, z):
            a = torch.exp(torch.clamp(x - x.max(), -35, 0)); v = cap * a * torch.sigmoid(z)
            nrm = G * (a ** r).sum() ** s
            return KS @ a, KS @ v, torch.sqrt(KS @ (v * v)), nrm
        for tau in stages:
            o = torch.optim.LBFGS([x, z], lr=1, max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-13, tolerance_change=1e-15)
            def closure():
                o.zero_grad(); cv, B, R2, nrm = vals(x, z)
                if mode == 'crit':
                    q = (nrm - cv + lam * R2) / B; loss = tau * torch.logsumexp(q / tau, 0)
                else:
                    q = (cv + c * B - lam * R2) / nrm; loss = tau * torch.logsumexp(-q / tau, 0)
                loss.backward(); return loss
            try: o.step(closure)
            except Exception: pass
        with torch.no_grad():
            cv, B, R2, nrm = vals(x, z)
            val = ((nrm - cv + lam * R2) / B).max().item() if mode == 'crit' else ((cv + c * B - lam * R2) / nrm).min().item()
            a = torch.exp(torch.clamp(x - x.max(), -35, 0)).numpy(); v = (cap * torch.sigmoid(z)).numpy() * a
        better = val < best[0] if mode == 'crit' else val > best[0]
        if better: best = (val, (name, len(S), int(np.sum(a > 1e-3 * a.max())), int(np.sum(v > 1e-3 * a.max()))))
    return best

if __name__ == '__main__':
    GT = gamma_table(); ps = [int(a) for a in sys.argv[1].split(',')]; lams = [float(a) for a in sys.argv[2].split(',')]
    out = {}
    for p in ps:
        if p not in GT: print('no Gamma for', p); continue
        G = GT[p]['Gamma']; F = (p - 1) // 2; L = math.log(p); rng = np.random.default_rng(p)
        Sy = np.array(GT[p]['S']) - 1; y = np.array(GT[p]['y'])
        seeds = [('full', np.ones(F))]
        for m in [2, 4, 8, 16, 32]:
            K = min(F, int(round(m * L)))
            for t in range(2):
                A = np.zeros(F); A[rng.choice(F, K, replace=False)] = 1; seeds.append((f'rand{m}L', A))
            A = np.zeros(F); A[Sy] = y; rest = np.setdiff1d(np.arange(F), Sy)
            A[rng.choice(rest, min(len(rest), K), replace=False)] = y.mean() * 0.7; seeds.append((f'ystar+{m}L', A))
        for lam in lams:
            cst = run(p, G, lam, seeds, 'crit')
            print(f'p={p:5d} Gamma={G:.4f} lam={lam:.3f}: c*_lam={cst[0]:.4f} {cst[1]}', flush=True)
            out[f'{p}_{lam}'] = dict(cstar=cst[0])
    json.dump(out, open(f'parseval_{sys.argv[1].replace(",", "_")}.json', 'w'))
