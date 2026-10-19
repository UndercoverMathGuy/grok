# KKT data of numerically found local maximisers (IF and non-IF) at small (p, n, b).
# For each polished local maximiser: gamma/gamma_IF, entropy H vs H_dec(IF) = log nu + min H(z*),
# KKT residual, identity check, critical values t_i, tau(q) (NNLS measure), x-uniformity of q,
# cheapness |beta_l|/(nu^{1-s} Q_l^s), interference max|beta_{l,T}|/|beta_l|.
import numpy as np, sys, time
from klib import *

p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 12
seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
nu = n + 1; s = nu / b; r = 1 / s
G, ys, zs = programme_exact(p, s)
Hdec = np.log(nu) + min(entropy(z) for z in zs)
gIF = gamma_if(p, n, b, G)
print(f'# (p,n,b)=({p},{n},{b}) s={s:.4f} Gamma={G:.6f} H_dec(IF)={Hdec:.5f} K*={[int((y>0).sum()) for y in ys]}')
print('# start width | gamma/gIF | H | H-Hdec | KKTres | idDiff | maxCritRes | tau/t_max | TVunif | cheap_max | interf_max | features')
rows = []
for k in range(starts):
    rng = np.random.default_rng(seed0 + k)
    m = int(rng.integers(nu - 1, 3 * nu + 1))
    U = rng.standard_normal((m, n, p)); W = rng.standard_normal((m, p))
    net = Net(U, W, b)
    t0 = time.time()
    try:
        pol = prune(balance(optimise(net)))
        pol = optimise(pol, taus=(1e-5, 3e-6, 1e-6, 3e-7), iters=5000)
        pol = prune(pol, 1e-10)
    except Exception as e:
        print('fail', e); continue
    gr = pol.gamma() / gIF
    if gr <= 0:
        print(f'{k:3d} {m:3d}  gamma<=0'); continue
    act, q, res, h = kkt_measure(pol, rtol=1e-6)
    Q = q_full(pol, act, q)
    t, cres, M = critical_values(pol, Q)
    w = pol.shares(); H = pol.H()
    idd = np.log(nu ** (nu / 2) * pol.gamma()) - (np.sum(w * np.log(np.abs(t))) - (s - 1) * H)
    tau, _ = injective_norm(M, starts=30, iters=150)
    mu = Q.sum(0); mu = mu / mu.sum()
    tv = 0.5 * np.abs(Q - mu[None, :] / pol.N).sum()
    beta = pol.clean(); at = pol.atoms(); Ql = (np.abs(at) ** r).sum(0)
    feats = np.where(np.abs(beta) > 1e-6 * np.abs(beta).max())[0] + 1
    cheap = np.max(np.abs(beta) / np.maximum(nu ** (1 - s) * Ql ** s, 1e-300))
    sf = pol.signflip()
    interf = max(abs(v) for (j, T), v in sf.items() if any(T)) / np.abs(beta).max()
    print(f'{k:3d} {m:3d} {pol.m:3d} | {gr:.6f} | {H:.4f} | {H-Hdec:+.4f} | {res:.1e} | {idd:+.1e} | {cres.max():.1e} | {tau/t.max():.4f} | {tv:.3f} | {cheap:.4f} | {interf:.2e} | {list(feats)}  ({time.time()-t0:.0f}s)', flush=True)
    rows.append(dict(k=k, gr=gr, H=H, Hdec=Hdec, res=res, idd=idd, tau=tau, t=t, w=w, tv=tv, cheap=cheap, interf=interf, feats=feats))
    np.savez(f'lm_p{p}_n{n}_b{b}_k{seed0+k}.npz', U=pol.U, W=pol.W, b=b)
