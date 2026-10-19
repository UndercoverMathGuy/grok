# Relations-case model (class C, n=2, all interference on T={1,2}).
#   N_delta(a,v,th) = sum_l kappa_l(delta) (a_l + c v_l) - osc_w D_delta(w)/2,   D_delta(w)=phi(w)-phi(w-delta),
#   phi(w) = sum_l v_l cos(2 pi l w/p + th_l),  w in Z_p,   0 <= v_l <= cap a_l  (cap = 3^{s-1}).
#   ratio = min_delta N_delta / (Gamma_s(p) (sum_l a_l^{1/s})^s).   ratio>1 would beat the IF value in this model.
#   cstar = smallest payment constant c at which a configuration reaches ratio 1.
import numpy as np, torch, math
from scipy.optimize import linprog
torch.set_default_dtype(torch.float64)

class Model:
    def __init__(self, p, s, c, G, cap=None):
        self.p, self.s, self.r, self.c, self.G = p, s, 1.0 / s, c, G
        self.cap = 3 ** (s - 1) if cap is None else cap
        F = (p - 1) // 2; self.F = F
        l = np.arange(1, F + 1); d = np.arange(1, F + 1); w = np.arange(p)
        self.Knp = 1 - np.cos(2 * np.pi * np.outer(d, l) / p)            # delta x l
        self.ang = 2 * np.pi * np.outer(w, l) / p                          # w x l
        self.idxnp = (w[None, :] - d[:, None]) % p                         # delta x w  -> w-delta
        self.idx = torch.tensor(self.idxnp)

    # ---------- exact evaluation (numpy, full-length vectors) ----------
    def phi(self, v, th):
        S = np.nonzero(v)[0]
        if len(S) == 0: return np.zeros(self.p)
        return np.cos(self.ang[:, S] + th[None, S]) @ v[S]
    def parts(self, a, v, th):
        ph = self.phi(v, th); D = ph[None, :] - ph[self.idxnp]
        osc = D.max(1) - D.min(1)
        return self.Knp @ a, self.Knp @ v, osc
    def norm(self, a):
        return self.G * np.sum(a[a > 0] ** self.r) ** self.s
    def ratio(self, a, v, th, c=None):
        c = self.c if c is None else c
        cov, B, osc = self.parts(a, v, th)
        return (cov + c * B - osc / 2).min() / self.norm(a)
    def cstar(self, a, v, th):
        cov, B, osc = self.parts(a, v, th)
        cd = (self.norm(a) - cov + osc / 2) / np.maximum(B, 1e-300)
        return cd.max(), int(np.argmax(cd)) + 1
    def info(self, a, v, th, c=None):
        c = self.c if c is None else c
        cov, B, osc = self.parts(a, v, th); N = cov + c * B - osc / 2; nrm = self.norm(a); d = int(np.argmin(N))
        return dict(ratio=round(N.min() / nrm, 4), cov=round(cov[d] / nrm, 4), bonus=round(c * B[d] / nrm, 4),
                    pen=round(osc[d] / 2 / nrm, 4), covmin=round(cov.min() / nrm, 4), delta=d + 1,
                    supp=int(np.sum(a > 1e-6 * a.max())), cheap=int(np.sum(v > 1e-3 * a.max())))

    # ---------- torch pieces on a support S ----------
    def sub(self, S):
        S = np.asarray(S)
        return dict(S=S, K=torch.tensor(self.Knp[:, S]), C=torch.tensor(np.cos(self.ang[:, S])),
                    Sn=torch.tensor(np.sin(self.ang[:, S])))
    def tparts(self, sb, a, v, th, tau):
        ph = sb['C'] @ (v * torch.cos(th)) - sb['Sn'] @ (v * torch.sin(th))
        D = ph[None, :] - ph[self.idx]
        sc = a.sum()
        mx = tau * sc * torch.logsumexp(D / (tau * sc), 1); mn = -tau * sc * torch.logsumexp(-D / (tau * sc), 1)
        nrm = self.G * (a ** self.r).sum() ** self.s
        return sb['K'] @ a, sb['K'] @ v, (mx - mn), nrm

def _unpack(M, x, z, fixv):
    a = torch.exp(torch.clamp(x - x.max(), -35, 0))
    v = M.cap * a * torch.sigmoid(z) if fixv is None else fixv * a
    return a, v

def optimise(M, S, a0, v0, th0, mode='ratio', stages=(0.05, 0.02, 0.008, 0.003, 0.001), iters=300, fixv=None, c=None):
    """mode='ratio': maximise ratio at payment constant c.   mode='crit': minimise cstar.
    a0,v0,th0 are full-length; only entries in S are used.  fixv: v = fixv*a (all of S cheap at that level)."""
    c = M.c if c is None else c
    sb = M.sub(S); S = sb['S']
    aS = np.maximum(a0[S], 1e-12 * a0[S].max())
    x = torch.tensor(np.log(aS)).requires_grad_()
    t0 = np.clip(v0[S] / (M.cap * aS), 1e-6, 1 - 1e-6)
    z = torch.tensor(np.log(t0 / (1 - t0))).requires_grad_()
    th = torch.tensor(th0[S].astype(float)).requires_grad_()
    params = [x, th] if fixv is not None else [x, z, th]
    for tau in stages:
        o = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-14)
        def closure():
            o.zero_grad(); a, v = _unpack(M, x, z, fixv)
            cov, B, osc, nrm = M.tparts(sb, a, v, th, tau)
            if mode == 'ratio':
                R = (cov + c * B - osc / 2) / nrm; t2 = tau / 2
                loss = t2 * torch.logsumexp(-R / t2, 0)
            else:
                cd = (nrm - cov + osc / 2) / B; t2 = tau / 4
                loss = t2 * torch.logsumexp(cd / t2, 0)
            if not torch.isfinite(loss): raise FloatingPointError
            loss.backward(); return loss
        try: o.step(closure)
        except Exception: pass
    with torch.no_grad():
        a, v = _unpack(M, x, z, fixv)
    A = np.zeros(M.F); V = np.zeros(M.F); TH = np.zeros(M.F)
    A[S] = a.numpy(); V[S] = v.numpy(); TH[S] = th.detach().numpy()
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(V)) and np.all(np.isfinite(TH))):
        A = a0.copy(); V = v0.copy(); TH = th0.copy()
    return A, V, TH

def lp_polish(M, a, v, th, S=None, rounds=10, cuts=60, fixv=None, c=None):
    """Given phases th, optimise (a,v) on support S by reweighted LP with cutting planes over (w1,w2):
    minimise sum om_l a_l  s.t.  N_delta(a,v) >= 1 for all delta, 0<=v<=cap a (or v=fixv*a).  Reweight om=a^{r-1}."""
    c = M.c if c is None else c
    p, F = M.p, M.F
    if S is None: S = np.nonzero(a > 1e-9 * a.max())[0]
    S = np.array(S); k = len(S)
    Kd = M.Knp[:, S]; E = np.cos(M.ang[:, S] + th[None, S]); idx = M.idxnp
    def Dmat(vS):
        ph = E @ vS; return ph[None, :] - ph[idx]
    best = (M.ratio(a, v, th, c), a.copy(), v.copy())
    aS = a[S].copy(); vS = v[S].copy()
    om = (aS / aS.max() + 1e-9) ** (M.r - 1)
    cut_set = {d: set() for d in range(F)}
    def add_cuts(vS, which=None):
        D = Dmat(vS)
        for d in (range(F) if which is None else which):
            cut_set[d].add((int(np.argmax(D[d])), int(np.argmin(D[d]))))
    add_cuts(vS)
    A2 = np.hstack([-M.cap * np.eye(k), np.eye(k)])
    Aeq = np.hstack([-fixv * np.eye(k), np.eye(k)]) if fixv is not None else None
    beq = np.zeros(k) if fixv is not None else None
    for rw in range(rounds):
        for it in range(cuts):
            rows = []
            for d, cs in cut_set.items():
                for (w1, w2) in cs:
                    e = (E[w1] - E[(w1 - d - 1) % p]) - (E[w2] - E[(w2 - d - 1) % p])
                    rows.append(np.concatenate([-Kd[d], -(c * Kd[d] - 0.5 * e)]))
            A = np.array(rows); b = -np.ones(len(rows))
            cost = np.concatenate([om, np.zeros(k)])
            res = linprog(cost, A_ub=np.vstack([A, A2]), b_ub=np.concatenate([b, np.zeros(k)]), A_eq=Aeq, b_eq=beq,
                          bounds=(0, None), method='highs')
            if res.status != 0: return best
            aS, vS = res.x[:k], res.x[k:]
            D = Dmat(vS); osc = D.max(1) - D.min(1)
            N = Kd @ aS + c * (Kd @ vS) - osc / 2
            viol = N < 1 - 1e-9
            if not viol.any(): break
            add_cuts(vS, np.nonzero(viol)[0])
        aa = np.zeros(F); vv = np.zeros(F); aa[S] = aS; vv[S] = vS
        R = M.ratio(aa, vv, th, c)
        if R > best[0]: best = (R, aa.copy(), vv.copy())
        if aS.max() <= 0: break
        omn = (aS / aS.max() + 1e-9) ** (M.r - 1)
        if np.allclose(omn, om, rtol=1e-7): break
        om = omn
    return best

def crit_polish(M, a, v, th, tol=2e-3, rounds=6):
    """Given phases th, lower c* by bisection on c: at each c, reweighted cutting-plane LP over (a,v) on spt a
    (v free in [0,cap a]); success if the LP reaches ratio > 1+1e-7 with v != 0."""
    cs0, _ = M.cstar(a, v, th)
    lo, hi = 0.0, cs0; best = (cs0, a.copy(), v.copy())
    while hi - lo > tol:
        c = (lo + hi) / 2
        R, a2, v2 = lp_polish(M, best[1], np.maximum(best[2], 1e-12), th, c=c, rounds=rounds)
        if R > 1 + 1e-7 and np.any(v2 > 1e-9 * a2.max()):
            cs2, _ = M.cstar(a2, v2, th)
            if cs2 < best[0]: best = (cs2, a2, v2)
            hi = min(c, cs2)
        else: lo = c
    return best

def optimise_ws(M, S, a0, v0, th0, mode='crit', stages=(0.05, 0.02, 0.008, 0.003, 0.001), iters=300, fixv=None, c=None, nws=60):
    """working-set version: each stage optimises over the nws worst directions (plus those kept from earlier stages)."""
    c = M.c if c is None else c
    a, v, th = a0.copy(), v0.copy(), th0.copy()
    W = set()
    S = np.asarray(S)
    for tau in stages:
        cov, B, osc = M.parts(a, v, th); nrm = M.norm(a)
        crit = (nrm - cov + osc / 2) / np.maximum(B, 1e-300) if mode == 'crit' else -(cov + c * B - osc / 2) / nrm
        W |= set(np.argsort(-crit)[:nws].tolist())
        Wl = np.array(sorted(W))
        sb = M.sub(S); sb['K'] = sb['K'][Wl]
        idxW = M.idx[Wl]
        aS = np.maximum(a[S], 1e-12 * a[S].max())
        x = torch.tensor(np.log(aS)).requires_grad_()
        t0 = np.clip(v[S] / (M.cap * aS), 1e-6, 1 - 1e-6)
        z = torch.tensor(np.log(t0 / (1 - t0))).requires_grad_()
        tt = torch.tensor(th[S].astype(float)).requires_grad_()
        params = [x, tt] if fixv is not None else [x, z, tt]
        o = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
        def closure():
            o.zero_grad(); aa, vv = _unpack(M, x, z, fixv)
            ph = sb['C'] @ (vv * torch.cos(tt)) - sb['Sn'] @ (vv * torch.sin(tt))
            D = ph[Wl][:, None] * 0 + ph[None, :] - ph[idxW]
            sc = aa.sum()
            mx = tau * sc * torch.logsumexp(D / (tau * sc), 1); mn = -tau * sc * torch.logsumexp(-D / (tau * sc), 1)
            nrm = M.G * (aa ** M.r).sum() ** M.s
            cv = sb['K'] @ aa; Bv = sb['K'] @ vv; os_ = mx - mn
            if mode == 'ratio':
                R = (cv + c * Bv - os_ / 2) / nrm; t2 = tau / 2; loss = t2 * torch.logsumexp(-R / t2, 0)
            else:
                cd = (nrm - cv + os_ / 2) / Bv; t2 = tau / 4; loss = t2 * torch.logsumexp(cd / t2, 0)
            if not torch.isfinite(loss): raise FloatingPointError
            loss.backward(); return loss
        try: o.step(closure)
        except Exception: pass
        with torch.no_grad():
            aa, vv = _unpack(M, x, z, fixv)
        an = np.zeros(M.F); vn = np.zeros(M.F); thn = th.copy()
        an[S] = aa.numpy(); vn[S] = vv.numpy(); thn[S] = tt.detach().numpy()
        if np.all(np.isfinite(an)) and np.all(np.isfinite(vn)) and np.all(np.isfinite(thn)):
            a, v, th = an, vn, thn
    return a, v, th
