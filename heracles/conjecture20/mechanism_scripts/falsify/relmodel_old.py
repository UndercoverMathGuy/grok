# Relations-case model (class C, n=2, all interference on T={1,2}).
#   N_delta(a,v,th) = sum_l kappa_l(delta) (a_l + c v_l) - osc_w D_delta(w)/2,   D_delta(w)=phi(w)-phi(w-delta),
#   phi(w) = sum_l v_l cos(2 pi l w/p + th_l),  w in Z_p,   0 <= v_l <= cap a_l  (cap = 3^{s-1}).
#   ratio = min_delta N_delta / (Gamma_s(p) (sum_l a_l^{1/s})^s).   ratio>1 would beat the IF value in this model.
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
        self.K = torch.tensor(self.Knp)
        self.ang = 2 * np.pi * np.outer(w, l) / p                          # w x l
        self.Cw = torch.tensor(np.cos(self.ang)); self.Sw = torch.tensor(np.sin(self.ang))
        self.idx = torch.tensor((w[None, :] - d[:, None]) % p)            # delta x w  -> w-delta

    # ---------- exact evaluation (numpy) ----------
    def phi(self, v, th):
        return np.cos(self.ang + th[None, :]) @ v
    def parts(self, a, v, th):
        ph = self.phi(v, th); D = ph[None, :] - ph[self.idx.numpy()]
        osc = D.max(1) - D.min(1)
        cov = self.Knp @ a; bon = self.c * (self.Knp @ v)
        return cov, bon, osc
    def ratio(self, a, v, th):
        cov, bon, osc = self.parts(a, v, th)
        N = cov + bon - osc / 2
        return N.min() / (self.G * np.sum(a[a > 0] ** self.r) ** self.s)
    def info(self, a, v, th):
        cov, bon, osc = self.parts(a, v, th); N = cov + bon - osc / 2
        nrm = self.G * np.sum(a[a > 0] ** self.r) ** self.s
        d = int(np.argmin(N))
        return dict(ratio=N.min() / nrm, cov=cov[d] / nrm, bonus=bon[d] / nrm, pen=osc[d] / 2 / nrm,
                    covmin=cov.min() / nrm, delta=d + 1, supp=int(np.sum(a > 1e-9 * a.max())),
                    cheap=int(np.sum(v > 1e-6 * a.max())))

    # ---------- smooth objective (torch) ----------
    def soft(self, a, v, th, tau, tau2):
        ph = self.Cw @ (v * torch.cos(th)) - self.Sw @ (v * torch.sin(th))
        D = ph[None, :] - ph[self.idx]
        sc = a.sum()
        mx = tau * sc * torch.logsumexp(D / (tau * sc), 1); mn = -tau * sc * torch.logsumexp(-D / (tau * sc), 1)
        N = self.K @ (a + self.c * v) - (mx - mn) / 2
        nrm = self.G * (a ** self.r).sum() ** self.s
        R = N / nrm
        return -tau2 * torch.logsumexp(-R / tau2, 0)

def unpack(M, x, z, th, mask, fixv=None):
    a = torch.exp(x - x.max()) * mask
    if fixv is None: v = M.cap * a * torch.sigmoid(z)
    else: v = fixv * a
    return a, v, th

def optimise(M, a0, v0, th0, mask=None, stages=(0.05, 0.02, 0.008, 0.003, 0.001), iters=300, fixv=None, opt_th_only=False):
    F = M.F
    mask = torch.ones(F) if mask is None else torch.tensor(mask, dtype=torch.float64)
    a0 = np.maximum(a0, 1e-12 * a0.max())
    x = torch.tensor(np.log(a0)).requires_grad_()
    t0 = np.clip(v0 / (M.cap * a0), 1e-6, 1 - 1e-6)
    z = torch.tensor(np.log(t0 / (1 - t0))).requires_grad_()
    th = torch.tensor(th0.astype(float)).requires_grad_()
    params = [th] if opt_th_only else ([x, th] if fixv is not None else [x, z, th])
    for tau in stages:
        o = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-14)
        def closure():
            o.zero_grad(); a, v, t = unpack(M, x, z, th, mask, fixv)
            loss = -M.soft(a, v, t, tau, tau / 2); loss.backward(); return loss
        try: o.step(closure)
        except Exception as e: pass
    with torch.no_grad():
        a, v, t = unpack(M, x, z, th, mask, fixv)
    a = a.detach().numpy().copy(); v = v.detach().numpy().copy(); t = t.detach().numpy().copy()
    a[mask.numpy() == 0] = 0; v[mask.numpy() == 0] = 0
    return a, v, t

def lp_polish(M, a, v, th, S=None, rounds=8, cuts=40, fixv=None, verbose=False):
    """Given phases th, optimise (a,v) on support S exactly by reweighted LP with cutting planes over (w1,w2).
    minimise sum om_l a_l  s.t.  N_delta(a,v) >= 1 for all delta, 0<=v<=cap a.  Reweight om=a^{r-1}."""
    p, F = M.p, M.F
    if S is None: S = np.nonzero(a > 1e-9 * a.max())[0]
    S = np.array(S); k = len(S)
    Kd = M.Knp[:, S]
    E = np.cos(M.ang[:, S] + th[None, S])                    # w x k  (cos(2 pi l w/p + th))
    idx = M.idx.numpy()
    def Dmat(vS):
        ph = E @ vS; return ph[None, :] - ph[idx]            # delta x w
    best = (M.ratio(a, v, th), a.copy(), v.copy())
    aS = a[S].copy(); vS = v[S].copy()
    om = (aS / aS.max() + 1e-9) ** (M.r - 1)
    cut_set = {}
    def add_cuts(vS):
        D = Dmat(vS)
        for d in range(F):
            cut_set.setdefault(d, set()).add((int(np.argmax(D[d])), int(np.argmin(D[d]))))
    add_cuts(vS)
    for rw in range(rounds):
        for it in range(cuts):
            rows = [];
            for d, cs in cut_set.items():
                for (w1, w2) in cs:
                    # row for: sum_l K(a + c v) - 0.5 sum_l v_l [ (E[w1]-E[w1-d]) - (E[w2]-E[w2-d]) ] >= 1
                    e = (E[w1] - E[(w1 - d - 1) % p]) - (E[w2] - E[(w2 - d - 1) % p])
                    rows.append(np.concatenate([-Kd[d], -(M.c * Kd[d] - 0.5 * e)]))
            A = np.array(rows); b = -np.ones(len(rows))
            # v_l - cap a_l <= 0
            A2 = np.hstack([-M.cap * np.eye(k), np.eye(k)])
            if fixv is not None:
                Aeq = np.hstack([-fixv * np.eye(k), np.eye(k)]); beq = np.zeros(k)
            else: Aeq = None; beq = None
            cost = np.concatenate([om, np.zeros(k)])
            res = linprog(cost, A_ub=np.vstack([A, A2]), b_ub=np.concatenate([b, np.zeros(k)]), A_eq=Aeq, b_eq=beq,
                          bounds=(0, None), method='highs')
            if res.status != 0:
                if verbose: print('LP fail', res.message)
                return best
            aS, vS = res.x[:k], res.x[k:]
            D = Dmat(vS); osc = D.max(1) - D.min(1)
            N = Kd @ aS + M.c * (Kd @ vS) - osc / 2
            viol = N < 1 - 1e-9
            if not viol.any(): break
            for d in np.nonzero(viol)[0]:
                cut_set[d].add((int(np.argmax(D[d])), int(np.argmin(D[d]))))
        aa = np.zeros(F); vv = np.zeros(F); aa[S] = aS; vv[S] = vS
        R = M.ratio(aa, vv, th)
        if R > best[0]: best = (R, aa.copy(), vv.copy())
        omn = (aS / max(aS.max(), 1e-300) + 1e-9) ** (M.r - 1)
        if np.allclose(omn, om, rtol=1e-7): break
        om = omn
    return best

# ---------- critical payment constant ----------
def cstar(M, a, v, th):
    """smallest c at which (a,v,th) reaches ratio 1:  max_delta (G||a|| - cov + osc/2)/B_delta."""
    ph = M.phi(v, th); D = ph[None, :] - ph[M.idx.numpy()]
    osc = D.max(1) - D.min(1); cov = M.Knp @ a; B = M.Knp @ v
    nrm = M.G * np.sum(a[a > 0] ** M.r) ** M.s
    cd = (nrm - cov + osc / 2) / B
    return cd.max(), int(np.argmax(cd)) + 1

def soft_cstar(M, a, v, th, tau, tau2):
    ph = M.Cw @ (v * torch.cos(th)) - M.Sw @ (v * torch.sin(th))
    D = ph[None, :] - ph[M.idx]
    sc = a.sum()
    mx = tau * sc * torch.logsumexp(D / (tau * sc), 1); mn = -tau * sc * torch.logsumexp(-D / (tau * sc), 1)
    nrm = M.G * (a ** M.r).sum() ** M.s
    cd = (nrm - M.K @ a + (mx - mn) / 2) / (M.K @ v)
    return tau2 * torch.logsumexp(cd / tau2, 0)

def optimise_crit(M, a0, v0, th0, mask=None, stages=(0.05, 0.02, 0.008, 0.003, 0.001), iters=300, fixv=None):
    F = M.F
    mask = torch.ones(F) if mask is None else torch.tensor(mask, dtype=torch.float64)
    a0 = np.maximum(a0, 1e-12 * a0.max())
    x = torch.tensor(np.log(a0)).requires_grad_()
    t0 = np.clip(v0 / (M.cap * a0), 1e-6, 1 - 1e-6)
    z = torch.tensor(np.log(t0 / (1 - t0))).requires_grad_()
    th = torch.tensor(th0.astype(float)).requires_grad_()
    params = [x, th] if fixv is not None else [x, z, th]
    for tau in stages:
        o = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-14)
        def closure():
            o.zero_grad(); a, v, t = unpack(M, x, z, th, mask, fixv)
            loss = soft_cstar(M, a, v, t, tau, tau / 4); loss.backward(); return loss
        try: o.step(closure)
        except Exception as e: pass
    with torch.no_grad():
        a, v, t = unpack(M, x, z, th, mask, fixv)
    a = a.detach().numpy().copy(); v = v.detach().numpy().copy(); t = t.detach().numpy().copy()
    a[mask.numpy() == 0] = 0; v[mask.numpy() == 0] = 0
    return a, v, t
