# Adversarial test of the label-row (pointwise) closure in the relations case.
# Allocation a (IF prices) on support S.  Per cheap feature l: interference ratio t_l = v_l/Re(beta_l) >= 0, phase theta_l.
# Largest admissible clean amplitude: Re(beta_l) = a_l / (max(1,t_l) - c t_l)   [P12 + two-atom cap].
# Label-row cost at label w (w = 2y ranges over Z_p):
#   C(w) = sum_l (Re beta_l + v_l cos(2 pi l w/p + theta_l))_+^r .
# Closure by the pointwise principle iff  min_w C(w) <= sum_l a_l^r.   The adversary maximises R = min_w C(w)/sum a^r.
# R = 1 at t = 0 (interference-free); R > 1 means the label rows alone do not close (detection is then needed).
import numpy as np, torch, json, sys, math
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

def adv(p, S, a, s=1.5, restarts=12, seed=0, tmax=3.0, iters=300):
    r = 1/s; c = 1-(2/3)**(s-1)
    S = np.asarray(S); a = np.asarray(a, float); a = a/np.sum(a**r)**(1/r)    # normalise sum a^r = 1
    w = torch.arange(p, dtype=torch.float64)
    ang = 2*math.pi*torch.outer(torch.tensor(S, dtype=torch.float64), w)/p       # K x p
    at = torch.tensor(a)
    rng = np.random.default_rng(seed); best = (-1, None)
    for k in range(restarts):
        if k == 0: th0 = np.zeros(len(S)); t0 = np.full(len(S), 0.8)
        elif k == 1: th0 = np.pi*np.arange(len(S))**2/len(S); t0 = np.full(len(S), 0.8)
        elif k == 2: th0 = np.zeros(len(S)); t0 = np.full(len(S), 0.05)
        else: th0 = rng.uniform(0, 2*np.pi, len(S)); t0 = rng.uniform(0.05, 1.2, len(S))
        th = torch.tensor(th0, requires_grad=True); zt = torch.tensor(np.log(t0/(tmax-t0)), requires_grad=True)
        def R_of(th, zt, tau):
            t = tmax*torch.sigmoid(zt)
            beta = at/(torch.clamp(t, min=1.0) - c*t)
            v = t*beta
            x = beta[:, None] + v[:, None]*torch.cos(ang + th[:, None])
            Cw = (torch.clamp(x, min=0) + 1e-300)**r
            Cw = Cw.sum(0)
            if tau is None: return Cw.min()
            return -tau*torch.logsumexp(-Cw/tau, 0)
        for tau in [3e-2, 1e-2, 3e-3, 1e-3]:
            opt = torch.optim.LBFGS([th, zt], lr=1, max_iter=iters, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(); loss = -R_of(th, zt, tau); loss.backward(); return loss
            try: opt.step(closure)
            except Exception: pass
        with torch.no_grad():
            val = R_of(th, zt, None).item(); t = (tmax*torch.sigmoid(zt)).numpy()
        if val > best[0]: best = (val, dict(t=t.round(3).tolist(), theta=th.detach().numpy().round(3).tolist()))
    return best

if __name__ == '__main__':
    sup = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/supports.json'))
    ps = [int(x) for x in sys.argv[1].split(',')]
    for p in ps:
        d = sup[str(p)]
        val, info = adv(p, d['S'], d['y'])
        print(f"p={p} K={len(d['S'])}: max_adv min_w C(w)/sum a^r = {val:.5f}   t={info['t']}", flush=True)
