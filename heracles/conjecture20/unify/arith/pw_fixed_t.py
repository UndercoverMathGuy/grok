# Label-row test at a fixed interference level t (all cheap features), phases adversarial.
# Reports R(t) = max_theta min_w C(w)/sum a^r.  R>1: the label rows alone do not close that configuration.
# Supports: programme supports (relations3/*.json) and progressions {1..N} (equal weights) with chirp/random starts.
import numpy as np, torch, json, sys, math
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
s = 1.5; r = 1/s; c = 1-(2/3)**(s-1); Kq = 5/(5+4*r)

def price(t):  # a / Re(beta) as a function of t = v/Re(beta)
    return max(1-c*t, 1-Kq*t*t) if t <= 1 else t*(1-c)

def Rfixed(p, S, a, t, restarts=10, seed=1, iters=300):
    S = np.asarray(S); a = np.asarray(a, float); a = a/np.sum(a**r)**(1/r)
    w = torch.arange(p, dtype=torch.float64)
    ang = 2*math.pi*torch.outer(torch.tensor(S, dtype=torch.float64), w)/p
    beta = torch.tensor(a/price(t)); v = t*beta
    rng = np.random.default_rng(seed); best = -1
    K = len(S)
    for k in range(restarts):
        th0 = [np.zeros(K), np.pi*np.arange(K)**2/K, np.pi*(np.arange(K)%2)][k] if k < 3 else rng.uniform(0, 2*np.pi, K)
        th = torch.tensor(th0, requires_grad=True)
        def f(th, tau):
            x = beta[:, None] + v[:, None]*torch.cos(ang + th[:, None])
            Cw = ((torch.clamp(x, min=0) + 1e-300)**r).sum(0)
            return Cw.min() if tau is None else -tau*torch.logsumexp(-Cw/tau, 0)
        for tau in [3e-2, 1e-2, 3e-3, 1e-3]:
            opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(); loss = -f(th, tau); loss.backward(); return loss
            try: opt.step(closure)
            except Exception: pass
        with torch.no_grad(): best = max(best, f(th, None).item())
    return best

if __name__ == '__main__':
    t = float(sys.argv[1]); mode = sys.argv[2]
    if mode == 'prog':
        sup = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/supports.json'))
        for p in [101, 211, 401, 1009]:
            d = sup[str(p)]; print(f"programme p={p} K={len(d['S'])} t={t}: R = {Rfixed(p, d['S'], d['y'], t):.5f}", flush=True)
        for p in [10007]:
            d = json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))
            for run in sorted(d['runs'], key=lambda x: -x['Gamma'])[:3]:
                print(f"programme p={p} K={run['K']} t={t}: R = {Rfixed(p, run['S'], run['y'], t, restarts=6):.5f}", flush=True)
    else:
        p = int(mode)
        for N in [5, 10, 20, 40, 80]:
            S = np.arange(1, N+1); a = np.ones(N)
            print(f"progression p={p} N={N} t={t}: R = {Rfixed(p, S, a, t):.5f}", flush=True)
