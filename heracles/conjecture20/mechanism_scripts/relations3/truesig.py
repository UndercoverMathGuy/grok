# Adversarial (upper) estimate of the weighted two-sided Sidon constant
#   sigma_delta(S) = inf_{b>=0,psi} (1/2)osc_w sum_l b_l cos(2 pi l w/p+psi_l) / sum_l b_l s_l(delta)
# by smoothed min-max + LBFGS with annealing, many restarts.  (Upper bound on sigma: any (b,psi) found is a witness.)
import numpy as np, torch, sys, json
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
def sigma_adv(p,S,sv,restarts=20,seed=0,b0=None):
    S=np.asarray(S); w=np.arange(p); ang=torch.tensor(2*np.pi*np.outer(w,S)/p); svt=torch.tensor(sv)
    C=torch.cos(ang); Sn=torch.sin(ang); rng=np.random.default_rng(seed); best=(9,None)
    def val(z,psi,tau=None):
        b=torch.exp(z-z.max()); P=C@(b*torch.cos(psi))-Sn@(b*torch.sin(psi))
        if tau is None: o=(P.max()-P.min())/2
        else: o=tau*(torch.logsumexp(P/tau,0)+torch.logsumexp(-P/tau,0))/2
        return o/(b*svt).sum()
    for R in range(restarts):
        z=torch.tensor(np.log(b0) if (b0 is not None and R==0) else rng.normal(0,0.7,len(S))).requires_grad_()
        psi=torch.tensor(rng.uniform(0,2*np.pi,len(S))).requires_grad_()
        for tau in [0.3,0.1,0.03,0.01,0.003]:
            opt=torch.optim.LBFGS([z,psi],lr=1,max_iter=300,line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(); bn=torch.exp(z-z.max()).sum(); l=val(z,psi,tau*bn.detach()); l.backward(); return l
            try: opt.step(closure)
            except Exception: pass
        with torch.no_grad(): v=val(z,psi).item()
        if v<best[0]: best=(v,(torch.exp(z-z.max()).detach().numpy(),psi.detach().numpy()))
    return best
if __name__=="__main__":
    sys.path.insert(0,'.')
    from cert import kapS
    p=int(sys.argv[1]); K=int(sys.argv[2]); d=json.load(open(f'large_{p}.json'))
    runs=[r for r in d['runs'] if len(r['S'])==K]; r=max(runs,key=lambda r:r['Gamma'])
    S=np.array(r['S']); y=np.array(r['y']); Km=kapS(p,S); cov=Km@y; B=np.nonzero(cov<=cov.min()*(1+1e-6))[0]
    d0=B[np.argmin(Km[B].sum(1))]; sv=np.sqrt(Km[d0]/2)
    v,(b,psi)=sigma_adv(p,S,sv,restarts=int(sys.argv[3]),b0=y*sv)
    print(f"p={p} K={K} delta0={d0+1} sumk={Km[d0].sum():.2f} sigP={1/np.sqrt(Km[d0].sum()):.3f}  adversarial sigma <= {v:.3f}  b(normalised)={np.round(b/b.max(),2)}",flush=True)
