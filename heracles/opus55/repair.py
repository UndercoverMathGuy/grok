# Repair test: start from X-blind-relaxation skeletons (pure, junky, high relaxed bound, negative true margin),
# add free neurons, and maximize the TRUE full-table margin/P^{3/2}. Crossing 1.0 = counterexample to (T).
# usage: repair.py p Gamma seed1,seed2,... extra_width
import torch, math, sys
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); Gam=float(sys.argv[2]); skel=[int(s) for s in sys.argv[3].split(",")]; extra=int(sys.argv[4])
h=(p-1)//2; target=3**-0.5*2*(6*p)**-1.5*Gam
sys.argv=[sys.argv[0],str(p),str(Gam),'0']
exec(open('xblind_relax.py').read().split('# calibration')[0])   # reuse gap_table/bounds/build/P
def relax(sd):
    g=torch.Generator().manual_seed(sd); Nper=[1,2,3][sd%3]; K=h
    par=torch.randn(K*Nper*6,generator=g)
    keep=torch.rand(K,generator=g)<torch.rand(1,generator=g).item()*0.8+0.1
    par=(par.view(K,Nper*6)*keep[:,None]).view(-1).requires_grad_()
    opt=torch.optim.Adam([par],lr=1e-2)
    for step in range(2500):
        tau=0.3*(0.01**(step/2500))
        E=build(par,K,Nper); b=bounds(gap_table(*E))/P(E)**1.5/target
        loss=tau*torch.logsumexp(-b/tau,0); opt.zero_grad();loss.backward();opt.step()
    return [e.detach() for e in build(par,K,Nper)]
def true_gamma(E): 
    G=gap_table(*E); return (G[:,:,1:].min()/P(E)**1.5/target).item()
for sd in skel:
    E=relax(sd); 
    with torch.no_grad(): rb=(bounds(gap_table(*E)).min()/P(E)**1.5/target).item()
    g=torch.Generator().manual_seed(1000+sd); sc=E[0].abs().max().item()
    th=[torch.cat([e,0.05*sc*torch.randn(extra,p,generator=g)]).requires_grad_() for e in E]
    opt=torch.optim.Adam(th,lr=2e-3); hist=[]
    for step in range(4000):
        tau=0.3*(0.01**(step/4000))
        Pn=sum((t**2).sum() for t in th); tn=[t/Pn.sqrt() for t in th]
        gq=gap_table(*tn)[:,:,1:].reshape(-1)/target
        loss=tau*torch.logsumexp(-gq/tau,0); opt.zero_grad();loss.backward();opt.step()
        if step%1000==999: hist.append(round(true_gamma([t.detach() for t in th]),4))
    print(f"skeleton seed {sd}: relaxed bound {rb:.4f}, start true {true_gamma(E):.4f} -> repaired true margin/ansatz {hist}",flush=True)
