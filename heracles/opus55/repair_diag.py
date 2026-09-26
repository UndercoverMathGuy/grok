# Same as repair.py but reports HOW the repaired net reaches the ansatz: cross-frequency energy share of the
# logit tensor (X), same-|k| junk share, and the frequency support. usage: repair_diag.py p Gamma seed1,... extra
import torch, math, sys
torch.set_default_dtype(torch.float64)
exec(open('repair.py').read().split('for sd in skel:')[0])
def anatomy(E):
    Z=torch.einsum('ia,ib,ic->abc',*E); Z=Z-Z.mean(2,keepdim=True)
    F=torch.fft.fftn(Z); a=torch.arange(p); f=torch.minimum(a,p-a)
    same=(f[:,None,None]==f[None,:,None])&(f[None,:,None]==f[None,None,:])&(f[None,None,:]>0)
    fa,fb,fc=[t.reshape(-1) for t in torch.meshgrid(a,a,a,indexing='ij')]
    clean=torch.zeros(p,p,p,dtype=torch.bool); clean[fa,fb,fc]=((fa+fc)%p==0)&((fb+fc)%p==0)&(fc!=0)
    tot=(F.abs()**2).sum()
    X=(F.abs()**2)[~same].sum()/tot; J=(F.abs()**2)[same&~clean].sum()/tot
    F2=(F.abs()**2).reshape(-1); cl=clean.reshape(-1)
    Gk=[F2[cl&((fc==k)|(fc==p-k))].sum().item()/tot.item() for k in range(1,h+1)]
    return X.item(),J.item(),[k+1 for k,g in enumerate(Gk) if g>0.02],[round(g,3) for g in Gk]
for sd in skel:
    E=relax(sd); x0=anatomy(E)
    g=torch.Generator().manual_seed(1000+sd); sc=E[0].abs().max().item()
    th=[torch.cat([e,0.05*sc*torch.randn(extra,p,generator=g)]).requires_grad_() for e in E]
    opt=torch.optim.Adam(th,lr=2e-3)
    for step in range(4000):
        tau=0.3*(0.01**(step/4000))
        Pn=sum((t**2).sum() for t in th); tn=[t/Pn.sqrt() for t in th]
        gq=gap_table(*tn)[:,:,1:].reshape(-1)/target
        loss=tau*torch.logsumexp(-gq/tau,0); opt.zero_grad();loss.backward();opt.step()
    Ef=[t.detach() for t in th]; x1=anatomy(Ef)
    print(f"seed {sd}: skeleton X={x0[0]:.3f} junk={x0[1]:.3f} supp={x0[2]}  ->  repaired gamma/ansatz={true_gamma(Ef):.4f} X={x1[0]:.4f} junk={x1[1]:.4f} supp={x1[2]} clean-share-by-k={x1[3]}",flush=True)
