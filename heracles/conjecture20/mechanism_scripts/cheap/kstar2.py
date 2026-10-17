# k*(n,s) = sup (Re A - nu^{1-s} Q^s)/sqrt(E/2) over atom systems (beta_i, xi_i in T^n).
# E = sum_{T!=0}|sum_i beta_i xi_i^T|^2 = sum_{ij} beta_i conj(beta_j) [prod_t(1+xi_it conj xi_jt) - 1].
import torch, math, sys, numpy as np
torch.set_default_dtype(torch.float64)
def f(rho,al,th,nu,s):
    r=1/s; b=torch.exp(rho)*torch.exp(1j*al)
    A=b.sum(); Q=torch.exp(r*rho).sum()
    X=torch.exp(1j*th)                       # N x n
    G=torch.prod(1+X[:,None,:]*X.conj()[None,:,:],dim=2)-1   # N x N
    E=torch.real(b.conj()@ (G.T) @ b) if False else torch.real((b[:,None]*b.conj()[None,:]*G).sum())
    num=A.real-nu**(1-s)*Q**s
    return num/torch.sqrt(torch.clamp(E,min=1e-30)/2), num, E
def search(n,s,N,starts,seed):
    nu=n+1; g=torch.Generator().manual_seed(seed); best=(-1e9,None)
    for k in range(starts):
        rho=(torch.rand(N,generator=g)*1.0-0.5).requires_grad_()
        al=(torch.rand(N,generator=g)*0.6-0.3).requires_grad_()
        th=(torch.rand(N,n,generator=g)*2*math.pi).requires_grad_()
        opt=torch.optim.Adam([rho,al,th],lr=0.03)
        for it in range(1500):
            opt.zero_grad(); val,num,E=f(rho,al,th,nu,s)
            loss=-val if num.item()>0 else -num*10
            loss.backward(); opt.step()
            with torch.no_grad(): rho-=rho.max()   # normalise scale
        val,num,E=f(rho,al,th,nu,s)
        if num.item()>0 and val.item()>best[0]: best=(val.item(),(rho.detach().numpy(),al.detach().numpy(),th.detach().numpy()))
    return best
if __name__=="__main__":
    for n in [int(a) for a in sys.argv[1:]]:
        nu=n+1; s=nu/2
        res=[]
        for N in range(max(1,nu-6),nu+1):
            v,_=search(n,s,N,16,N+100)
            res.append(v)
        print(f"n={n} nu={nu} s={s}: k* by width = {np.round(res,4)}  max {max(res):.4f}",flush=True)
