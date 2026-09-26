# Kill-test for "junk is never locally optimal": ascend the TRUE normalized margin (n=2,q=2) from random/junky
# inits, polish to a KKT point with SLSQP (max t s.t. every gap >= t, P = 1), then report gamma/ansatz and the
# logit-tensor anatomy (cross-frequency X share, same-|k| junk share). usage: lmax_junk.py p Gamma seeds N
import torch, math, sys, numpy as np
from scipy.optimize import minimize, nnls
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); Gam=float(sys.argv[2]); seeds=int(sys.argv[3]); N=int(sys.argv[4]); h=(p-1)//2
target=3**-0.5*2*(6*p)**-1.5*Gam
x=torch.arange(p); Y=(x[:,None]+x[None,:])%p
mask=torch.ones(p,p,p,dtype=torch.bool); mask[x[:,None],x[None,:],Y]=False
def gaps_flat(v):
    E1,E2,W=v.reshape(3,N,p)
    Z=torch.einsum('ia,ib,ic->abc',E1,E2,W)
    corr=Z[x[:,None],x[None,:],Y]
    return (corr[:,:,None]-Z)[mask]
def anatomy(v):
    E1,E2,W=v.reshape(3,N,p)
    Z=torch.einsum('ia,ib,ic->abc',E1,E2,W); Z=Z-Z.mean(2,keepdim=True)
    F=torch.fft.fftn(Z); a=torch.arange(p); f=torch.minimum(a,p-a)
    same=(f[:,None,None]==f[None,:,None])&(f[None,:,None]==f[None,None,:])&(f[None,None,:]>0)
    fa,fb,fc=[t.reshape(-1) for t in torch.meshgrid(a,a,a,indexing='ij')]
    clean=((fa+fc)%p==0)&((fb+fc)%p==0)&(fc!=0); clean=clean.reshape(p,p,p)
    P2=F.abs()**2; tot=P2.sum()
    Gk=[P2.reshape(-1)[clean.reshape(-1)&((fc==k)|(fc==p-k))].sum().item()/tot.item() for k in range(1,h+1)]
    return P2[~same].sum().item()/tot.item(), P2[same&~clean].sum().item()/tot.item(), [round(g,3) for g in Gk]
res=[]
for s in range(seeds):
    g=torch.Generator().manual_seed(s)
    v=torch.randn(3*N*p,generator=g)
    if s%2==1:  # junky init: 1-2 pure neurons per frequency on a random support
        v=0.05*v; E=v.reshape(3,N,p); i=0
        K=int(torch.randint(1,h+1,(1,),generator=g)); S=(torch.randperm(h,generator=g)[:K]+1).tolist()
        per=int(torch.randint(1,3,(1,),generator=g))
        for k in S:
            for _ in range(per):
                if i>=N: break
                a_,b_=(torch.rand(2,generator=g)*2*math.pi).tolist()
                E[0,i]+=torch.cos(2*math.pi*k*x/p+a_);E[1,i]+=torch.cos(2*math.pi*k*x/p+b_);E[2,i]+=torch.cos(2*math.pi*k*x/p+a_+b_);i+=1
        v=E.reshape(-1).clone()
    v.requires_grad_()
    opt=torch.optim.Adam([v],lr=3e-3)
    for step in range(2500):
        tau=0.3*(0.01**(step/2500))
        vn=v/(v**2).sum().sqrt()
        loss=tau*torch.logsumexp(-gaps_flat(vn)/target/tau,0)
        opt.zero_grad();loss.backward();opt.step()
    v0=(v/(v**2).sum().sqrt()).detach()
    t0=gaps_flat(v0).min().item()
    # SLSQP polish: z=(v,t), maximize t
    def fobj(z): return -z[-1]
    def gobj(z): gg=np.zeros_like(z); gg[-1]=-1; return gg
    def cin(z):
        vv=torch.tensor(z[:-1]); return (gaps_flat(vv)-z[-1]).numpy()
    def jin(z):
        vv=torch.tensor(z[:-1]); J=torch.autograd.functional.jacobian(gaps_flat,vv,vectorize=True).numpy()
        return np.hstack([J,-np.ones((J.shape[0],1))])
    def ceq(z): return np.array([np.dot(z[:-1],z[:-1])-1.0])
    def jeq(z): return np.hstack([2*z[:-1],[0.0]])[None,:]
    z0=np.concatenate([v0.numpy(),[t0]])
    r=minimize(fobj,z0,jac=gobj,method='SLSQP',constraints=[{'type':'ineq','fun':cin,'jac':jin},{'type':'eq','fun':ceq,'jac':jeq}],options={'maxiter':800,'ftol':1e-14})
    vz=torch.tensor(r.x[:-1]); vz=vz/(vz**2).sum().sqrt()
    gv=gaps_flat(vz); m=gv.min().item()
    # KKT residual: grad of sum pi*gap on active set = mu * grad P (=2 v), pi>=0, sum pi = 1
    act=(gv-m<1e-7*max(1,abs(m))).nonzero().reshape(-1)
    J=torch.autograd.functional.jacobian(gaps_flat,vz,vectorize=True)[act].numpy()
    A=np.vstack([J.T, np.ones((1,len(act)))]); mu=m*3/2  # Euler: sum pi*grad gap . v = 3m => mu*2*|v|^2=3m
    b=np.concatenate([2*mu*vz.numpy(),[1.0]])
    pi,rn=nnls(A,b)
    X,Jk,Gk=anatomy(vz)
    res.append((m/target,X,Jk))
    if m/target<0.999:
        def polish(zz):
            rr=minimize(fobj,zz,jac=gobj,method='SLSQP',constraints=[{'type':'ineq','fun':cin,'jac':jin},{'type':'eq','fun':ceq,'jac':jeq}],options={'maxiter':800,'ftol':1e-14})
            vv=torch.tensor(rr.x[:-1]); vv=vv/(vv**2).sum().sqrt(); return vv,gaps_flat(vv).min().item()
        rng=np.random.default_rng(s)
        for eps in [1e-4,1e-3,1e-2,3e-2,1e-1]:
            outs=[]
            for trial in range(4):
                vp=vz.numpy()+eps*rng.standard_normal(vz.numel()); vp/=np.linalg.norm(vp)
                vv,mm=polish(np.concatenate([vp,[gaps_flat(torch.tensor(vp)).min().item()]]))
                outs.append(round(mm/target,6))
            print(f"   escape eps {eps}: {outs}",flush=True)
        np.save(f'endpoint_p{p}_s{s}.npy',vz.numpy())
    print(f"seed {s} {'junk' if s%2 else 'rand'}: pre {t0/target:.4f} -> post {m/target:.6f} ({r.message[:20]}) active {len(act)} KKTres {rn:.1e}  X={X:.4f} junk={Jk:.4f} clean-by-k={Gk}",flush=True)
