# Unrestricted raw-weight search for n=2,q=2 normalized margin, vs the ansatz value.
# usage: search.py p Gamma seeds
import torch, math, sys
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); Gam=float(sys.argv[2]); seeds=int(sys.argv[3]); h=(p-1)//2
nu=3; target=nu**-0.5*2*(6*p)**-1.5*Gam
x=torch.arange(p); Y=(x[:,None]+x[None,:])%p
mask=torch.ones(p,p,p,dtype=torch.bool); mask[x[:,None],x[None,:],Y]=False
def gaps(E1,E2,W):
    Z=torch.einsum('ia,ib,ic->abc',E1,E2,W)
    corr=Z[x[:,None],x[None,:],Y]
    return (corr[:,:,None]-Z)[mask]
def gamma(E1,E2,W):
    P=(E1**2).sum()+(E2**2).sum()+(W**2).sum()
    return gaps(E1,E2,W).min()/P**1.5
def init(kind,N,g):
    if kind=='rand': return [torch.randn(N,p,generator=g) for _ in range(3)]
    K=int(torch.randint(min(2,h),h+1,(1,),generator=g)); S=torch.randperm(h,generator=g)[:K]+1
    per=int(torch.randint(1,3,(1,),generator=g))  # 1 or 2 neurons per frequency = junk
    E1=torch.randn(N,p,generator=g)*0.05;E2=torch.randn(N,p,generator=g)*0.05;W=torch.randn(N,p,generator=g)*0.05
    i=0
    for k in S.tolist():
        for _ in range(per):
            a,b=(torch.rand(2,generator=g)*2*math.pi).tolist()
            E1[i]+=torch.cos(2*math.pi*k*x/p+a);E2[i]+=torch.cos(2*math.pi*k*x/p+b);W[i]+=torch.cos(2*math.pi*k*x/p+a+b);i+=1
            if i>=N: break
        if i>=N: break
    return [E1,E2,W]
# calibration: dense block must reproduce the closed form exactly
E1=[];E2=[];W=[]
for k in range(1,h+1):
    for l in range(3):
        ph=math.pi*l/3
        E1.append(torch.cos(2*math.pi*k*x/p+ph));E2.append(torch.cos(2*math.pi*k*x/p+ph));W.append(torch.cos(2*math.pi*k*x/p+2*ph))
gd=gamma(torch.stack(E1),torch.stack(E2),torch.stack(W)).item()
print('calibration dense block gamma / formula', gd/(nu**-0.5*2*(6*p)**-1.5*(p/(2*h**1.5))), ' dense/target',gd/target,flush=True)
best=-1;N=3*h+3
for s in range(seeds):
    g=torch.Generator().manual_seed(s)
    kind='rand' if s%3==0 else 'junk'
    th=[t.requires_grad_() for t in init(kind,N,g)]
    opt=torch.optim.Adam(th,lr=3e-3)
    for step in range(3000):
        tau=0.3*(0.01**(step/3000))
        P=sum((t**2).sum() for t in th)
        th_n=[t/P.sqrt() for t in th]
        gq=gaps(*th_n)/target
        loss=tau*torch.logsumexp(-gq/tau,0)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): val=gamma(*th).item()
    best=max(best,val)
    print(f"seed {s} {kind}: gamma/target = {val/target:.4f}   best {best/target:.4f}",flush=True)
