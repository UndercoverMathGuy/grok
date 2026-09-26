# How close do junk-carrying PURE networks get to the ansatz as p grows? (n=2, q=2)
# Per frequency k (all h of them) N atoms: c_l in C, slot phases v_l1, v_l2 on the circle. Uses the exact profile identity
#   4*gap(x,D) = sum_k Re[(1 - w^{-kD}) H_k(x)],  H_k(x) = sum_l c_l (1+v_l1 w^{-2k x1})(1+v_l2 w^{-2k x2})
# and the balanced cost P = (3p/2) sum|c|^{2/3}. ratio = sqrt3 * min gap' / (sum|c|^{2/3})^{3/2} / Gamma (=1 at the ansatz).
# usage: ptrend.py p N seeds [steps]
import torch, math, sys
torch.set_default_dtype(torch.float64)
p=int(sys.argv[1]); N=int(sys.argv[2]); seeds=int(sys.argv[3]); steps=int(sys.argv[4]) if len(sys.argv)>4 else 3000
h=(p-1)//2; k=torch.arange(1,h+1); x=torch.arange(p); D=torch.arange(1,p)
A=1-torch.cos(2*math.pi*torch.outer(k,k)/p)
def gamma_prog(trials=200):
    best=0;bb=None; g=torch.Generator().manual_seed(0)
    for t in range(trials):
        z=(torch.randn(h,generator=g)*2).requires_grad_(); opt=torch.optim.Adam([z],lr=0.05)
        for s in range(600):
            tau=0.05*(0.01**(s/600)); b=torch.softmax(z,0); v=A@b**1.5
            loss=tau*torch.logsumexp(-v/tau,0); opt.zero_grad();loss.backward();opt.step()
        b=torch.softmax(z,0).detach(); val=(A@b**1.5).min().item()
        if val>best: best=val;bb=b
    return best,bb
Gam,bopt=gamma_prog()
print(f"p={p} Gamma(lower bd)={Gam:.6f} support={[i+1 for i in range(h) if bopt[i]>1e-3]}",flush=True)
W=1-torch.exp(-2j*math.pi*torch.outer(k,D)/p)            # [h, p-1]
U=torch.exp(-2j*math.pi*2*torch.outer(k,x)/p)            # [h, p]  u = w^{-2kx}
def profile(cr,ci,a1,a2):
    c=torch.complex(cr,ci); v1=torch.exp(1j*a1); v2=torch.exp(1j*a2)   # [h,N]
    C0=c.sum(1); C1=(c*v1).sum(1); C2=(c*v2).sum(1); C12=(c*v1*v2).sum(1)
    H=C0[:,None,None]+C1[:,None,None]*U[:,:,None]+C2[:,None,None]*U[:,None,:]+C12[:,None,None]*U[:,:,None]*U[:,None,:]
    g=torch.einsum('kab,kd->abd',H,W).real
    cost=(cr**2+ci**2).pow(1/3).sum()
    return g,cost
best=-9
for sd in range(seeds):
    gg=torch.Generator().manual_seed(sd)
    if sd%2==0: amp=torch.rand(h,1,generator=gg)**3            # random support weights
    else: amp=(bopt[:,None]**1.5)+0.02*torch.rand(h,1,generator=gg)   # start near the optimal support
    cr=(amp*torch.randn(h,N,generator=gg)).requires_grad_(); ci=(amp*torch.randn(h,N,generator=gg)).requires_grad_()
    a1=(2*math.pi*torch.rand(h,N,generator=gg)).requires_grad_(); a2=(2*math.pi*torch.rand(h,N,generator=gg)).requires_grad_()
    opt=torch.optim.Adam([cr,ci,a1,a2],lr=1e-2)
    for st in range(steps):
        tau=0.2*(0.005**(st/steps))
        g,cost=profile(cr,ci,a1,a2); r=math.sqrt(3)*g.reshape(-1)/cost**1.5/Gam
        loss=tau*torch.logsumexp(-r/tau,0); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        g,cost=profile(cr,ci,a1,a2); val=(math.sqrt(3)*g.min()/cost**1.5/Gam).item()
        mag=(cr**2+ci**2).sqrt(); share=mag.pow(2/3).sum(1); share=share/share.sum()
        supp=[i+1 for i in range(h) if share[i]>0.01]
    best=max(best,val)
    print(f"seed {sd} N={N}: ratio {val:.4f} best {best:.4f}  support(K={len(supp)}) {supp}",flush=True)
