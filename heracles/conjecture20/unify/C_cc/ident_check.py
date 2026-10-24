import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
def decomp(net):
    p,n,nu,s=net.p,net.n,net.nu,net.s
    B=G.Bp(p,n); G1=p/(p-1)
    a=((net.U**2).sum((1,2))+(net.W**2).sum(1))**(nu/2); m=a/a.sum()
    h=net.margin(); beta=net.clean(); at=net.atoms()
    assert np.allclose(at.sum(0),beta)
    sig=1-np.real(at).sum(1)/(B*a)                       # per-neuron clean misalignment
    budget=np.abs(at).sum(1)/(B*a)                       # single-neuron budget ratio (<=1)
    K=G.kappa_mat(p)[:(p-1)//2]
    clean_min=(K@np.real(beta)).min()
    dprog=(G1*np.real(beta).sum()-clean_min)/(B*G1*a.sum())
    dint=(clean_min-h)/(B*G1*a.sum())
    H=np.log((m**(1/s)).sum())/(1-1/s) if s>1 else -(m*np.log(m)).sum()
    pred=B*G1*(1-(m*sig).sum()-dprog-dint)*np.exp(-(s-1)*H)
    return dict(gamma=net.gamma(),pred=pred,Em_sig=(m*sig).sum(),dprog=dprog,dint=dint,H=H,maxbudget=budget.max(),minsig=sig.min())
rng=np.random.default_rng(1)
for (p,n,b,mw) in [(5,2,2.0,6),(7,2,2.5,9),(5,3,2.0,8),(3,2,2.2,4)]:
    for t in range(3):
        net=G.Net(rng.normal(size=(mw,n,p)),rng.normal(size=(mw,p)),b)
        d=decomp(net); print((p,n,b),{k:round(float(v),6) for k,v in d.items()})
# IF optimum at (5,2), b=2 and (7,2) b=2.5
for (p,n,b) in [(5,2,2.0),(7,2,2.5),(11,2,2.0)]:
    s=(n+1)/b; Gm,y,_=G.programme(p,s)
    net=G.if_network(p,n,y,b); d=decomp(net)
    print('IF',(p,n,b),{k:round(float(v),6) for k,v in d.items()}, 'gamma_IF',G.gamma_if(p,n,b,Gm))
print('--- violators / near-optima')
for f in ['viol_p5_n2_b2','viol_p7_n2_b2','viol_p11_n2_b2']:
    d=np.load('/home/user/grok/heracles/conjecture20/unify/split/'+f+'.npz')
    
    for kk in range(4):
      net=G.Net(d['U%d'%kk],d['W%d'%kk],2.0); s=net.s; Gm,y,_=G.programme(net.p,s)
    dd=decomp(net); print(f,{k:round(float(v),5) for k,v in dd.items()},'gamma/gIF',round(net.gamma()/G.gamma_if(net.p,net.n,2.0,Gm),5))
