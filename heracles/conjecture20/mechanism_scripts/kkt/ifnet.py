import numpy as np, sys
sys.path.insert(0,'.'); sys.dont_write_bytecode=True
sys.path.insert(0,'../..')
from gamma import Gamma
from kkt import Net, gamma_IF
from scipy.optimize import nnls
def if_net(p,n,b):
    nu=n+1; s=nu/b; G,y=Gamma(p,s,True)
    U=[];W=[]
    a=np.arange(p)
    for jj,yj in enumerate(y):
        j=jj+1
        if yj<1e-12: continue
        lam=(yj*2**n/nu)**(1/nu)
        for l in range(nu):
            al=np.pi*l/nu
            U.append(lam*np.cos(2*np.pi*j*a/p+al)); W.append(lam*np.cos(2*np.pi*j*a/p+n*al))
    return Net(U,W,n,b),G,y
if __name__=='__main__':
    p,n,b=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
    net,G,y=if_net(p,n,b)
    Gm=net.gaps()
    print('p',p,'n',n,'b',b,'Gamma',G,'y*',np.round(y,4),'m',net.m)
    print('gap const in x per delta?',np.round(Gm[:,1:].max(0)-Gm[:,1:].min(0),12),' gaps',np.round(Gm[0,1:],5))
    print('gamma/gIF',net.gamma()/gamma_IF(p,n,b,G))
    A,q,lam,res,h=net.kkt(1e-9)
    print('|A|',len(A),'KKT res (NNLS)',res,'lam',lam,'expect',net.nu*h/(b*net.cost()))
    # uniform-in-x multiplier: q_{k,d} = cs_k/p^n mu_d
    B=sorted(set(d for k,d in A)); J=np.array([sum(net.cs[k]/p**n*net.grad_gap(k,d) for k in range(len(net.Nc))) for d in B]).T
    mt,r2=nnls(J,net.grad_cost()); print('uniform-in-x multiplier: binding deltas',B,'mu',np.round(mt/mt.sum(),4),'rel residual',r2/np.linalg.norm(net.grad_cost()))
    # multiplier polytope dimension
    JA=np.array([net.grad_gap(k,d) for k,d in A]).T
    sv=np.linalg.svd(JA,compute_uv=False); rk=(sv>1e-9*sv[0]).sum()
    print('J_A',JA.shape,'rank',rk,'=> multiplier affine dim <=',len(A)-rk)
    np.savez('if_p%d_n%d_b%g.npz'%(p,n,b),U=net.U,W=net.W,n=n,p=p)
