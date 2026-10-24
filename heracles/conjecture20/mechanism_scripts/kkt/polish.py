import numpy as np, sys
sys.path.insert(0,'.')
from scipy.optimize import minimize
from kkt import Net

def polish(net, iters=500, ftol=1e-15, verbose=True):
    m,p=net.m,net.p; P0=net.cost(); K=len(net.Nc)
    pairs=[(k,d) for k in range(K) for d in range(1,p)]
    def setx(x):
        net.U=x[:m*p].reshape(m,p).copy(); net.W=x[m*p:2*m*p].reshape(m,p).copy()
    x0=np.concatenate([net.U.ravel(),net.W.ravel(),[net.margin()]])
    def obj(x): return -x[-1]
    def objg(x): g=np.zeros_like(x); g[-1]=-1; return g
    def cons(x):
        setx(x); G=net.gaps(); return np.array([G[k,d] for k,d in pairs])-x[-1]
    def consj(x):
        setx(x); J=np.array([net.grad_gap(k,d) for k,d in pairs]); return np.hstack([J,-np.ones((len(pairs),1))])
    def eq(x): setx(x); return np.array([net.cost()-P0])
    def eqj(x): setx(x); return np.concatenate([net.grad_cost(),[0]])[None,:]
    res=minimize(obj,x0,jac=objg,method='SLSQP',constraints=[{'type':'ineq','fun':cons,'jac':consj},{'type':'eq','fun':eq,'jac':eqj}],options={'maxiter':iters,'ftol':ftol})
    setx(res.x)
    if verbose: print('SLSQP',res.message,'margin',net.margin(),'cost',net.cost())
    return net

if __name__=='__main__':
    from load import build1d
    from kkt import gamma_IF
    from math import cos,pi
    net,p,n=build1d(sys.argv[1])
    h=net.margin(); c=(1/h)**(1/net.nu); net.U*=c; net.W*=c
    gIF=gamma_IF(p,n,2.0,1-cos(2*pi/p))
    print('before', net.gamma()/gIF)
    polish(net)
    print('after', net.gamma()/gIF)
    h=net.margin(); c=(1/h)**(1/net.nu); net.U*=c; net.W*=c
    for rtol in [1e-9,1e-7,1e-5]:
        A,q,lam,res,h=net.kkt(rtol)
        print(rtol,len(A),'res',res,'lam',lam,'expected',net.nu*h/(2*net.cost()),'nnz',(q>1e-12).sum())
    np.savez(sys.argv[2],U=net.U,W=net.W,n=n,p=p)
