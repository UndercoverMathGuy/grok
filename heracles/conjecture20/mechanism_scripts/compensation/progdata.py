# Programme maximisers y*, binding sets and KKT dual measures for n=2 (nu=3).  Uses the repo solvers prog3/prog2/progx (read-only import).
import numpy as np, sys, json, itertools
sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'..'))
from prog2 import kappa, solve as solve2
from prog3 import solve as solve3
from scipy.optimize import linprog
def exhaustive(p,s,kmax):
    r=1/s; K=kappa(p); F=K.shape[0]; best=(np.inf,None)
    for k in range(1,kmax+1):
        for S in itertools.combinations(range(F),k):
            KS=K[:,S]
            for B in itertools.combinations(range(F),k):
                M=KS[list(B),:]
                if abs(np.linalg.det(M))<1e-12: continue
                y=np.linalg.solve(M,np.ones(k))
                if np.any(y<-1e-12) or np.any(KS@y<1-1e-9): continue
                Q=np.sum(np.clip(y,0,None)**r)
                if Q<best[0]: best=(Q,(S,y))
    Q,(S,y)=best; yy=np.zeros(F); yy[list(S)]=y
    return Q**(-s), yy/Q**s
def dual(p,s,y,B):
    # lambda>=0 on B_half (sum 1) with sum_d lambda_d kappa_j(d) = c * y_j^{r-1} for j in S (KKT); returns lambda, residual
    r=1/s; K=kappa(p); S=np.nonzero(y)[0]
    g=y[S]**(r-1); A=K[np.ix_(B,S)].T   # |S| x |B|
    # min ||A lam - c g||  s.t. lam>=0, sum lam=1, c free : LP in l1
    nB=len(B); nS=len(S)
    # variables lam (nB), c (1), e+ e- (nS)
    cvec=np.concatenate([np.zeros(nB+1),np.ones(2*nS)])
    Aeq=np.hstack([A,-g[:,None],np.eye(nS),-np.eye(nS)]); beq=np.zeros(nS)
    Aeq=np.vstack([Aeq,np.concatenate([np.ones(nB),[0],np.zeros(2*nS)])]); beq=np.append(beq,1)
    res=linprog(cvec,A_eq=Aeq,b_eq=beq,bounds=[(0,None)]*nB+[(None,None)]+[(0,None)]*(2*nS),method='highs')
    return res.x[:nB],res.fun
def get(p,s,exh=None):
    G2,y2,_=solve2(p,s,restarts=300,seed=1)
    G3,y3,_=solve3(p,s,starts=40,seed=2)
    G,y=(G2,y2) if G2>=G3 else (G3,y3)
    src='prog2' if G2>=G3 else 'prog3'
    if exh:
        Ge,ye=exhaustive(p,s,exh)
        if Ge>G*(1+1e-9): G,y,src=Ge,ye,'exh'
    K=kappa(p); cov=K@y; G=cov.min()
    B=np.nonzero(cov<G*(1+1e-7))[0]
    lam,res=dual(p,s,y,B)
    return dict(p=p,s=s,Gamma=float(G),S=[int(j+1) for j in np.nonzero(y)[0]],y=[float(v) for v in y[y>0]],
                Bhalf=[int(d+1) for d in B],lam=[float(v) for v in lam],kkt_res=float(res),src=src,G2=float(G2),G3=float(G3))
if __name__=="__main__":
    s=float(sys.argv[1]); out={}
    for p in [int(a) for a in sys.argv[2:]]:
        exh={5:2,7:3,11:4,13:4,17:4,19:4,23:4}.get(p)
        d=get(p,s,exh); out[p]=d
        print(json.dumps(d),flush=True)
    json.dump(out,open(f'prog_s{s}.json','w'),indent=1)
