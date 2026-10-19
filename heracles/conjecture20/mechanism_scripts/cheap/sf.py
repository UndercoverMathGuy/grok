# Task 2: slot-symmetric single-feature networks (outputs at frequency 1 only), arbitrary inputs, p=5,7.
# Efficiency = gamma / (nu^{1-s} B_p kappa_min), b=2.  CMA over inputs, exact SOCP over outputs.
import math, sys, time, numpy as np, cma, warnings, cvxpy as cp
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from famp import counts
warnings.filterwarnings("ignore")
class SF:
    def __init__(self,p,n,freqs=(1,)):
        self.p,self.n,self.nu=p,n,n+1
        N,y,mult=counts(n,p); self.N,self.y=N.numpy(),y.numpy()
        cs=np.arange(p); Bs=[]
        for l in freqs:
            for f in (np.cos,np.sin):
                v=f(2*math.pi*l*cs/p); Bs.append(v/np.linalg.norm(v))
        self.B=np.array(Bs).T
        self.D=[self.B[self.y]-self.B[(self.y-d)%p] for d in range(1,p)]
        nu=self.nu; s=nu/2
        self.lref=(1-s)*math.log(nu)+math.log(2)-(nu/2)*math.log(2*p*nu)+math.log(1-math.cos(2*math.pi/p))
    def solve(self,U,rew=2):
        p,nu=self.p,self.nu; r=2/nu; m=len(U); k=self.B.shape[1]
        Un=U/np.linalg.norm(U,axis=1,keepdims=True)
        logP=self.N@np.log(np.abs(Un)+1e-300).T; sgn=(-1.0)**(self.N@(Un<0).T)
        rowmax=logP.max(1); G=sgn*np.exp(logP-rowmax[:,None]); sc=np.exp(-rowmax); sc=sc/sc.max()
        A=np.vstack([(G[:,:,None]*D[:,None,:]).reshape(len(self.y),-1) for D in self.D]); cc=np.concatenate([sc]*(p-1))
        Z=cp.Variable((m,k)); t=cp.Variable(m); h=cp.Variable(); w=cp.Parameter(m,nonneg=True)
        prob=cp.Problem(cp.Maximize(h),[A@cp.reshape(Z,(m*k,),order='C')>=h*cc,cp.norm(Z,2,axis=1)<=t,w@t<=1])
        wv=np.ones(m); best=-np.inf
        for it in range(rew+1):
            w.value=wv
            try: prob.solve(solver=cp.CLARABEL)
            except Exception: break
            if Z.value is None: break
            Zv=Z.value; W=Zv@self.B.T; f=(G*np.exp(rowmax[:,None]))@W
            gaps=np.array([f[np.arange(len(self.y)),self.y]-f[np.arange(len(self.y)),(self.y-d)%p] for d in range(1,p)])
            ht=gaps.min(); a=np.linalg.norm(Zv,axis=1)
            if ht>0:
                keep=a>1e-9*a.max(); P=(nu*a[keep]**(2/nu)).sum()
                best=max(best,math.exp(math.log(ht)-(nu/2)*math.log(P)-self.lref))
            wv=(a/max(a.max(),1e-300)+1e-4)**(r-1); wv=wv/wv.min()
        return best
def run(p,n,m,R,iters,seed=0):
    pr=SF(p,n); rng=np.random.default_rng(seed); best=-1; a=np.arange(p)
    for rr in range(R):
        if rr==0: x0=np.array([np.cos(2*math.pi*a/p+math.pi*l/(n+1)) for l in range(m)]).ravel()
        elif rr==1: x0=np.array([np.cos(2*math.pi*a/p+rng.uniform(0,6.3))+0.4*np.cos(6*math.pi*a/p+rng.uniform(0,6.3))+0.3*rng.standard_normal() for l in range(m)]).ravel()
        else: x0=rng.standard_normal(m*p)
        v0=pr.solve(x0.reshape(m,p),rew=2); best=max(best,v0)
        es=cma.CMAEvolutionStrategy(x0,0.15 if rr<2 else 0.5,{'seed':seed+rr+1,'verbose':-9,'popsize':10+m,'maxiter':iters})
        while not es.stop():
            X=es.ask(); F=[]
            for x in X:
                v=pr.solve(np.array(x).reshape(m,p),rew=1); F.append(v if v>-np.inf else -1.0); best=max(best,v)
            es.tell(X,[-f for f in F])
    return best
if __name__=="__main__":
    p=int(sys.argv[1]); t0=time.time()
    for n in [int(v) for v in sys.argv[2].split(",")]:
        res={}
        for m in range(max(1,n-1),n+4):
            res[m]=run(p,n,m,R=5,iters=120)
        print(f"p={p} n={n}: best single-feature efficiency by width {{{', '.join(f'{m}:{v:.4f}' for m,v in res.items())}}}  ({time.time()-t0:.0f}s)",flush=True)
