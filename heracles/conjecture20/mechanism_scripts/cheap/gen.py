# Task 3: general (not slot-symmetric) networks on all x in Z_p^n, outputs on all frequencies by SOCP with l_r reweighting.
# Efficiency relative to gamma_IF = nu^{1-s} B_p Gamma_s ; reports feature amplitudes and cheapness |beta_l| vs nu^{1-s} Q_l^s.
import math, sys, time, itertools, numpy as np, cma, warnings, cvxpy as cp
sys.path.insert(0,'/home/user/grok/heracles/conjecture20'); sys.path.insert(0,'/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/cu')
from famp import log_gamma_if
warnings.filterwarnings("ignore")
class Gen:
    def __init__(self,p,n):
        self.p,self.n,self.nu=p,n,n+1
        self.X=np.array(list(itertools.product(range(p),repeat=n))); self.y=self.X.sum(1)%p
        cs=np.arange(p); Bs=[]
        for l in range(1,(p-1)//2+1):
            for f in (np.cos,np.sin):
                v=f(2*math.pi*l*cs/p); Bs.append(v/np.linalg.norm(v))
        self.B=np.array(Bs).T
        self.D=[self.B[self.y]-self.B[(self.y-d)%p] for d in range(1,p)]
        import os
        self.lgif=log_gamma_if(p,n,2.0) if p<=19 else (1-(n+1)/2)*math.log(n+1)+math.log(2)-((n+1)/2)*math.log(2*p*(n+1))+math.log(float(os.environ['GAMMA']))
    def prods(self,U):   # U: n x m x p (unit-normalised per row)
        Un=U/np.linalg.norm(U,axis=2,keepdims=True)
        G=np.ones((len(self.X),U.shape[1]))
        for t in range(self.n): G=G*Un[t][:,self.X[:,t]].T
        return G
    def solve(self,U,rew=3,ret=False):
        p,nu=self.p,self.nu; r=2/nu; m=U.shape[1]; k=p-1
        G=self.prods(U); sc=1.0
        A=np.vstack([(G[:,:,None]*D[:,None,:]).reshape(len(self.y),-1) for D in self.D])
        Z=cp.Variable((m,k)); t=cp.Variable(m); h=cp.Variable(); w=cp.Parameter(m,nonneg=True)
        prob=cp.Problem(cp.Maximize(h),[A@cp.reshape(Z,(m*k,),order='C')>=h,cp.norm(Z,2,axis=1)<=t,w@t<=1])
        wv=np.ones(m); best=(-np.inf,None)
        for it in range(rew+1):
            w.value=wv
            try: prob.solve(solver=cp.CLARABEL)
            except Exception: break
            if Z.value is None: break
            Zv=Z.value; W=Zv@self.B.T; f=G@W
            gaps=np.array([f[np.arange(len(self.y)),self.y]-f[np.arange(len(self.y)),(self.y-d)%p] for d in range(1,p)])
            ht=gaps.min(); a=np.linalg.norm(Zv,axis=1)
            if ht>0:
                keep=a>1e-9*a.max(); P=(nu*a[keep]**(2/nu)).sum()
                e=math.exp(math.log(ht)-(nu/2)*math.log(P)-self.lgif)
                if e>best[0]: best=(e,(U,W,a))
            wv=(a/max(a.max(),1e-300)+1e-4)**(r-1); wv=wv/wv.min()
        return best
    def analyse(self,U,W,a):
        # per-neuron clean coefficients beta_il = 2 p^{-nu} sum_{x,c} f_i(x)[c] rho^{-l(y-c)}, neuron i has output norm a_i, unit inputs
        p,nu,n=self.p,self.nu,self.n; G=self.prods(U); out=[]
        for l in range(1,(p-1)//2+1):
            ph=np.exp(-2j*math.pi*l*(self.y[:,None]-np.arange(p)[None,:])/p)     # |X| x p
            bil=np.array([2*p**(-nu)*np.sum(np.outer(G[:,i],W[i])*ph) for i in range(len(W))])
            bl=abs(bil.sum()); Q=np.sum(np.abs(bil)**(2/nu)); al=nu**(1-nu/2)*Q**(nu/2)
            out.append((l,bl,al))
        return out
def ifseed(p,n,support,extra,rng,noise):
    nu=n+1; a=np.arange(p); rows=[[] for _ in range(n)]
    for j in support:
        for l in range(nu):
            for t in range(n): rows[t].append(np.cos(2*math.pi*j*a/p+math.pi*l/nu))
    for e in range(extra):
        for t in range(n): rows[t].append(rng.standard_normal(p))
    U=np.array(rows)
    return U+noise*rng.standard_normal(U.shape)
if __name__=="__main__":
    from prog import solve as psolve
    p,n=int(sys.argv[1]),int(sys.argv[2]); s=(n+1)/2; t0=time.time()
    G,y,cov=psolve(p,s,restarts=30); S=list(np.nonzero(y)[0]+1)
    if 1-math.cos(2*math.pi/p)>G+1e-9: S=[1]; y=np.zeros((p-1)//2); y[0]=1.0
    print(f"p={p} n={n} programme support {S} shares {np.round(y[np.array(S)-1],4)}",flush=True)
    g=Gen(p,n); rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 0)
    best=(-1,None,None)
    F=(p-1)//2
    trials=[(tuple(S),0,0.0),(tuple(S),2,0.1),(tuple(S),n+1,0.2)]
    if len(S)<F: trials+=[(tuple(range(1,F+1)),0,0.1)]
    for supp,extra,noise in trials:
        for rep in range(2):
            U0=ifseed(p,n,supp,extra,rng,noise); sh=U0.shape
            v0=g.solve(U0,rew=4)
            bestloc=v0
            es=cma.CMAEvolutionStrategy(U0.ravel(),0.15,{'seed':rep+1+extra,'verbose':-9,'popsize':14,'maxiter':int(sys.argv[4]) if len(sys.argv)>4 else 40})
            while not es.stop():
                X=es.ask(); Fv=[]
                for x in X:
                    v=g.solve(np.array(x).reshape(sh),rew=1); Fv.append(v[0] if v[0]>-np.inf else -1)
                    if v[0]>bestloc[0]: bestloc=v
                es.tell(X,[-f for f in Fv])
            if bestloc[1] is not None:
                v=g.solve(bestloc[1][0],rew=6); bestloc=v if v[0]>bestloc[0] else bestloc
                U,W,a=bestloc[1]; an=g.analyse(U,W,a)
                feats=[(l,round(bl/max(x[1] for x in an),4),round(bl/al,4) if al>0 else None) for (l,bl,al) in an for x in [ (0,1) ] ]
                print(f"  seed supp={supp} extra={extra} noise={noise} rep={rep}: eff={bestloc[0]:.5f} (IF seed {v0[0]:.5f}); per freq (l, rel amp, |beta|/IF-price): {feats}  ({time.time()-t0:.0f}s)",flush=True)
                if bestloc[0]>best[0]: best=(bestloc[0],bestloc[1],an)
    print(f"p={p} n={n} BEST eff {best[0]:.6f}",flush=True)
    np.save(f"gen_p{p}_n{n}.npy",np.concatenate([best[1][0].reshape(-1),best[1][1].ravel()]))
