# L^2-over-T (quadratic Riesz) version of the compensation game, n=2.
# Detection at delta: f*sqrt(2)*|| (ghat_{delta,T})_{T in {1},{2},{12}} ||, f = 1 (single T) or f(gamma_max)=0.6724 (general)
# ghat_{delta,T} = 1/2 ( R_T c_l + sum_{(m,q) in fam_T} z_{m,q,T} c_m ),  fam_T: riding+partner (+ stray for T=12)
# Costs: Lorentz per payer q over ALL its modes at all T.   Dual:
#   PiL2(R) = max_{mu,u} sum_T R_T Re Y_{l,T} - sum_q a_q(||(K_q, Y_{m,T})|| - K_q),  ||(u_{d,T})_T|| <= f sqrt2 mu_d, K_l(mu)=1
import numpy as np, cvxpy as cp
from game2 import family
FMIN=(np.sqrt(14)-np.sqrt(2))/(2*np.sqrt(3))
class GameL2:
    def __init__(self,p,S,Dhalf,l,a,f=FMIN,Ts=('1','2','12'),cov=None,**kw):
        B=np.array(sorted(set(Dhalf)|{p-d for d in Dhalf})); nB=len(B); idx={d:i for i,d in enumerate(B)}
        mu=cp.Variable(nB,nonneg=True)
        U={T:(cp.Variable(nB),cp.Variable(nB)) for T in Ts}
        cons=[cp.SOC(f*np.sqrt(2)*mu,cp.vstack([c for T in Ts for c in U[T]]),axis=0)]
        cons+=[mu[idx[d]]==mu[idx[p-d]] for d in Dhalf if idx[d]!=idx[p-d]]
        K={j:(1-np.cos(2*np.pi*j*B/p))@mu for j in S}; cons.append(K[l]==1)
        def Y(m,T):
            t=2*np.pi*m*B/p; al=1-np.cos(t); be=np.sin(t); ur,ui=U[T]
            return [0.5*(al@ur+be@ui),0.5*(al@ui-be@ur)]
        pay={q:[] for q in S}
        for T in Ts:
            for (m,q) in family(p,S,l,T12=(T=='12'),**kw):
                if q is None: cons+=[y==0 for y in Y(m,T)]
                else: pay[q]+=Y(m,T)
        self.R={T:cp.Parameter(nonneg=True) for T in Ts}
        costs=[a[q]*(cp.norm(cp.hstack([K[q]]+pay[q]))-K[q]) for q in S if pay[q]]
        covterm=0 if cov is None else cov@mu
        self.prob=cp.Problem(cp.Maximize(covterm+sum(self.R[T]*Y(l,T)[0] for T in Ts)-sum(costs)),cons)
        self.Ts=Ts
    def Pi(self,RT):
        for T in self.Ts: self.R[T].value=RT[T]
        try: self.prob.solve(solver='CLARABEL')
        except Exception: self.prob.solve(solver='SCS',eps=1e-9,max_iters=200000)
        return self.prob.value
