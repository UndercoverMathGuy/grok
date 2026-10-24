# General compensation game at a sign-flipped character chi_{l,T}, n=2.
# Compensation family: list of (m, payer) with m a signed residue mod p (vector c_m(d) = 1 - rho^{-m d}), payer in S (Lorentz
# cost charged to that feature's clean amplitude, weighted by K_payer(mu)) or None (free).
#   riding:  (j,j),(-j,j) for j in S\{l}          [input 2l-+j on T in feature-j neurons]
#   partner: (-l,l)                                [input 3l on T in feature-l neurons]
#   stray:   (2l-q,q),(2l+q,q) for q in S, T=[n]   [output m on feature-q neurons]   (m != l, m != 0)
#   shared (F2, free): m=+-j with 2l = +-j +- q, j != q both in S
# Dual:  Pi(R) = max_{mu,u} R Re Y_l - sum_q a_q (||(K_q, Y_m : (m,q))|| - K_q)   s.t. K_l(mu)=1, |u_d|<=mu_d, Y_m = 0 for free m
#        Y_m = 1/2 sum_d u_d conj(c_m(d))
import numpy as np, cvxpy as cp
def family(p,S,l,T12=False,partner=True,riding=True,stray=True,shared=False,partner_free=False):
    fam=[]
    sm=lambda m: int(m%p)
    if riding:
        for j in S:
            if j!=l: fam+=[(sm(j),j),(sm(-j),j)]
    if partner: fam.append((sm(-l),None if partner_free else l))
    if T12 and stray:
        for q in S:
            for m in (2*l-q,2*l+q):
                m=sm(m)
                if m!=0 and m!=sm(l): fam.append((m,q))
    if shared:
        Sp=set(sm(j) for j in S)|set(sm(-j) for j in S)
        for j in S:
            for q in S:
                if j==q: continue
                for sj in (1,-1):
                    for sq in (1,-1):
                        if sm(2*l-sj*j-sq*q)==0: fam.append((sm(sj*j),None))
    # dedupe (same vector & payer)
    out=[]
    for f in fam:
        if f not in out: out.append(f)
    return out
def has_shared(p,S,l):
    return len([f for f in family(p,S,l,partner=False,riding=False,shared=True)])>0
class Game:
    def __init__(self,p,S,Dhalf,l,a,fam,mu_dir=None,cov=None):
        self.p=p; self.S=S; self.l=l; self.a=a
        B=np.array(sorted(set(Dhalf)|{p-d for d in Dhalf})); self.B=B; nB=len(B)
        idx={d:i for i,d in enumerate(B)}
        self.mu=mu=cp.Variable(nB,nonneg=True); ur=cp.Variable(nB); ui=cp.Variable(nB)
        cons=[cp.SOC(mu,cp.vstack([ur,ui]),axis=0)]
        cons+=[mu[idx[d]]==mu[idx[p-d]] for d in Dhalf if idx[d]!=idx[p-d]]
        K={j:(1-np.cos(2*np.pi*j*B/p))@mu for j in S}
        def Y(m):  # 1/2 sum u conj(c_m),  c_m = 1-cos(t) + i sin(t), t=2 pi m d/p
            t=2*np.pi*m*B/p; al=1-np.cos(t); be=np.sin(t)
            # u*conj(c) = (ur+i ui)(al - i be) = ur al + ui be + i(ui al - ur be)
            return [0.5*(al@ur+be@ui),0.5*(al@ui-be@ur)]
        self.Yl=Y(l)
        cons.append(K[l]==1)
        if mu_dir is not None:
            tt=cp.Variable(nonneg=True); cons.append(mu==tt*mu_dir)
        self.covterm=0 if cov is None else cov@mu
        pay={q:[] for q in S}; free=[]
        for m,q in fam:
            if q is None: free.append(m)
            else: pay[q].append(m)
        for m in free: cons+=[y==0 for y in Y(m)]
        self.Rpar=cp.Parameter(nonneg=True)
        costs=[]
        for q in S:
            if pay[q]:
                comps=[K[q]]+[c for m in pay[q] for c in Y(m)]
                costs.append(a[q]*(cp.norm(cp.hstack(comps))-K[q]))
        self.prob_R=cp.Problem(cp.Maximize(self.covterm+self.Rpar*self.Yl[0]-(sum(costs) if costs else 0)),cons)
        cons_s=list(cons)+[y==0 for q in S for m in pay[q] for y in Y(m)]
        self.prob_h=cp.Problem(cp.Maximize(self.Yl[0]),cons_s)
    def slope(self):
        self.prob_h.solve(solver='CLARABEL'); return self.prob_h.value
    def Pi(self,R):
        self.Rpar.value=R
        try: self.prob_R.solve(solver='CLARABEL')
        except Exception: self.prob_R.solve(solver='SCS',eps=1e-9,max_iters=200000)
        return self.prob_R.value
