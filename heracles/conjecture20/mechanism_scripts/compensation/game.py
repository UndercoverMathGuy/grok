# Single-cheap-feature compensation game, n=2 (nu=3).  Dual SOCP formulation.
#   Prot(R,mu) = min_{b,z,z'} sum_d mu_d |ghat_d| + K_l (a_l - sqrt(a_l^2-|b|^2)) + sum_{j!=l} K_j (a_j - sqrt(a_j^2-|z_j|^2-|z'_j|^2))
#   Pi(R)      = max { Prot(R,mu) : mu >= 0 symmetric on B, K_l(mu) = 1 }       (Prot is 1-homogeneous in mu)
#   dual: Pi(R) = max_{mu,u: |u_d|<=mu_d} R Re N_l + a_l (K_l - ||(K_l,P_l)||) + sum_{j!=l} a_j (K_j - ||(K_j,N_j,P_j)||)
#   P_j = 1/2 sum u_d c_j(d), N_j = 1/2 sum u_d conj c_j(d), c_j(d) = 1 - rho^{-j d}.
#   Safety:  network cannot win with this cheap feature iff Pi(R) >= G(R) for all 0 < R <= 3^{s-1} a_l.
#   Free-hiding slope h_l = lim_{R->0} Pi(R)/R = max Re N_l s.t. K_l = 1, compensated coefficients = 0.
import numpy as np, cvxpy as cp, json, sys
def setup(p,S,Bhalf):
    B=np.array(sorted(set(Bhalf)|{p-d for d in Bhalf}))
    th=lambda j:2*np.pi*j*B/p
    kap={j:1-np.cos(th(j)) for j in S}; sn={j:np.sin(th(j)) for j in S}
    return B,kap,sn
def build(p,S,Bhalf,l,a,mode='full',R=None,mu_fixed=None):
    """returns problem pieces; mode in {'full','riding','partner','none'}; R=None -> free-hiding slope problem"""
    B,kap,sn=setup(p,S,Bhalf); nB=len(B)
    idx={d:i for i,d in enumerate(B)}; pair=[(idx[d],idx[p-d]) for d in Bhalf]
    mu=cp.Variable(nB,nonneg=True); ur=cp.Variable(nB); ui=cp.Variable(nB)
    cons=[cp.SOC(mu,cp.vstack([ur,ui]),axis=0)]
    cons+=[mu[i]==mu[k] for i,k in pair if i!=k]
    if mu_fixed is not None: cons+=[mu==mu_fixed]
    K={j:kap[j]@mu for j in S}
    # c_j = kap + i sn ; conj c_j = kap - i sn ; u = ur + i ui
    def P(j): return 0.5*(kap[j]@ur-sn[j]@ui), 0.5*(kap[j]@ui+sn[j]@ur)   # 1/2 sum u c
    def N(j): return 0.5*(kap[j]@ur+sn[j]@ui), 0.5*(kap[j]@ui-sn[j]@ur)   # 1/2 sum u conj c
    obj=N(l)[0]
    pen=0
    if mu_fixed is None: cons+=[K[l]==1]
    useP= mode in ('full','partner'); useR= mode in ('full','riding')
    if R is None:
        if useP: cons+=[P(l)[0]==0,P(l)[1]==0]
        if useR:
            for j in S:
                if j==l: continue
                cons+=[N(j)[0]==0,N(j)[1]==0,P(j)[0]==0,P(j)[1]==0]
        prob=cp.Problem(cp.Maximize(obj),cons)
    else:
        terms=[R*obj]
        if useP: terms.append(a[l]*(K[l]-cp.norm(cp.hstack([K[l],*P(l)]))))
        if useR:
            for j in S:
                if j==l: continue
                terms.append(a[j]*(K[j]-cp.norm(cp.hstack([K[j],*N(j),*P(j)]))))
        prob=cp.Problem(cp.Maximize(sum(terms)),cons)
    return prob,mu,ur,ui,B
def solve(prob):
    try: prob.solve(solver='CLARABEL')
    except Exception:
        prob.solve(solver='SCS',eps=1e-9,max_iters=200000)
    return prob.value
def slope(p,S,Bhalf,l,a,mode='full',mu_fixed=None):
    prob,mu,ur,ui,B=build(p,S,Bhalf,l,a,mode,None,mu_fixed)
    v=solve(prob)
    if mu_fixed is not None:
        Kl=np.sum(mu_fixed*(1-np.cos(2*np.pi*l*B/p))); return v/Kl, mu_fixed
    return v, mu.value
def Pi(p,S,Bhalf,l,a,R,mode='full'):
    prob,mu,ur,ui,B=build(p,S,Bhalf,l,a,mode,R)
    return solve(prob)
