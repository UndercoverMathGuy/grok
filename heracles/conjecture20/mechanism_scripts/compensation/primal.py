# Primal form of the compensation game (mu = symmetric probability on a direction set D):
#   V(R) = min_{b,z,z',C} max_{d in Dhalf} 1/2(Phi_d+Phi_{-d}) + Ded(C) + Cov_d
#   Phi_d = |ghat_d| + sum_{j in S} kappa_j(d) cost_j - kappa_l(d) * gain
#   ghat_d = 1/2 [ R c_l + b conj c_l + sum_{j!=l} (z_j c_j + z'_j conj c_j) + sum_{k ded} (C_k c_k + C'_k conj c_k) ]
#   riding cost_j = a_j - sqrt(a_j^2 - |z_j|^2 - |z'_j|^2), partner cost_l = a_l - sqrt(a_l^2-|b|^2)
#   dedicated: one neuron per frequency k in ded, budget q_k = (|C_k|^2+|C'_k|^2)^{r/2};  Ded = 3^{1-s} Gamma (1-(1-sum q)^s)  (concave; CCP)
#   Cov_d = 3^{1-s}(Gamma - sum_j y_j kappa_j(d))   (zero on the binding set at y = y*; used for allocation moves)
import numpy as np, cvxpy as cp
def solve_primal(p,s,S,a,l,R,gain,Dhalf,Gamma=None,ded=(),cov=None,w0=None,iters=30,partner=True,riding=True):
    r=1/s; rho=np.exp(2j*np.pi/p)
    Dh=np.array(Dhalf); nD=len(Dh)
    def cvec(j,d): return 1-rho**(-j*d)
    # complex vector helpers: represent ghat_d for d and -d
    zr={j:cp.Variable(2) for j in S if j!=l}; zpr={j:cp.Variable(2) for j in S if j!=l}
    b=cp.Variable(2); tau={j:cp.Variable() for j in S}
    Cv={k:cp.Variable(2) for k in ded}; Cpv={k:cp.Variable(2) for k in ded}
    cons=[]
    def cmul(x,c):  # x = (re,im) variable, c complex const -> (re,im) affine
        return cp.hstack([np.real(c)*x[0]-np.imag(c)*x[1], np.real(c)*x[1]+np.imag(c)*x[0]])
    def ghat(d):
        e=np.array([np.real(R*cvec(l,d)),np.imag(R*cvec(l,d))])
        terms=[e]
        if partner: terms.append(cmul(b,np.conj(cvec(l,d))))
        if riding:
            for j in zr: terms+= [cmul(zr[j],cvec(j,d)), cmul(zpr[j],np.conj(cvec(j,d)))]
        for k in ded: terms+= [cmul(Cv[k],cvec(k,d)), cmul(Cpv[k],np.conj(cvec(k,d)))]
        return 0.5*sum(terms)
    # riding / partner costs (epigraph tau_j >= a_j - sqrt(a_j^2 - |.|^2))
    for j in S:
        if j==l:
            if partner: cons.append(cp.norm(cp.hstack([b,a[l]-tau[l]]))<=a[l])
            else: cons.append(tau[l]==0)
        else:
            if riding: cons.append(cp.norm(cp.hstack([zr[j],zpr[j],a[j]-tau[j]]))<=a[j])
            else: cons.append(tau[j]==0)
    # dedicated (linearised concave cost)
    dedterm=0; qk={}
    if ded:
        q0={k:max(np.linalg.norm(w0[k]),1e-9) for k in ded}; Q0=sum(v**r for v in q0.values())
        Q0=min(Q0,0.999)
        Dp=3**(1-s)*Gamma*s*(1-Q0)**(s-1)
        D0=3**(1-s)*Gamma*(1-(1-Q0)**s)
        lin=sum(r*q0[k]**(r-1)*(cp.norm(cp.hstack([Cv[k],Cpv[k]]))-q0[k]) for k in ded)
        dedterm=D0+Dp*lin
    t=cp.Variable()
    for i,d in enumerate(Dh):
        kap=lambda j,dd:1-np.cos(2*np.pi*j*dd/p)
        Phi=0
        for dd in (d,p-d):
            Phi=Phi+cp.norm(ghat(dd))+sum(kap(j,dd)*tau[j] for j in S)-kap(l,dd)*gain
        cons.append(t>=0.5*Phi+(cov[i] if cov is not None else 0))
    prob=cp.Problem(cp.Minimize(t+dedterm),cons)
    try: prob.solve(solver='CLARABEL')
    except Exception: prob.solve(solver='SCS',eps=1e-9,max_iters=100000)
    w={k:np.concatenate([Cv[k].value,Cpv[k].value]) for k in ded} if ded else None
    return prob.value,t.value,w
def ded_game(p,s,S,a,l,R,gain,Dhalf,Gamma,ded,starts,rng,iters=40,cov=None):
    """CCP local search over dedicated compensators; returns best (true) value"""
    r=1/s; best=np.inf; bestw=None
    for st in starts:
        w0={k:st*rng.standard_normal(4) for k in ded}
        for it in range(iters):
            val,tv,w=solve_primal(p,s,S,a,l,R,gain,Dhalf,Gamma,ded,cov,w0)
            if w is None: break
            Q=sum(np.linalg.norm(w[k])**r for k in ded)
            true=tv+(3**(1-s)*Gamma*(1-(1-min(Q,1))**s))
            if all(np.linalg.norm(w[k]-w0[k])<1e-7*(1+np.linalg.norm(w0[k])) for k in ded): w0=w; break
            w0=w
        if true<best: best=true; bestw=w0
    return best,bestw
