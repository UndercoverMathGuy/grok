# Dedicated compensators (all factors unpaid: concave cost), on top of riding/partner/stray (Lorentz costs).
# Tests: T1 (character chi_{l,{1}}, sign-flipped size Rs) and T12 (chi_{l,{1,2}}, size Rt); bonus gain = absolute value.
# Primal with mu = symmetric probability on B, value = max_tests t_test + Ded;  network wins iff value < 0.
# Dedicated neuron/block at unsigned frequency k carries modes (C_k at c_k, C'_k at c_{-k}) at one test character;
#   budget q = (f * ||(C,C')||)^r, f=1 single neuron (model a: siblings uncancelled -> sibling tests), f=3^{s-1} block (model b).
#   Ded = 3^{1-s} Gamma (1-(1-sum q)^s), handled by CCP (tangent majorisation) from several starts.
import numpy as np, cvxpy as cp
from game2 import family
def run(p,s,S,a,l,Bhalf,Gamma,Rs,Rt,gain,model='b',kset=None,starts=(0.0,0.02,0.05,0.1),iters=25,seed=0,pfree=False,joint=False,cov=None,cpool=None):
    r=1/s; rho=np.exp(2j*np.pi/p); F=(p-1)//2; rng=np.random.default_rng(seed)
    if kset is None: kset=list(range(1,F+1))
    f=1.0 if model=='a' else 3**(s-1)
    Bh=np.array(Bhalf)
    def cv(m,d): return 1-rho**(-m*d)
    def cm(x,c): return cp.hstack([np.real(c)*x[0]-np.imag(c)*x[1],np.real(c)*x[1]+np.imag(c)*x[0]])
    kap=lambda j,d:1-np.cos(2*np.pi*j*d/p)
    best=(np.inf,None)
    for st in starts:
        w0=None if st==0 else {(T,k):st*a[l]*rng.standard_normal(4) for T in ('1','12') for k in kset}
        val_prev=np.inf
        for it in range(iters if w0 is not None else 1):
            cons=[]; tests=[]; qn={}
            tauJ={q:cp.Variable() for q in S}; zJ={q:[] for q in S}
            Cvar={(T,k):cp.Variable(4) for T in ('1','12') for k in kset} if w0 is not None else {}
            for T,R in (('1',Rs),('12',Rt)):
                fam=family(p,S,l,T12=(T=='12'),partner_free=pfree)
                z={(m,q):cp.Variable(2) for (m,q) in fam}
                tau={q:cp.Variable() for q in S} if not joint else tauJ
                for q in S:
                    zs=[z[(m,qq)] for (m,qq) in fam if qq==q]
                    if joint: zJ[q]+=zs; continue
                    if zs: cons.append(cp.norm(cp.hstack(zs+[a[q]-tau[q]]))<=a[q])
                    else: cons.append(tau[q]==0)
                t=cp.Variable(); tests.append(t)
                for d in Bh:
                    Phi=0
                    for dd in (d,p-d):
                        terms=[np.array([np.real(R*cv(l,dd)),np.imag(R*cv(l,dd))])]
                        terms+=[cm(z[(m,q)],cv(m,dd)) for (m,q) in fam]
                        for k in kset:
                            if (T,k) in Cvar:
                                terms+=[cm(Cvar[(T,k)][:2],cv(k,dd)),cm(Cvar[(T,k)][2:],cv(-k,dd))]
                        Phi=Phi+cp.norm(0.5*sum(terms))+sum(kap(q,dd)*tau[q] for q in S)-kap(l,dd)*gain
                    cons.append(t>=0.5*Phi+(0 if cov is None else cov[int(d)]))
                    if model=='a':
                        for k in kset:
                            if (T,k) in Cvar:
                                for half in (Cvar[(T,k)][:2],Cvar[(T,k)][2:]):
                                    ts=cp.Variable(); tests.append(ts)
                                    Phis=sum(0.5*cp.norm(half)*abs(cv(k,dd))+sum(kap(q,dd)*tau[q] for q in S)-kap(l,dd)*gain for dd in (d,p-d))
                                    cons.append(ts>=0.5*Phis)
            if joint:
                for q in S:
                    if zJ[q]: cons.append(cp.norm(cp.hstack(zJ[q]+[a[q]-tauJ[q]]))<=a[q])
                    else: cons.append(tauJ[q]==0)
            M=cp.Variable(); cons+=[M>=t_ for t_ in tests]
            if Cvar and cpool is not None:
                cons.append(sum(cp.norm(Cvar[key]) for key in Cvar)<=cpool)
                Qp=min((f*cpool)**r,0.999999); ded=3**(1-s)*Gamma*(1-(1-Qp)**s)
            elif Cvar:
                q0={key:max(f*np.linalg.norm(w0[key]),1e-9) for key in Cvar}
                Q0=min(sum(v**r for v in q0.values()),0.999)
                D0=3**(1-s)*Gamma*(1-(1-Q0)**s); Dp=3**(1-s)*Gamma*s*(1-Q0)**(s-1)
                ded=D0+Dp*sum(r*q0[key]**(r-1)*(f*cp.norm(Cvar[key])-q0[key]) for key in Cvar)
            else: ded=0
            prob=cp.Problem(cp.Minimize(M+ded),cons)
            try: prob.solve(solver='CLARABEL')
            except Exception:
                try: prob.solve(solver='SCS',eps=1e-8,max_iters=50000)
                except Exception: break
            if M.value is None: break
            if Cvar and cpool is not None:
                true=M.value+ded; Q=(f*cpool)**r
                if true<best[0]: best=(true,dict(Q=Q,start=st,it=it))
                break
            if Cvar:
                w={key:Cvar[key].value for key in Cvar}
                Q=sum((f*np.linalg.norm(w[key]))**r for key in Cvar)
                true=M.value+3**(1-s)*Gamma*(1-(1-min(Q,0.999999))**s)
                w0=w
            else: true=M.value
            if true<best[0]: best=(true,dict(Q=(Q if Cvar else 0),start=st,it=it))
            if abs(val_prev-true)<1e-10: break
            val_prev=true
    return best
