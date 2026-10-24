# Task 2: allocation moves.  sigma(y) = min_grid max(Pi_1^y(rs a_l), Pi_12^y(rt a_l)) / (a_l(y) g*(rs,rt)),
# Pi_T^y(R) = max{ cov_y(mu) + Prot_T(mu,R) : mu>=0 symmetric on ALL directions, K_l(mu)=1 },  cov_y = 3^{1-s}(Gamma sum mu - sum_j y_j K_j(mu))
import json,numpy as np,glob,sys
from game2 import Game, family
def load_grid(s):
    G={}; ax=None
    for f in glob.glob(f'grid_s{s}_part*.json'):
        d=json.load(open(f)); ax=d['ax']; G.update(d['G'])
    return np.array(ax),G
def sigma_y(p,s,S,yv,l,Gamma,ax,G,mu_dir_half=None,Dhalf=None,variant='base'):
    F=(p-1)//2; Dh=list(range(1,F+1)) if Dhalf is None else Dhalf
    B=np.array(sorted(set(Dh)|{p-d for d in Dh}))
    y=dict(zip(S,yv)); a={j:3**(1-s)*y[j] for j in S}
    cov=3**(1-s)*(Gamma-sum(y[j]*(1-np.cos(2*np.pi*j*B/p)) for j in S))
    mu_dir=None
    if mu_dir_half is not None:
        mu_dir=np.array([mu_dir_half.get(min(d,p-d),0.0) for d in B])
    kw=dict(partner_free=(variant=='pfree'),shared=(variant=='shared'))
    g1=Game(p,S,Dh,l,a,family(p,S,l,T12=False,**kw),mu_dir=mu_dir,cov=cov)
    g2=Game(p,S,Dh,l,a,family(p,S,l,T12=True,**kw),mu_dir=mu_dir,cov=cov)
    P1=np.array([0.0]+[g1.Pi(r*a[l]) for r in ax[1:]]); P2=np.array([0.0]+[g2.Pi(r*a[l]) for r in ax[1:]])
    best=(np.inf,None)
    for i,rs in enumerate(ax):
        for k,rt in enumerate(ax):
            g=G.get(f"{rs},{rt}")
            if g is None or g<=1e-7: continue
            v=max(P1[i],P2[k])/(a[l]*g)
            if v<best[0]: best=(v,(rs,rt))
    return best,P1,P2
