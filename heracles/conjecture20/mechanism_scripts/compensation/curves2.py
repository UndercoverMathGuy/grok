# Protection curves Pi_T(rho)/a_l for T={1} (riding+partner) and T={1,2} (+stray), all programme features.
import json,numpy as np,sys,time
from game2 import Game, family, has_shared
s=float(sys.argv[1]); D=[json.loads(l) for l in open(sys.argv[2])]
variant=sys.argv[3] if len(sys.argv)>3 else 'base'   # base | pfree (own partner free) | shared (F2 free vectors)
rmax=3**(s-1); rhos=np.concatenate([[0.005,0.01,0.02,0.05,0.1],np.arange(0.15,rmax,0.05),[rmax]])
res={}; t0=time.time()
for d in D:
    p=d['p']; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    for l in S:
        kw=dict(partner_free=(variant=='pfree'),shared=(variant=='shared'))
        g1=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=False,**kw)); g2=Game(p,S,d['Bhalf'],l,a,family(p,S,l,T12=True,**kw))
        h1=g1.slope(); h12=g2.slope()
        P1=[g1.Pi(r*a[l])/a[l] for r in rhos]; P2=[g2.Pi(r*a[l])/a[l] for r in rhos]
        res[f"{p},{l}"]=dict(p=p,l=l,h1=h1,h12=h12,rho=list(rhos),Pi1=P1,Pi12=P2,shared=has_shared(p,S,l),a=a[l])
        print(f"p={p} l={l} h1={h1:.4f} h12={h12:.4f} Pi12/rho^2(0.005,0.01,0.02)={P2[0]/rhos[0]**2:.4f},{P2[1]/rhos[1]**2:.4f},{P2[2]/rhos[2]**2:.4f} "
              f"Pi1/rho@1={np.interp(1,rhos,P1):.4f} Pi12/rho@1={np.interp(1,rhos,P2):.4f} shared={has_shared(p,S,l)} t={time.time()-t0:.0f}",flush=True)
json.dump(res,open(f'curves2_s{s}_{variant}.json','w'))
