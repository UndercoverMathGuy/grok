# Pi^{L2} on a coarse (rho_s, rho_12) grid (general quad test, f = f_min, all three T) and the single-T (T12, f=1) curve.
import json,numpy as np,sys,time
from gameL2 import GameL2
s=float(sys.argv[1]); D=[json.loads(l) for l in open(sys.argv[2])]; variant=sys.argv[3] if len(sys.argv)>3 else 'base'
rmax=3**(s-1)
ax=np.unique(np.round(np.concatenate([[0,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.1,1.3,1.5],[rmax]]),4)); ax=ax[ax<=rmax+1e-9]
ax1=np.unique(np.round(np.concatenate([[0,0.02,0.05,0.1,0.15,0.2],np.arange(0.3,rmax,0.1),[rmax]]),4))
res={}; t0=time.time()
kw=dict(partner_free=(variant=='pfree'),shared=(variant=='shared'))
for d in D:
    p=d['p']; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    for l in S:
        g=GameL2(p,S,d['Bhalf'],l,a,**kw); g1=GameL2(p,S,d['Bhalf'],l,a,f=1.0,Ts=('12',),**kw)
        grid={}
        for rs in ax:
            for rt in ax:
                if rs==0 and rt==0: continue
                grid[f"{rs},{rt}"]=g.Pi({'1':rs*a[l],'2':rs*a[l],'12':rt*a[l]})/a[l]
        single={str(rt):(g1.Pi({'12':rt*a[l]})/a[l] if rt>0 else 0.0) for rt in ax1}
        res[f"{p},{l}"]=dict(p=p,l=l,grid=grid,single=single)
        print(f"p={p} l={l} done t={time.time()-t0:.0f}",flush=True)
        json.dump(dict(ax=list(ax),ax1=list(ax1),res=res),open(f'curvesL2_s{s}_{variant}.json','w'))
