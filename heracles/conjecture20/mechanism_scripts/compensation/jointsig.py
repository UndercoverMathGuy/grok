# Joint-cost Riesz game (tests T={1} and T={1,2}, one Lorentz account per payer for all its modes):
#   sigma_joint(rs,rt) = sup{ t : network cannot make both tests negative with gain t*G*(rs,rt) }   (bisection on t)
# evaluated at the grid points where the separate-cost ratio is < thr.
import json,numpy as np,sys,glob,time
from ded import run
s=float(sys.argv[1]); keys=sys.argv[2].split(';'); thr=float(sys.argv[3]) if len(sys.argv)>3 else 1.25
D={json.loads(x)['p']:json.loads(x) for x in open('prog15.out' if s==1.5 else 'prog12.out')}
G={}
for f in glob.glob(f'grid_s{s}_part*.json'): d_=json.load(open(f)); ax=np.array(d_['ax']); G.update(d_['G'])
cur=json.load(open(f'curves2_s{s}_base.json'))
for key in keys:
    p,l=map(int,key.split(',')); d=D[p]; S=d['S']; y=dict(zip(S,d['y'])); a={j:3**(1-s)*y[j] for j in S}
    c=cur[key]; rho=np.concatenate([[0],c['rho']]); P1=np.interp(ax,rho,np.concatenate([[0],c['Pi1']])); P2=np.interp(ax,rho,np.concatenate([[0],c['Pi12']]))
    pts=[]
    for i,rs in enumerate(ax):
        for j,rt in enumerate(ax):
            g=G.get(f"{rs},{rt}")
            if g is None or g<1e-7: continue
            sep=max(P1[i],P2[j])/g
            if sep<thr: pts.append((sep,rs,rt,g))
    pts.sort(); t0=time.time(); best=(np.inf,None)
    for sep,rs,rt,g in pts[:12]:
        lo,hi=sep*0.9,sep*3.0
        f=lambda t: run(p,s,S,a,l,d['Bhalf'],d['Gamma'],rs*a[l],rt*a[l],t*g*a[l],starts=(0.0,),joint=True)[0]
        while f(hi)>0: hi*=1.5
        if f(lo)<0: lo=0.0
        for it in range(12):
            mid=0.5*(lo+hi)
            if f(mid)>=0: lo=mid
            else: hi=mid
        if lo<best[0]: best=(lo,(rs,rt))
        print(f"  {key} ({rs},{rt}) sep={sep:.3f} joint={lo:.3f}",flush=True)
    print(f"p={p} l={l}: sigma_joint={best[0]:.3f} at {best[1]}  ({len(pts)} pts below thr; t={time.time()-t0:.0f})",flush=True)
