import json,numpy as np,sys
s=float(sys.argv[1]); cur=json.load(open(f'curves2_s{s}_base.json')); Gs=json.load(open(f'G_sym_s{s}.json'))
gr=np.array(sorted(float(k) for k in Gs)); gv=np.array([Gs[k] for k in sorted(Gs,key=float)])
rmax=3**(s-1); ax=np.unique(np.round(np.concatenate([[0,0.02,0.05,0.1,0.15,0.2],np.arange(0.3,rmax,0.1),[rmax]]),4))
und=set()
for k,c in cur.items():
    rho=np.concatenate([[0],c['rho']]); P1=np.interp(ax,rho,np.concatenate([[0],c['Pi1']])); P2=np.interp(ax,rho,np.concatenate([[0],c['Pi12']]))
    for i,rs in enumerate(ax):
        for j,rt in enumerate(ax):
            if i==0 and j==0: continue
            g=np.interp(max(rs,rt),gr,gv)
            if max(P1[i],P2[j])<1.02*g: und.add((rs,rt))
print(len(und),'undecided of',len(ax)**2); print(sorted(und))
