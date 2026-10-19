# Safety ratio of the T-resolved single-cheap-feature game:
#   sigma(p,l) = min_{(rs,rt) != 0} max(Pi_1(rs), Pi_12(rt)) / Gstar(rs,rt)      (all per unit K_l, units a_l)
# network wins (in the model) iff sigma < 1.  Also reports the linear-gain verdicts.
import json,numpy as np,sys,glob
s=float(sys.argv[1]); cur=json.load(open(sys.argv[2])); tag=sys.argv[3] if len(sys.argv)>3 else ''
G={}; ax=None
for f in glob.glob(f'grid_s{s}_part*.json'):
    d=json.load(open(f)); ax=d['ax']; G.update(d['G'])
ax=np.array(ax); c1=1-3**(1-s)
rows=[]
for key,c in cur.items():
    rho=np.array(c['rho']); P1=np.interp(ax,np.concatenate([[0],rho]),np.concatenate([[0],c['Pi1']]))
    P2=np.interp(ax,np.concatenate([[0],rho]),np.concatenate([[0],c['Pi12']]))
    best=(np.inf,None)
    for i,rs in enumerate(ax):
        for k,rt in enumerate(ax):
            g=G.get(f"{rs},{rt}")
            if g is None or (i==0 and k==0) or g<=1e-12: continue
            val=max(P1[i],P2[k])/g
            if val<best[0]: best=(val,(rs,rt,g,P1[i],P2[k]))
    # linear gains (sym, max_T R): sigma_lin = min_R max(Pi1,Pi12)/(g R) -> h at R->0 (T-blind)
    rows.append((c['p'],c['l'],c['h1'],c['h12'],best[0],best[1]))
rows.sort()
print(f"s={s} {tag}   p  l   h1     h12   sigma   at (rho_s,rho_12)  G*   Pi1   Pi12")
for p,l,h1,h12,sg,info in rows:
    print(f"   {p:3d} {l:3d}  {h1:.3f}  {h12:.3f}  {sg:7.3f}   ({info[0]:.2f},{info[1]:.2f})  {info[2]:.4f} {info[3]:.4f} {info[4]:.4f}")
print("min sigma:",min(r[4] for r in rows))
