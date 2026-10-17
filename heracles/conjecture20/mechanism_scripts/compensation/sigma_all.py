# Combined verdicts for the T-resolved game (n=2) at a fixed programme allocation.
#  riesz : min_{(rs,rt)} max(Pi1(rs),Pi12(rt)) / G*(rs,rt)                       [single-character Riesz tests, T={1},{2},{1,2}]
#  L2    : min_{(rs,rt)} PiL2(rs,rs,rt) / G*(rs,rt)                               [quadratic test over all T, f=f_min]
#  best  : min_{(rs,rt)} max(Pi1,Pi12,PiL2) / G*                                  [prover uses the best test]
#  single: min_rt Piquad12(rt) / G*(0,rt)                                         [interference on T={1,2} only; f=1]
#  t12riesz: min_rt Pi12(rt)/G*(0,rt)
import json,numpy as np,glob,sys
s=float(sys.argv[1]); var=sys.argv[2] if len(sys.argv)>2 else 'base'
G={}; ax=None
for f in glob.glob(f'grid_s{s}_part*.json'):
    d=json.load(open(f)); ax=d['ax']; G.update(d['G'])
ax=np.array(ax); have=set(G)
cur=json.load(open(f'curves2_s{s}_{var}.json'))
import os
L2=json.load(open(f'curvesL2_s{s}_{var}.json')) if os.path.exists(f'curvesL2_s{s}_{var}.json') else dict(ax=[0],ax1=[0],res={})
axL=np.array(L2['ax']); ax1=np.array(L2['ax1'])
def key(a,b): return f"{a},{b}"
rows=[]
for k,c in cur.items():
    rho=np.concatenate([[0],c['rho']]); P1=np.interp(ax,rho,np.concatenate([[0],c['Pi1']])); P2=np.interp(ax,rho,np.concatenate([[0],c['Pi12']]))
    ri=(np.inf,None)
    for i,rs in enumerate(ax):
        for j,rt in enumerate(ax):
            g=G.get(key(rs,rt))
            if g is None or g<=1e-7: continue
            v=max(P1[i],P2[j])/g
            if v<ri[0]: ri=(v,(rs,rt))
    l2=L2['res'].get(k); l2v=(np.nan,None); bv=(np.nan,None); sg=(np.nan,None); t12=(np.inf,None)
    if l2:
        l2v=(np.inf,None); bv=(np.inf,None)
        for rs in axL:
            for rt in axL:
                g=G.get(key(rs,rt)); pl=l2['grid'].get(key(rs,rt))
                if g is None or pl is None or g<=1e-7: continue
                i=int(np.argmin(abs(ax-rs))); j=int(np.argmin(abs(ax-rt)))
                if pl/g<l2v[0]: l2v=(pl/g,(rs,rt))
                b=max(P1[i],P2[j],pl)/g
                if b<bv[0]: bv=(b,(rs,rt))
        sg=(np.inf,None)
        for rt in ax1:
            g=G.get(key(0.0,rt)); q=l2['single'].get(str(rt))
            if g is None or q is None or g<=1e-7: continue
            if q/g<sg[0]: sg=(q/g,rt)
    for j,rt in enumerate(ax):
        g=G.get(key(0.0,rt))
        if g is None or g<=1e-7: continue
        if P2[j]/g<t12[0]: t12=(P2[j]/g,rt)
    rows.append((c['p'],c['l'],ri,l2v,bv,sg,t12))
rows.sort(key=lambda r:(r[0],r[1]))
print(f"s={s} variant={var}  grid rows have {len(have)} pts")
print("  p   l | riesz (rs,rt)        | L2 (rs,rt)          | best           | single-T quad | single-T riesz")
for p,l,ri,l2v,bv,sg,t12 in rows:
    print(f"{p:3d} {l:3d} | {ri[0]:6.3f} {str(ri[1]):14s} | {l2v[0]:6.3f} {str(l2v[1]):12s} | {bv[0]:6.3f} {str(bv[1]):12s} | {sg[0]:6.3f} {str(sg[1]):6s} | {t12[0]:6.3f} {str(t12[1])}")
for nm,idx in (('riesz',2),('L2',3),('best',4),('singleQ',5),('singleR',6)):
    vals=[r[idx][0] for r in rows if np.isfinite(r[idx][0])]
    if vals: print(nm,'min',round(min(vals),4))
