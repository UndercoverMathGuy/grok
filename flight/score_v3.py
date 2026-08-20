"""Score v3 per flight/PREREG_V3.md (tiered)."""
import json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
F=json.load(open(ROOT/'flight'/'forecasts_v3.json'))
FE=F['meta']['forecast_epoch']
IN_DIST={'f0.3_wd1.0_lr0.001_m512','f0.25_wd1.0_lr0.001_m512','f0.4_wd1.0_lr0.001_m512',
         'f0.3_wd3.0_lr0.001_m512','f0.25_wd3.0_lr0.001_m512'}
EXTRAP={'f0.3_wd5.0_lr0.001_m512','f0.22_wd1.0_lr0.001_m512'}
def fs(m):
    i=np.where(m[:-1]&m[1:])[0]; return int(i[0]) if len(i) else -1
def realized(rd):
    z1=np.load(rd/'spectra.npz')
    if fs(z1['test_acc']>=0.9)>=0: return int(z1['epochs'][fs(z1['test_acc']>=0.9)])
    z2=np.load(rd/'seg2'/'spectra.npz'); ig=fs(z2['test_acc']>=0.9)
    return None if ig<0 else FE+int(z2['epochs'][ig])
tiers={'in-dist':[], 'new-value':[], 'extrap':[]}
pairs=[]; ratios=[]
for row in F['forecasts']:
    if row['status']!='ok': continue
    cell=row['run'].split('/')[1]
    tier='in-dist' if cell in IN_DIST else ('extrap' if cell in EXTRAP else 'new-value')
    act=realized(ROOT/'runs_flight'/row['run'])
    pred=row['forecast']
    if pred=='CENSORED': ok=act is None; r=None
    elif act is None: ok=False; r=None
    else:
        r=max(pred/act,act/pred); ok=r<=1.5; pairs.append((pred,act)); ratios.append(r)
    tiers[tier].append(ok)
tot=sum(len(v) for v in tiers.values()); hit=sum(sum(v) for v in tiers.values())
print(f'P-V3a: {hit}/{tot} = {100*hit/tot:.0f}% within bars (bar >= 60%) -> {"PASS" if hit/tot>=0.6 else "FAIL"}')
print(f'P-V3b: median miss x{np.median(ratios):.2f} (bar <= x1.35) -> {"PASS" if np.median(ratios)<=1.35 else "FAIL"}')
a=np.array(pairs,float)
ra,rb=np.argsort(np.argsort(a[:,0])),np.argsort(np.argsort(a[:,1]))
rho=float(np.corrcoef(ra,rb)[0,1])
print(f'P-V3c: Spearman {rho:.3f} over {len(pairs)} pairs (bar >= 0.85) -> {"PASS" if rho>=0.85 else "FAIL"}')
for t,v in tiers.items():
    print(f'  {t:10}: {sum(v)}/{len(v)} = {100*sum(v)/max(len(v),1):.0f}%')
ind=tiers['in-dist']
print(f'P-V3d: in-dist {sum(ind)}/{len(ind)} (bar >= 65%) -> {"PASS" if sum(ind)/len(ind)>=0.65 else "FAIL"}')
