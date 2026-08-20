"""Predictability-horizon curve on the 30 preregistered forecast runs
(both cohorts, stitched seg1+seg2): forecast hit-rate (x1.5 band) vs
observed fraction of the flight, linear-lambda + top-4 committee at t_fc.
Result 2026-08-20: 29/52/37/44/56% at 5/10/20/30/50% observed — roughly
flat; median ratio ~x1.5 at every horizon. Point-timing at x1.5 is not
reachable by early observation in this model class; rate evolution is
seed-stochastic."""
import numpy as np, glob
def fs(m):
    i=np.where(m[:-1]&m[1:])[0]; return int(i[0]) if len(i) else -1
RS=10**-2.25; LRS=np.log(RS)
runs=[]
for dd in sorted(glob.glob('runs_flight/t2d/*/*')+glob.glob('runs_flight/t2dv2/*/*')):
    if not glob.glob(dd+'/seg2/spectra.npz'): continue
    z1=np.load(dd+'/spectra.npz'); z2=np.load(dd+'/seg2/spectra.npz')
    ep=np.concatenate([z1['epochs'],500+z2['epochs']]); E=np.concatenate([z1['energy'],z2['energy']])
    ta=np.concatenate([z1['train_acc'],z2['train_acc']]); te=np.concatenate([z1['test_acc'],z2['test_acc']])
    im,ig=fs(ta>=0.99),fs(te>=0.9)
    if im<0 or ig<0: continue
    runs.append((ep,E,int(ep[im]),int(ep[ig])))
print(f'{len(runs)} runs')
for fr in (0.05,0.1,0.2,0.3,0.5):
    hits=0;n=0;ratios=[]
    for ep,E,tm,tg in runs:
        tfc=tm+fr*(tg-tm)
        if tfc-tm<50: continue
        ifc=np.searchsorted(ep,tfc,'right')-1
        comm=np.argsort(E[ifc])[::-1][:4].tolist()
        junk=[j for j in range(E.shape[1]) if j not in comm]
        r=E[:,junk].sum(1)/E.sum(1)
        w=np.where((ep>=tm)&(ep<=tfc))[0]
        if len(w)<4: continue
        lam=-np.polyfit(ep[w].astype(float),np.log(np.clip(r[w],1e-12,None)),1)[0]
        im2=np.searchsorted(ep,tm)
        if r[im2]<=np.e*RS: pred=tm+132
        elif lam<=0: continue
        else: pred=tm+max(0,np.log(r[im2])-LRS)/lam
        n+=1; rat=max(pred/tg,tg/pred); ratios.append(rat); hits+=rat<=1.5
    if n: print(f'{int(100*fr)}% observed: {hits}/{n} ({100*hits/n:.0f}%) median x{np.median(ratios):.2f}')
