"""Score the replica noise-floor experiment per flight/PREREG_FLOOR.md.
Result 2026-08-20: rho_id x1.049, rho_pert x1.182 (medians of 8 groups).
Registered band 1.1 < rho_pert < 1.4 -> PARTIAL, with the quantitative
split: sigma_floor(log10) ~ log10(rho_pert)/2.06 ~ 0.035 dex vs law
residual ~0.19-0.28 dex -> the noise/chaos floor explains only ~2-4% of
the forecast residual variance. Headroom dominates: the x1.5 residual is
mostly information our observables miss (under-modelling), not physics.
Secondary finding: training is NOT deterministic (identical configs
diverge, up to x1.24 at f0.25) and the floor GROWS with flight length —
chaotic amplification of hardware noise over long flights."""
import numpy as np, glob
def fs(m):
    i=np.where(m[:-1]&m[1:])[0]; return int(i[0]) if len(i) else -1
groups={}
for d in sorted(glob.glob('runs_flight/floornoise/*/*')):
    if not glob.glob(d+'/spectra.npz'): continue
    z=np.load(d+'/spectra.npz'); ig=fs(z['test_acc']>=0.9)
    tg=int(z['epochs'][ig]) if ig>=0 else None
    grp,rep=d.split('/')[-2],d.split('/')[-1]
    groups.setdefault(grp,{})[rep]=tg
rid,rpe=[],[]
for g,rr in sorted(groups.items()):
    ids=[rr[k] for k in sorted(rr) if k.startswith('rep')]
    pes=[rr[k] for k in sorted(rr) if k.startswith('pert')]
    if all(v is not None for v in ids): rid.append(max(ids)/min(ids))
    if all(v is not None for v in pes): rpe.append(max(pes)/min(pes))
    print(g, 'id', ids, 'pert', pes)
print(f'rho_id x{np.median(rid):.3f}  rho_pert x{np.median(rpe):.3f}')
sf=np.log10(np.median(rpe))/2.06
print(f'sigma_floor ~ {sf:.4f} dex; law residual ~0.19-0.28 dex; '
      f'floor variance share ~ {100*(sf/0.25)**2:.1f}%')
