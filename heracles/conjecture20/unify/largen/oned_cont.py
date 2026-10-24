# Continuation in n of the released 1-D network (p=5): start from the n=6 released local maximiser, add a slot
# (same neurons, slot-symmetric) and one neuron per step, re-optimise all frequencies; report ratio and clean amplitudes.
import sys, json, math, numpy as onp
sys.path.insert(0,'.')
from mfnet import MF, optimise
J=json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/kkt/oned_p5_n6.json'))
p,n=J['p'],J['n']; m=J['width']
a=onp.arange(p); e=onp.where(a<p/2,2*a,2*a-p).astype(float); eps=onp.where(a<p/2,1.0,-1.0)
lam=onp.array(J['lam']); logV=onp.array(J['logV']); argV=onp.array(J['argV'])
U=onp.array([eps*onp.exp(l*e) for l in lam]); nrm=onp.linalg.norm(U,axis=1)
W=onp.array([onp.cos(2*onp.pi*a/p+ag) for ag in argV]); l=logV+n*onp.log(nrm)
z=onp.concatenate([U.ravel(),W.ravel(),l])
mf=MF(p,n); v,z=optimise(mf,z,m); print(f"n={n} m={m}: ratio={v:.5f} clean={ {k:float('%.3e'%x) for k,x in mf.clean(z,m).items()} }",flush=True)
rng=onp.random.default_rng(3)
for n2 in range(n+1,int(sys.argv[1])+1):
    mf=MF(p,n2); best=(-onp.inf,None,None)
    Uc=z[:m*p].reshape(m,p); Wc=z[m*p:2*m*p].reshape(m,p); lc=z[2*m*p:]
    for trial in range(4):
        # new neuron: copy a random existing neuron with perturbation, small mass; adjust masses for the extra slot
        k=rng.integers(m)
        U2=onp.vstack([Uc,Uc[k]*(1+0.1*rng.standard_normal(p))]); W2=onp.vstack([Wc,rng.standard_normal(p)])
        l2=onp.concatenate([lc,[lc.min()]])+0.0
        z2=onp.concatenate([U2.ravel(),W2.ravel(),l2])
        v2,zz=optimise(mf,z2,m+1)
        if v2>best[0]: best=(v2,zz,m+1)
        v3,zz3=optimise(mf,z.copy(),m)          # same width
        if v3>best[0]: best=(v3,zz3,m)
    v,z,m=best
    print(f"n={n2} m={m} (nu={n2+1}): ratio={v:.5f} clean={ {k:float('%.3e'%x) for k,x in mf.clean(z,m).items()} }",flush=True)
