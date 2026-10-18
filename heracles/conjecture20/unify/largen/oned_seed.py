# Load the Note's optimised 1-D network (p=5,n=6) into the multi-frequency framework, check its ratio,
# then release every input/output coordinate (all frequencies) and re-optimise; report clean amplitudes.
import sys, json, math, numpy as onp
sys.path.insert(0,'.')
from mfnet import MF, optimise
J=json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/kkt/oned_p5_n6.json'))
p,n=J['p'],J['n']; m=J['width']; mf=MF(p,n)
a=onp.arange(p); e=onp.where(a<p/2,2*a,2*a-p).astype(float); eps=onp.where(a<p/2,1.0,-1.0)
lam=onp.array(J['lam']); logV=onp.array(J['logV']); argV=onp.array(J['argV'])
U=onp.array([eps*onp.exp(l*e) for l in lam]); nrm=onp.linalg.norm(U,axis=1)
W=onp.array([onp.cos(2*onp.pi*a/p+ag) for ag in argV])
l=logV+n*onp.log(nrm)
z0=onp.concatenate([U.ravel(),W.ravel(),l])
r0=mf.ratio(z0,m); print("1-D seed ratio (Note: %.4f):"%J['numerical_ratio'], r0, " clean:",mf.clean(z0,m))
# conjugate-phase variant in case of convention mismatch
W2=onp.array([onp.cos(2*onp.pi*a/p-ag) for ag in argV]); z1=onp.concatenate([U.ravel(),W2.ravel(),l])
print("conjugate convention ratio:", mf.ratio(z1,m))
z=z0 if r0>mf.ratio(z1,m) else z1
rng=onp.random.default_rng(5)
for eps_ in [0.0,0.01,0.03]:
    zz=z.copy(); zz[:2*m*p]+=eps_*rng.standard_normal(2*m*p)
    v,zo=optimise(mf,zz,m)
    print(f"released (noise {eps_}): ratio={v:.5f}  clean eta/IFprice={mf.clean(zo,m)}",flush=True)
    # also with one extra neuron
    zz2=onp.concatenate([zo[:m*p],rng.standard_normal(p)*0.3,zo[m*p:2*m*p],rng.standard_normal(p)*0.3,zo[2*m*p:],[zo[2*m*p:].min()-2]])
    v2,zo2=optimise(mf,zz2,m+1)
    print(f"   +1 neuron: ratio={v2:.5f}  clean={mf.clean(zo2,m+1)}",flush=True)
