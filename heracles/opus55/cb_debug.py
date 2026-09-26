exec(open('v3_certify.py').read().split('fails=[]')[0])
cell=(0.4203125, 0.425, 1.0325, 1.0339)
mu,E2,E3=cparams(*cell); print("mu",mu,"E2",E2,"E3",E3)
import numpy as np
e1f=cell[2]; E2f=float(E2.a); E3f=max(0.0,float(E3.a))
disc=e1f**2-4*E2f; a=(e1f+np.sqrt(disc))/2; b=(e1f-np.sqrt(disc))/2
print("roots a,b",a,b,"value",a**(2/3)+b**(2/3),"target",float(TGT(cell[0]).b))
aa=iv.mpf(a*(1-1e-9)); bb=iv.mpf(b*(1-1e-9))
print("checks", (aa+bb).b<=cell[2], (aa*bb).b<=E2.a)
print("cbound returns",cbound(*cell))
