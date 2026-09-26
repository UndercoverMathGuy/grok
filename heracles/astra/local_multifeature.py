"""Numerical critical-space Hessian checks; only local_exact.py is exact."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from verify import vertex_amplitudes,ansatz
from raw_margin import setup

records=[]
for p in (3,5,7,11,13,17):
 v=vertex_amplitudes(p);w,gamma=ansatz(p,v);N=w.shape[1];h=(p-1)//2
 A=1-np.cos(2*np.pi*np.outer(np.arange(1,h+1),np.arange(1,h+1))/p)
 S=np.flatnonzero(v);T=np.flatnonzero(abs(A@v-1)<1e-9)
 b=v[S]**(2/3);b/=b.sum()
 dual=np.linalg.lstsq(A[np.ix_(T,S)].T,gamma/np.sqrt(b),rcond=None)[0]
 assert dual.min()>0 and abs(dual.sum()-1)<1e-9
 offsets=np.zeros(p)
 offsets[T+1]=dual/2;offsets[p-(T+1)]=dual/2
 tensor=np.empty((p,p,p))
 for a in range(p):
  for b0 in range(p):
   for c in range(p):
    D=(c-a-b0)%p
    tensor[a,b0,c]=(1 if D==0 else -offsets[D])/p**2
 H=np.zeros((w.size,w.size));gradient=np.zeros_like(w)
 for i in range(N):
  gradient[0,i]=np.einsum('abc,b,c->a',tensor,w[1,i],w[2,i])
  gradient[1,i]=np.einsum('abc,a,c->b',tensor,w[0,i],w[2,i])
  gradient[2,i]=np.einsum('abc,a,b->c',tensor,w[0,i],w[1,i])
  blocks=[np.einsum('abc,c->ab',tensor,w[2,i]),np.einsum('abc,b->ac',tensor,w[1,i]),np.einsum('abc,a->bc',tensor,w[0,i])]
  for (j,k),block in zip(((0,1),(0,2),(1,2)),blocks):
   jx=slice((j*N+i)*p,(j*N+i+1)*p);kx=slice((k*N+i)*p,(k*N+i+1)*p)
   H[jx,kx]=block;H[kx,jx]=block.T
 mu=gamma/(2*p)
 error=np.max(abs(gradient-mu*w));assert error<1e-11
 H-=mu*np.eye(w.size)
 ev,con,jac,norm,nj,p0,bm=setup(p,2,N)
 z=np.r_[w.ravel(),0.]
 active=np.tile(offsets[1:]>0,p*p)
 J=jac(z)[active,:-1]
 Q=null_space(J,rcond=1e-10)
 eig=np.linalg.eigvalsh(Q.T@H@Q)/mu
 rec={'p':p,'K':len(S),'positive_dual_offsets':list((T+1).astype(int)),
      'critical_dimension':Q.shape[1],'positive_eigenvalues':int(sum(eig>1e-8)),
      'nullity':int(sum(abs(eig)<=1e-8)),'largest_eigenvalue_scaled':float(eig.max()),
      'largest_negative_eigenvalue_scaled':float(eig[eig< -1e-8].max()),
      'stationarity_error':float(error),'expected_phase_nullity':2*len(S)}
 rec['positive_dual_offsets']=list(map(int,rec['positive_dual_offsets']))
 records.append(rec);print(rec,flush=True)
Path(__file__).with_name('local_multifeature.json').write_text(json.dumps(records,indent=2)+'\n')
