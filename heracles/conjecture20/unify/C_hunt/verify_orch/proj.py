# Orchestrator check: is the (11,2,2.5) candidate a genuine RD_{1} violation once foreign clean coefficients are set exactly to 0?
import sys, numpy as np
sys.path.insert(0,'/home/user/grok/heracles/conjecture20/unify/global')
import lib as G
from scipy.optimize import minimize
z=np.load('../out/VIOL_b_p11_n2_b2.5_pen_s51_9.npz')
U,W,p,n,b=z['U'],z['W'],int(z['p']),int(z['n']),float(z['b'])
net=G.Net(U,W,b); s=net.s; nu=n+1; F=(p-1)//2
D1=3**(1-s)*G.Bp(p,n)*(1-np.cos(2*np.pi/p))
print('stored ratio',float(z['ratio']),' gamma/D1',net.gamma()/D1)
print('|beta|',np.abs(net.clean()))
def foreign(Uv,Wv):
    nt=G.Net(Uv,Wv,b); c=nt.clean(); return np.concatenate([c[1:].real,c[1:].imag])
# exact projection: zero beta_j (j>=2) by least-norm change of the frequency-j output components (cos/sin of W)
def project(Uv,Wv):
    Wv=Wv.copy(); x=np.arange(p)
    for j in range(2,F+1):
        for it in range(3):
            nt=G.Net(Uv,Wv,b); bj=nt.clean()[j-1]
            # beta_j is linear in W; build its gradient wrt W via finite basis
            basis=[np.cos(2*np.pi*j*x/p),np.sin(2*np.pi*j*x/p)]
            cols=[]
            for i in range(Uv.shape[0]):
                for bvec in basis:
                    W2=Wv.copy(); W2[i]+=bvec; cols.append(G.Net(Uv,W2,b).clean()[j-1]-bj)
            A=np.array(cols); M=np.stack([A.real,A.imag])          # 2 x (2m)
            dx=-np.linalg.lstsq(M,np.array([bj.real,bj.imag]),rcond=None)[0]
            k=0
            for i in range(Uv.shape[0]):
                for bvec in basis: Wv[i]+=dx[k]*bvec; k+=1
    return Wv
Wp=project(U,W); netp=G.Net(U,Wp,b)
print('after projection |beta|',np.abs(netp.clean()),' gamma/D1',netp.gamma()/D1)
# local re-optimisation inside N_{1}: maximise margin at fixed cost, with exact equality constraints beta_j=0 (j>=2)
m=U.shape[0]
def unpack(v): return v[:m*n*p].reshape(m,n,p), v[m*n*p:].reshape(m,p)
v0=np.concatenate([U.ravel(),Wp.ravel(),[netp.margin()]])
def obj(v): return -v[-1]
cons=[{'type':'ineq','fun':lambda v: (G.Net(*unpack(v[:-1]),b).gaps().ravel()-v[-1])},
      {'type':'eq','fun':lambda v: foreign(*unpack(v[:-1]))},
      {'type':'eq','fun':lambda v: np.array([G.Net(*unpack(v[:-1]),b).cost()-netp.cost()])}]
res=minimize(obj,v0,constraints=cons,method='SLSQP',options={'maxiter':400,'ftol':1e-14})
Uo,Wo=unpack(res.x[:-1]); Wo=project(Uo,Wo); neto=G.Net(Uo,Wo,b)
print('SLSQP',res.message,' after re-projection |beta|',np.abs(neto.clean()))
print('re-optimised in N_{1}: gamma/D1 =',neto.gamma()/D1)
np.savez('reopt_N1.npz',U=Uo,W=Wo,p=p,n=n,b=b)
