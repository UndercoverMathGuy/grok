# Collective undercutting test (class C, n=2): every programme feature realised by a cheap 2-neuron sub-network,
# translations (phases) optimised to keep the joint field inside the margin.  Ratio > 1 means it beats IF.
import numpy as np, sys, time
from prog import solve, kappa
p=int(sys.argv[1]); s=float(sys.argv[2]); restarts=int(sys.argv[3]) if len(sys.argv)>3 else 4
th=2*np.pi/3; r=1/s
G,y,cov=solve(p,s); S=np.nonzero(y)[0]+1; yS=y[S-1]; K=len(S)
boost=(2/3)**(1-s)              # amplitude per cost of 2 aligned atoms relative to the IF block
A=boost*yS
# 2-atom polynomial P(z)=A/2[(1+z1)(1+z2)+(1+e^{i th}z1)(1+e^{i th}z2)]: coefficients of z1, z2, z1z2
b1=A/2*(1+np.exp(1j*th)); b2=b1.copy(); b12=A/2*(1+np.exp(2j*th))
rho=np.exp(2j*np.pi/p); x=np.arange(p)
X1,X2=np.meshgrid(x,x,indexing='ij'); X1=X1.ravel(); X2=X2.ravel()
Kd=kappa(p)  # rows delta=1..F
covS=Kd[:,S-1]@A
def margin_all(ph1,ph2,dels):
    out=np.inf
    for d in dels:
        c=1-rho**(-S*d)
        g=np.full(p*p,covS[min(d,p-d)-1])
        for k,l in enumerate(S):
            z1=rho**(2*l*X1); z2=rho**(2*l*X2)
            f=b1[k]*np.exp(1j*ph1[k])*z1+b2[k]*np.exp(1j*ph2[k])*z2+b12[k]*np.exp(1j*(ph1[k]+ph2[k]))*z1*z2
            g+=np.real(f*c[k])
        out=min(out,g.min())
    return out
# working set: near-binding deltas (both signs)
dF=np.arange(1,(p-1)//2+1); near=dF[covS<np.sort(covS)[0]*1.0+ (np.sort(covS)[min(len(covS)-1,3*K)]-np.sort(covS)[0])]
W=np.concatenate([near,p-near])
rng=np.random.default_rng(1); best=(-np.inf,None)
Z1={l:rho**(2*l*X1) for l in S}; Z2={l:rho**(2*l*X2) for l in S}
C={d:1-rho**(-S*d) for d in W}
def evalgrad(ph1,ph2,tau):
    gs=[];grads1=[];grads2=[]
    for d in W:
        c=C[d]; g=np.full(p*p,covS[min(d,p-d)-1]); d1=np.zeros((K,p*p)); d2=np.zeros((K,p*p))
        for k,l in enumerate(S):
            t1=b1[k]*np.exp(1j*ph1[k])*Z1[l]*c[k]; t2=b2[k]*np.exp(1j*ph2[k])*Z2[l]*c[k]
            t12=b12[k]*np.exp(1j*(ph1[k]+ph2[k]))*Z1[l]*Z2[l]*c[k]
            g+=np.real(t1+t2+t12); d1[k]=np.real(1j*(t1+t12)); d2[k]=np.real(1j*(t2+t12))
        gs.append(g);grads1.append(d1);grads2.append(d2)
    g=np.concatenate(gs); m=g.min(); wts=np.exp(-(g-m)/tau); wts/=wts.sum()
    D1=np.concatenate(grads1,axis=1); D2=np.concatenate(grads2,axis=1)
    return m, D1@wts, D2@wts
t0=time.time()
for rs in range(restarts):
    ph1=rng.uniform(0,2*np.pi,K); ph2=rng.uniform(0,2*np.pi,K)
    mA=[np.zeros(K),np.zeros(K)]; vA=[np.zeros(K),np.zeros(K)]
    for it in range(400):
        tau=G*0.05*(0.97**it)+1e-4*G
        m,g1,g2=evalgrad(ph1,ph2,tau)
        for q,(ph,gr) in enumerate(((ph1,g1),(ph2,g2))):
            mA[q]=0.9*mA[q]+0.1*gr; vA[q]=0.999*vA[q]+0.001*gr**2
            ph+=0.05*mA[q]/(np.sqrt(vA[q]/(1-0.999**(it+1)))+1e-12)
    mt=margin_all(ph1,ph2,range(1,p))
    print(f"  restart {rs}: margin/Gamma = {mt/G:.4f}  ({time.time()-t0:.0f}s)",flush=True)
    if mt>best[0]: best=(mt,(ph1.copy(),ph2.copy()))
print(f"p={p} s={s} K*={K} boost={boost:.4f}  BEST collective 2-atom margin / IF margin = {best[0]/G:.4f}",flush=True)
np.save(f"best_p{p}_s{s}.npy",np.concatenate(best[1]))
