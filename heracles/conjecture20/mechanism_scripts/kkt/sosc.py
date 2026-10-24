# Second-order sufficient condition check in the untied parameter space.
# Critical cone C = {d: DN d = 0, J_k d = 0 (k in A+ = union of multiplier supports), J_k d >= 0 (k in A \ A+)}.
# If the rows {J_k : k in A+} together with DN span the orthogonal complement of the balancing symmetries, C = symmetries.
import numpy as np, sys, itertools
sys.path.insert(0,'.')
from kkt import Net
from scipy.optimize import linprog
d=np.load(sys.argv[1]); net=Net(d['U'],d['W'],int(d['n']),2.0); p,n,m=net.p,net.n,net.m
A,q,lam,res,h=net.kkt(1e-7)
J=np.array([net.grad_gap(k,dl) for k,dl in A]).T; gN=net.grad_cost()
# union of supports: LP max q_k with ||Jq-gN||_inf <= tol
tol=1e-7*np.abs(gN).max()
Aub=np.vstack([J,-J]); bub=np.concatenate([gN+tol,-gN+tol])
Aplus=[]
for k in range(len(A)):
    c=np.zeros(len(A)); c[k]=-1
    r=linprog(c,A_ub=Aub,b_ub=bub,bounds=(0,None),method='highs')
    if r.status==0 and -r.fun>1e-9*np.abs(r.x).sum(): Aplus.append(k)
print('active pairs',len(A),'in union of multiplier supports',len(Aplus))
# untied jacobian rows for all x in classes of A+
U=np.repeat(net.U[:,None,:],n,1); W=net.W
X=np.array(list(itertools.product(range(p),repeat=n))); Y=X.sum(1)%p
cls={tuple(N):k for k,N in enumerate(net.Nc)}
Xc=np.array([cls[tuple(np.bincount(x,minlength=p))] for x in X])
rows=[]
Aset=set((A[k][0],A[k][1]) for k in Aplus)
for xi,x in enumerate(X):
    for dl in range(1,p):
        if (Xc[xi],dl) not in Aset: continue
        y=Y[xi]; dw=W[:,y]-W[:,(y-dl)%p]
        f=np.array([[U[i,t,x[t]] for t in range(n)] for i in range(m)])
        gU=np.zeros((m,n,p)); 
        for t in range(n):
            gU[np.arange(m),t,x[t]]=np.prod(np.delete(f,t,1),1)*dw
        z=np.prod(f,1); gW=np.zeros((m,p)); gW[:,y]+=z; gW[:,(y-dl)%p]-=z
        rows.append(np.concatenate([gU.ravel(),gW.ravel()]))
R=np.array(rows)
s=np.linalg.svd(R,compute_uv=False)
npar=m*n*p+m*p
rk=(s>1e-10*s[0]).sum()
print('untied params',npar,'rows',R.shape[0],'rank',rk,'nullity',npar-rk,'balancing symmetries dim',m*n)
print('smallest singular values', s[rk-3:rk+2] if rk+2<=len(s) else s[-5:])
# --- Hessian of the Lagrangian on null(R) modulo balancing symmetries
import scipy.linalg as sla
Vt=np.linalg.svd(R)[2]; Z=Vt[rk:].T      # basis of null space (npar x nullity)
# balancing symmetry tangents: neuron i, slot t: U[i,t]+=U[i,t], W[i]-= W[i]  (scale factor up in slot t, down in output)
S=[]
for i in range(m):
    for t in range(n):
        dU=np.zeros((m,n,p)); dW=np.zeros((m,p)); dU[i,t]=U[i,t]; dW[i]=-W[i]
        S.append(np.concatenate([dU.ravel(),dW.ravel()]))
S=np.array(S).T
# output-DC directions
D=[]
for i in range(m):
    dU=np.zeros((m,n,p)); dW=np.zeros((m,p)); dW[i]=1.0; D.append(np.concatenate([dU.ravel(),dW.ravel()]))
D=np.array(D).T
print('residual of symmetries in null space', np.linalg.norm(R@S)/np.linalg.norm(S), ' DC', np.linalg.norm(R@D)/np.linalg.norm(D))
Qs=sla.orth(S); Zp=Z-Qs@(Qs.T@Z); Zp=sla.orth(Zp)
print('null space modulo balancing: dim',Zp.shape[1])
# q on untied pairs
qmap={}
for (k,dl),qq in zip(A,q): qmap[(k,dl)]=qq/net.cs[k]
# Hessian of F_q - lam*N  in untied params, contracted with Zp
npar=m*n*p+m*p
def unpack(v): return v[:m*n*p].reshape(m,n,p), v[m*n*p:].reshape(m,p)
H=np.zeros((Zp.shape[1],Zp.shape[1]))
Zs=[unpack(Zp[:,c]) for c in range(Zp.shape[1])]
for xi,x in enumerate(X):
    for dl in range(1,p):
        w=qmap.get((Xc[xi],dl),0.0)
        if w==0: continue
        y=Y[xi]; ym=(y-dl)%p
        f=np.array([[U[i,t,x[t]] for t in range(n)] for i in range(m)])  # m x n
        dw=W[:,y]-W[:,ym]
        # for each direction c: first-order factor perturbations e[c][i,t]=dU[i,t,x_t], ew[c][i]=dW[i,y]-dW[i,ym]
        E=np.array([[[Zs[c][0][i,t,x[t]] for t in range(n)] for i in range(m)] for c in range(len(Zs))])  # C x m x n
        EW=np.array([Zs[c][1][:,y]-Zs[c][1][:,ym] for c in range(len(Zs))])  # C x m
        # second derivative of prod_t f_t * dw along (a,b): sum_{t!=t'} e_a[t] e_b[t'] prod_{others} dw + sum_t (e_a[t] ew_b + e_b[t] ew_a) prod_{t'!=t} f
        for i in range(m):
            fi=f[i]
            P1=np.array([np.prod(np.delete(fi,t)) for t in range(n)])          # prod over t'!=t
            P2=np.array([[np.prod(np.delete(fi,[t,s])) if t!=s else 0 for s in range(n)] for t in range(n)])
            Ei=E[:,i,:]   # C x n
            Huu=Ei@P2@Ei.T*dw[i]
            Huw=np.outer(Ei@P1,EW[:,i]); 
            H+=w*(Huu+Huw+Huw.T)
H-=lam*2*np.eye(H.shape[0])*1.0  # b=2: N = ||params||^2 summed (tied->untied norm is sum over slots), Hessian 2I on orthonormal basis
ev=np.linalg.eigvalsh(H)
print('eigenvalues of reduced Lagrangian Hessian (should be <0 for SOSC):',np.round(ev,8))
