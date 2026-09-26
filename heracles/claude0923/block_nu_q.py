# Numerical nu-neuron analogue of astra's local_block.py: Hessian of (clean multilinear form - |theta|^2/2)
# on the kernel of the block-tensor Jacobian. Expect nullity n (phase motions), rest strictly negative.
import numpy as np
def run(nu,q=2.0):
    n=nu-1; L=nu; phis=[np.pi*l/nu for l in range(L)]
    base=np.zeros((L,nu,2))
    for l,ph in enumerate(phis):
        for j in range(n): base[l,j]=[np.cos(ph),np.sin(ph)]
        base[l,n]=[np.cos(n*ph),np.sin(n*ph)]
    th0=base.ravel(); D=th0.size
    def cplx(v): return v[0]+1j*v[1]
    def f(th):
        t=th.reshape(L,nu,2); s=0
        for l in range(L):
            z=np.prod([cplx(t[l,j]) for j in range(n)])*np.conj(cplx(t[l,n])); s+=z.real
        return s
    def tensor(th):
        t=th.reshape(L,nu,2); T=np.zeros([2]*nu)
        for l in range(L):
            o=t[l,0]
            for j in range(1,nu): o=np.multiply.outer(o,t[l,j])
            T+=o
        return T.ravel()
    e=1e-5;I=np.eye(D)
    J=np.array([(tensor(th0+e*I[i])-tensor(th0-e*I[i]))/(2*e) for i in range(D)]).T
    g=np.array([(f(th0+e*I[i])-f(th0-e*I[i]))/(2*e) for i in range(D)])
    assert np.allclose(g,th0,atol=1e-8)
    e2=1e-4;H=np.zeros((D,D))
    for i in range(D):
        for k in range(i,D):
            H[i,k]=H[k,i]=(f(th0+e2*(I[i]+I[k]))-f(th0+e2*(I[i]-I[k]))-f(th0-e2*(I[i]-I[k]))+f(th0-e2*(I[i]+I[k])))/(4*e2*e2)
    H-=I
    t=th0.reshape(L,-1)
    for l in range(L):
        sl=slice(l*2*nu,(l+1)*2*nu); th=th0[sl]; H[sl,sl]-=(q-2)*np.outer(th,th)/(th@th)
    U,sv,Vt=np.linalg.svd(J); rank=(sv>1e-7).sum(); N=Vt[rank:].T
    ev=np.linalg.eigvalsh(-N.T@H@N)
    print(f"nu={nu} q={q}: vars={D} rank J={rank} crit dim={N.shape[1]} eig(-H) on ker J:",np.round(ev,4))
for nu in (3,4,5):
    for q in (2.0,2.5,nu-0.01): run(nu,q)
