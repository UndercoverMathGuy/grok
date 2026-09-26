"""Check the proposed theorem directly, retaining DC and impure weights.
A constrained nonlinear optimizer is only a witness generator, never a proof.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent


def setup(p,n,N):
    rows=np.indices((p,)*n).reshape(n,-1).T;lab=rows.sum(1)%p
    wrong=(lab[:,None]+np.arange(1,p)[None,:])%p
    nu=n+1;blockmargin=nu/2**n*(1-np.cos(2*np.pi/p))
    P0=nu**2*p/2
    def eval(x,jac=False):
        E=x[:-1].reshape(nu,N,p)
        # Gather shapes are made explicit to avoid mixed advanced-index ambiguity.
        vals=np.stack([E[j][:,rows[:,j]].T for j in range(n)])
        h=vals.prod(axis=0)
        wd=E[n][:,lab].T[:,:,None]-E[n][:,wrong].transpose(1,0,2)
        gap=np.sum(h[:,:,None]*wd,axis=1)/blockmargin
        if not jac:return gap.ravel()
        J=np.zeros((len(rows),p-1,nu,N,p))
        for j in range(n):
            other=np.prod(np.delete(vals,j,axis=0),axis=0)
            for a in range(p):J[:,:,j,:,a]=(rows[:,j]==a)[:,None,None]*other[:,None,:]*wd.transpose(0,2,1)/blockmargin
        for a in range(p):
            mask=(lab==a)[:,None]-(wrong==a).astype(int)
            J[:,:,n,:,a]=mask[:,:,None]*h[:,None,:]/blockmargin
        return J.reshape(len(rows)*(p-1),nu*N*p)
    def cons(x):return eval(x)-x[-1]
    def cj(x):return np.c_[eval(x,True),-np.ones(len(rows)*(p-1))]
    def norm(x):return np.sum(x[:-1]**2)/P0-1
    def nj(x):return np.r_[2*x[:-1]/P0,0]
    return eval,cons,cj,norm,nj,P0,blockmargin


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p',type=int,default=3);ap.add_argument('--n',type=int,nargs='+',default=[2,3,4]);ap.add_argument('--starts',type=int,default=8);ap.add_argument('--extra',type=int,default=0)
    args=ap.parse_args();rng=np.random.default_rng(113);records=[]
    if args.p != 3:
        ap.error('This witness threshold is certified only for p=3; use verify.py for multifeature ansatz comparisons.')
    for n in args.n:
        nu=n+1;N=nu+args.extra;ev,con,jac,norm,nj,P0,bm=setup(args.p,n,N)
        base=np.zeros((nu,N,args.p))
        for l in range(nu):
            for j in range(n):base[j,l]=np.cos(2*np.pi*np.arange(args.p)/args.p+np.pi*l/nu)
            base[n,l]=np.cos(2*np.pi*np.arange(args.p)/args.p+n*np.pi*l/nu)
        base*=np.sqrt(P0/np.sum(base**2))
        assert abs(ev(np.r_[base.ravel(),1]).min()-1)<1e-12
        # Check the analytic Jacobian once against directional finite differences.
        x=np.r_[(base+.03*rng.normal(size=base.shape)).ravel(),.9];v=rng.normal(size=x.shape)
        error=np.max(abs((con(x+1e-6*v)-con(x-1e-6*v))/2e-6-jac(x)@v))
        assert error<1e-6,error
        for i in range(args.starts):
            t=time.time();scale=[0,.03,.15,.4,1][min(i,4)]
            weights=base+scale*rng.normal(size=base.shape)
            weights*=np.sqrt(P0/np.sum(weights**2))
            x=np.r_[weights.ravel(),0.];x[-1]=ev(x).min()
            res=minimize(lambda z:-z[-1],x,jac=lambda z:np.r_[np.zeros(len(z)-1),-1.],
                         constraints=[{'type':'eq','fun':norm,'jac':nj},{'type':'ineq','fun':con,'jac':jac}],
                         method='SLSQP',options={'maxiter':600,'ftol':1e-11})
            normratio=np.sum(res.x[:-1]**2)/P0
            ratio=float(ev(res.x).min()/normratio**(nu/2))
            record={'p':args.p,'n':n,'width':N,'start':i,'ratio':ratio,'success':bool(res.success),'message':res.message,'seconds':time.time()-t,'jac_error':float(error)}
            records.append(record);print(record,flush=True)
            if ratio>1+1e-7:
                np.savez(ROOT/f'witness_p{args.p}_n{n}_N{N}.npz',weights=res.x[:-1].reshape(nu,N,args.p),ratio=ratio)
        (ROOT/f'raw_p{args.p}_n{n}_extra{args.extra}.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
