import numpy as np, sys, itertools
sys.path.insert(0,'.')
from kkt import Net
from scipy.optimize import linprog
d=np.load(sys.argv[1]); net=Net(d['U'],d['W'],int(d['n']),2.0); p,n=net.p,net.n
A,q,lam,res,h=net.kkt(1e-7)
print('p',p,'n',n,'KKT res',res)
# coverage of active set by uniform measure, per delta
for dl in range(1,p):
    ks=[k for (k,dd) in A if dd==dl]
    print(' delta',dl,'active classes',len(ks),'fraction of inputs active %.4f'%(net.cs[ks].sum()/p**n), ' q-mass %.4f'%sum(qq for (k,dd),qq in zip(A,q) if dd==dl))
# min TV distance from KKT polytope to uniform-in-x measures U x mu (mu free prob on delta):
J=np.array([net.grad_gap(k,dl) for k,dl in A]).T; gN=net.grad_cost()
K=len(net.Nc); nA=len(A)
# variables: qt (nA, scaled so J qt = gN, so q = qt*lam), mu (p-1) scaled same, slack e_{k,d} >= |qt_{k,d} - cs_k/p^n mu_d| for all (k,d) classes (inactive have qt=0)
pairs=[(k,dl) for k in range(K) for dl in range(1,p)]
idx={pr:i for i,pr in enumerate(pairs)}
nv=nA+(p-1)+len(pairs)
c=np.zeros(nv); c[nA+p-1:]=1
Aeq=np.zeros((J.shape[0]+1,nv)); Aeq[:J.shape[0],:nA]=J; beq=np.concatenate([gN,[0]])
Aeq[-1,:nA]=1; Aeq[-1,nA:nA+p-1]=-1   # sum qt = sum mu
rows=[];b=[]
Aidx={pr:i for i,pr in enumerate(A)}
for i,(k,dl) in enumerate(pairs):
    r1=np.zeros(nv); r2=np.zeros(nv)
    if (k,dl) in Aidx: r1[Aidx[(k,dl)]]=1; r2[Aidx[(k,dl)]]=-1
    r1[nA+dl-1]=-net.cs[k]/p**n; r2[nA+dl-1]=net.cs[k]/p**n
    r1[nA+p-1+i]=-1; r2[nA+p-1+i]=-1
    rows+=[r1,r2]; b+=[0,0]
res2=linprog(c,A_ub=np.array(rows),b_ub=np.array(b),A_eq=Aeq,b_eq=beq,bounds=(0,None),method='highs')
if res2.status==0:
    tot=res2.x[:nA].sum(); print('min over KKT multipliers of TV(q, U x mu) = %.4f (normalised, max 2)'%(res2.fun/tot))
else: print('LP status',res2.status,res2.message)
# Fourier transform of q (S_n-symmetric): average over x in class of rho^{-chi.x} depends on chi pattern
# compute qhat(chi,delta) for chi = c*1_T patterns with |T|=t, c in 1..p-1 via generating function
rho=np.exp(2j*np.pi/p)
def qhat_pattern(chi_counts):
    # chi_counts[c] = number of slots with chi value c (c=0..p-1), sum = n; returns array over delta
    out=np.zeros(p-1,complex)
    for (k,dl),qq in zip(A,q):
        N=net.Nc[k]
        # E over x uniform in class N of rho^{-chi.x}: coefficient extraction; brute force via multinomial generating function
        # = [prod_c (sum_a z_a rho^{-c a})^{chi_c}] coefficient of z^N / multinomial(N)
        poly={(0,)*p:1.0+0j}
        for cval,cnt in enumerate(chi_counts):
            for _ in range(cnt):
                new={}
                for key,v in poly.items():
                    for a in range(p):
                        if key[a]+1>N[a]: continue
                        kk=list(key); kk[a]+=1; kk=tuple(kk)
                        new[kk]=new.get(kk,0)+v*rho**(-cval*a)
                poly=new
        out[dl-1]+=qq*poly.get(tuple(N),0)/net.cs[k]
    return out
print('|qhat| at patterns (chi values multiset) summed over delta:')
for t in range(0,n+1,max(1,n//6)):
    for cval in range(1,p):
        cc=[0]*p; cc[0]=n-t; cc[cval]+=t
        v=qhat_pattern(cc)
        print('  T size %d value %d: |qhat| per delta'%(t,cval), np.round(np.abs(v),4))
