# Why does the embedded flat block lose?  Where are the worst wrong answers, and what does the block contribute there?
import numpy as np, json, sys
s=1.5; c=1-(2/3)**(s-1); p=int(sys.argv[1]); m=int(sys.argv[2]); f=float(sys.argv[3])
runs=json.load(open(f'/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/large_{p}.json'))['runs']
r0=max(runs,key=lambda r:r['Gamma']); SR=np.array(r0['S']); yR=np.array(r0['y']); G=r0['Gamma']
d=np.arange(1,p)
K=lambda S: 1-np.cos(2*np.pi*np.outer(S,d)/p)
KR=K(SR); covR=yR@KR; d0=int(d[np.argmin(covR)]); inv=pow(d0,-1,p)
fold=lambda l: min(l%p,p-l%p)
B=np.array(sorted({fold(((p-1)//2-j)*inv) for j in range(m)}-set(SR.tolist()))); KB=K(B)
aR=(1-f)**s*yR; aB=np.full(len(B),(f/len(B))**s)
cR=aR@KR; cB=aB@KB; tot=cR+cB
bind=np.argsort(tot)[:5]
print(f"p={p} m={len(B)} f={f}: block budget share {f:.2f}")
print(f"  at the old bottleneck d0={d0}: block supplies {cB[d0-1]/tot[d0-1]:.2f} of coverage (block avg strength {KB[:,d0-1].mean():.2f} of max 2)")
for i in bind:
    print(f"  new worst wrong answer d={d[i]:5d} (= {d[i]*inv%p:5d} x d0): coverage {tot[i]/G:.3f}xGamma, block supplies {cB[i]/tot[i]:.3f} of it (avg strength {KB[:,i].mean():.3f})")
print(f"  block's share of coverage averaged over all wrong answers: {cB.mean()/tot.mean():.3f}")
