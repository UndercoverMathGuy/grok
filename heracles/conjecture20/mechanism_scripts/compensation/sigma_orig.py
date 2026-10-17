# Original (T-blind) game: riding + partner only, prover's single character at argmax_T, sharp gain Gsym(rho):
#   sigma_orig = min_rho Pi1(rho)/Gsym(rho);  also linear gains: h1/(1/2), h1/c1
import json,numpy as np,sys
s=float(sys.argv[1]); cur=json.load(open(sys.argv[2])); Gs=json.load(open(f'G_sym_s{s}.json'))
gr=np.array(sorted(float(k) for k in Gs)); gv=np.array([Gs[k] for k in sorted(Gs,key=float)])
c1=1-3**(1-s); rows=[]
for key,c in cur.items():
    rho=np.array(c['rho']); P1=np.array(c['Pi1']); m=rho>=gr.min()-1e-12
    val=P1[m]/np.interp(rho[m],gr,gv); i=np.argmin(val)
    rows.append((c['p'],c['l'],c['h1'],c['h1']/0.5,c['h1']/c1,val[i],rho[m][i]))
rows.sort()
for r_ in rows: print(f"p={r_[0]:3d} l={r_[1]:3d} h1={r_[2]:.3f}  lin R/2: {r_[3]:.3f}  lin c1R: {r_[4]:.3f}   sharp: sigma={r_[5]:.3f} at rho={r_[6]:.2f}")
print('min over (p,l): lin R/2 %.3f, lin c1 %.3f, sharp %.3f'%(min(r_[3] for r_ in rows),min(r_[4] for r_ in rows),min(r_[5] for r_ in rows)))
