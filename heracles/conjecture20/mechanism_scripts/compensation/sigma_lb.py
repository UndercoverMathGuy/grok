# Conservative safety ratio using G*(rs,rt) <= Gsym(max(rs,rt)):  sigma_lb = min_{rs,rt} max(Pi1(rs),Pi12(rt))/Gsym(max(rs,rt))
# = min_rho  max( Pi1(rho), Pi12(rho) ... ) -- the min over (rs,rt) with max = rho is attained at rs=rt=rho? No: Pi increasing, so
# for fixed m=max(rs,rt): min over the pair of max(Pi1(rs),Pi12(rt)) = min(Pi1(m), Pi12(m)) (put the other at 0).
import json,numpy as np,sys
s=float(sys.argv[1]); cur=json.load(open(sys.argv[2])); Gs=json.load(open(f'G_sym_s{s}.json'))
gr=np.array(sorted(float(k) for k in Gs)); gv=np.array([Gs[str(k)] if str(k) in Gs else Gs[repr(k)] for k in gr])
c1=1-3**(1-s); out=[]
for key,c in cur.items():
    rho=np.array(c['rho']); P1=np.array(c['Pi1']); P2=np.array(c['Pi12'])
    m=(rho>=gr.min()-1e-12)
    G=np.interp(rho[m],gr,gv); val=np.minimum(P1[m],P2[m])/G
    i=np.argmin(val)
    # linear-gain verdicts (T-blind, R = max_T): sigma_lin(g) = min_R min(Pi1,Pi12)(R)/(g R)
    lin=np.min(np.minimum(P1,P2)/rho)
    out.append((c['p'],c['l'],c['h1'],c['h12'],val[i],rho[m][i],lin/(0.5),lin/c1))
out.sort()
for o in out: print(f"p={o[0]:3d} l={o[1]:3d} h1={o[2]:.3f} h12={o[3]:.3f}  sigma_lb={o[4]:.3f} at rho={o[5]:.2f}   lin: vs R/2 {o[6]:.3f}  vs c1R {o[7]:.3f}")
print('min sigma_lb',min(o[4] for o in out))
