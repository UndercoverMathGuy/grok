"""Exact full-weight second-order certificate at p=3,n=2,q=2."""
import json
from pathlib import Path
import sympy as s

p=3; nu=3; width=3
x=s.symbols('x:27')
def at(j,i,a): return x[(j*width+i)*p+a]
base=[]
for j in range(nu):
 for i in range(width):
  for a in range(p):
   base.append(s.expand_trig(s.cos(2*s.pi*a/p+(1 if j<2 else 2)*s.pi*i/nu)).simplify())
sub=dict(zip(x,base))
gaps=[]
for a in range(p):
 for b in range(p):
  y=(a+b)%p
  for D in (1,2):
   c=(y+D)%p
   gaps.append(sum((at(2,i,y)-at(2,i,c))*at(0,i,a)*at(1,i,b) for i in range(width)))
assert set(s.simplify(g.subs(sub)) for g in gaps)=={s.Rational(9,8)}
assert sum(z*z for z in base).simplify()==s.Rational(27,2)
f=sum(gaps)/len(gaps)
J=s.Matrix(gaps).jacobian(x).subs(sub).applyfunc(s.simplify)
H=(s.hessian(f,x).subs(sub)-s.eye(len(x))/4).applyfunc(s.simplify)
assert s.Matrix([s.diff(f,z).subs(sub) for z in x])==s.Matrix(base)/4
N=s.Matrix.hstack(*J.nullspace()).applyfunc(s.simplify)
Q=(-N.T*H*N).applyfunc(s.simplify)
# Congruence elimination: positive pivots, zero remaining block certifies PSD.
R=Q.copy(); pivots=[]
for i in range(R.rows):
 pivot=s.simplify(R[i,i])
 if pivot==0:
  assert all(s.simplify(R[i,j])==0 for j in range(i,R.cols))
  continue
 assert pivot.is_positive, pivot
 pivots.append(str(pivot))
 for j in range(i+1,R.rows):
  for k in range(i+1,R.cols):
   R[j,k]=s.simplify(R[j,k]-R[j,i]*R[i,k]/pivot)
# Two exact constant-logit phase motions: phi(input j) += t, phi(readout) += t.
phase=[]
for slot in range(2):
 v=[]
 for j in range(nu):
  for i in range(width):
   for a in range(p):
    theta=2*s.pi*a/p+(1 if j<2 else 2)*s.pi*i/nu
    v.append(-s.sin(theta) if j in (slot,2) else s.Integer(0))
 phase.append(s.Matrix(v).applyfunc(s.simplify))
T=s.Matrix.hstack(*phase)
assert T.rank()==2 and J*T==s.zeros(J.rows,2)
assert N.T*H*T==s.zeros(N.cols,2)
assert Q.rows-len(pivots)==2
record={'p':p,'n':2,'q':2,'width':width,'weight_variables':len(x),
 'active_constraints':len(gaps),'active_jacobian_rank':J.rank(),
 'critical_dimension':N.cols,'positive_pivots':pivots,
 'restricted_negative_hessian_rank':len(pivots),'kernel_dimension':2,
 'kernel_equals_constant_logit_phase_tangent':True,
 'margin':'9/8','squared_norm':'27/2','norm_multiplier':'1/8',
 'scope':'Exact local second-order certificate allowing all raw weight directions; not a global bound.'}
Path(__file__).with_name('local_p3_exact.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
