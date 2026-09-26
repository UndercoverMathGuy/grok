"""Universal exact 2x2x2 block certificate, independent of p and support."""
import json
from pathlib import Path
import sympy as s
x=s.symbols('x:18')
def slot(j,i): return s.Matrix(x[(j*3+i)*2:(j*3+i+1)*2])
base=[]
for j in range(3):
 for i in range(3):
  angle=(1 if j<2 else 2)*s.pi*i/3
  base += [s.cos(angle),s.sin(angle)]
sub=dict(zip(x,base))
f=0
for i in range(3):
 a,b,c=[slot(j,i) for j in range(3)]
 f+=(a[0]*b[0]-a[1]*b[1])*c[0]+(a[0]*b[1]+a[1]*b[0])*c[1]
tensor=[sum(slot(0,i)[a]*slot(1,i)[b]*slot(2,i)[c] for i in range(3)) for a in range(2) for b in range(2) for c in range(2)]
J=s.Matrix(tensor).jacobian(x).subs(sub).applyfunc(s.simplify)
H=(s.hessian(f,x).subs(sub)-s.eye(18)).applyfunc(s.simplify)
assert s.Matrix([s.diff(f,z).subs(sub) for z in x])==s.Matrix(base)
N=s.Matrix.hstack(*J.nullspace()).applyfunc(s.simplify)
Q=(-N.T*H*N).applyfunc(s.simplify)
R=Q.copy();pivots=[]
for i in range(R.rows):
 pivot=s.simplify(R[i,i])
 if pivot==0:
  assert all(s.simplify(R[i,j])==0 for j in range(i,R.cols));continue
 assert pivot.is_positive
 pivots.append(str(pivot))
 for j in range(i+1,R.rows):
  for k in range(i+1,R.cols): R[j,k]=s.simplify(R[j,k]-R[j,i]*R[i,k]/pivot)
phase=[]
for selected in range(2):
 v=[]
 for j in range(3):
  for i in range(3):
   angle=(1 if j<2 else 2)*s.pi*i/3
   v += [-s.sin(angle),s.cos(angle)] if j in (selected,2) else [0,0]
 phase.append(s.Matrix(v))
T=s.Matrix.hstack(*phase)
assert T.rank()==2 and J*T==s.zeros(8,2) and N.T*H*T==s.zeros(N.cols,2)
assert Q.rows-len(pivots)==2
# A stronger, coordinate-invariant gap: -H >= I on the critical space normal
# to the two phase motions. Equivalently -H + projection_phase - I >=0 on ker J.
P=T*(T.T*T).inv()*T.T
Gap=(N.T*(-H+P-s.eye(18))*N).applyfunc(s.simplify)
# Check this gap with exact congruence elimination too.
R=Gap.copy();gap_pivots=[]
for i in range(R.rows):
 pivot=s.simplify(R[i,i])
 if pivot==0:
  assert all(s.simplify(R[i,j])==0 for j in range(i,R.cols));continue
 assert pivot.is_positive,pivot
 gap_pivots.append(str(pivot))
 for j in range(i+1,R.rows):
  for k in range(i+1,R.cols): R[j,k]=s.simplify(R[j,k]-R[j,i]*R[i,k]/pivot)
record={'variables':18,'tensor_constraints':8,'jacobian_rank':J.rank(),'critical_dimension':N.cols,
 'negative_hessian_positive_pivots':pivots,'kernel_dimension':2,'kernel_is_phase_tangent':True,
 'unit_spectral_gap_on_normal_space':True,'gap_positive_pivots':gap_pivots}
Path(__file__).with_name('local_block_exact.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
