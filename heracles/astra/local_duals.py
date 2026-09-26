"""Exact active rows and interval-positive multipliers for the local theorem."""
import json
from pathlib import Path
import sympy as s
import mpmath as mp
from vertices import solve
ROOT=Path(__file__).resolve().parent
mp.iv.dps=80
records=[]
for p in (3,5,7,11,13,17):
 record=json.loads((ROOT/f'vertices_p{p}.json').read_text())
 F=s.QQ.algebraic_field(2*s.cos(2*s.pi/p));zero,one=F.zero,F.one
 alpha=F.from_sympy(2*s.cos(2*s.pi/p));h=(p-1)//2
 v=[]
 for coeffs in record['amplitudes_descending_polynomial_coefficients']:
  z=zero
  for c in coeffs:z=z*alpha+F.from_sympy(s.Rational(c))
  v.append(z)
 root=2*mp.iv.cos(2*mp.iv.pi/p)
 def iv(z):
  out=mp.iv.mpf(0)
  for c in z.to_list():out=out*root+mp.iv.mpf(int(c.numerator))/int(c.denominator)
  return out
 t=[2*one,alpha]
 for j in range(2,h+1):t.append(alpha*t[-1]-t[-2])
 A=[[one-t[min((d*k)%p,(-d*k)%p)]/2 for k in range(1,h+1)] for d in range(1,h+1)]
 S=[j for j,z in enumerate(v) if z!=zero]
 T=[i for i in range(h) if sum((A[i][j]*v[j] for j in S),zero)==one]
 assert len(S)==len(T)
 matrix=[[A[i][j] for i in T] for j in S]
 inverse_columns=[solve(matrix,[one if i==j else zero for i in range(len(S))],zero) for j in range(len(S))]
 cost=sum((iv(v[j])**(mp.iv.mpf(2)/3) for j in S),mp.iv.mpf(0))
 rhs=[iv(v[j])**(-mp.iv.mpf(1)/3)/cost for j in S]
 weights=[sum((iv(inverse_columns[j][i])*rhs[j] for j in range(len(S))),mp.iv.mpf(0)) for i in range(len(T))]
 assert all(z.a>0 for z in weights)
 out={'p':p,'support':[j+1 for j in S],'active_positive_offsets':[i+1 for i in T],
      'positive_multiplier_intervals':[str(z) for z in weights],
      'normalization':'Sum weights = 1 exactly by (Av)_T=1 and sum(v^(2/3))/C=1.',
      'local_theorem_hypotheses_certified':True}
 records.append(out);print({'p':p,'minimum_multiplier_lower_bound':float(min(z.a for z in weights))},flush=True)
(ROOT/'local_duals.json').write_text(json.dumps(records,indent=2)+'\n')
