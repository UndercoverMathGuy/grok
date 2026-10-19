import numpy as np, sys
sys.argv = ['x', '1', '0']
src = open('hs_sup.py').read().split("rng = np.random")[0]
exec(src)
v = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
A = np.exp(1j * v[0]); B, C, D = 1.0, 1.0, 1.0
print("F", F_of(A, B, C, D))
print("BHS", B_HS(A, [B, C, D], 1.0))
print("ratio", ratio(v, verbose=True))
v0 = np.array([-0.02, 1.005, 0.978, 0.946, 0.072])
print("ratio v0", ratio(v0, verbose=True))
from scipy.optimize import minimize
res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 20})
print(res)
