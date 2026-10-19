import numpy as np, sys
sys.argv = ['x', '1', '0']
exec(open('hs_sup.py').read().split("rng = np.random")[0])
for v0 in [np.array([0.123, 0.946, 0.864, 0.974, 0.089]), np.array([0.087, 1.058, 1.054, 1.041, 0.109])]:
    A=np.exp(1j*v0[0]); B=v0[1]; C=v0[2]; D=v0[3]*np.exp(1j*v0[4])
    print("F", F_of(A,B,C,D), "tw", twisted_lb(A,B,C,D,1.5,tries=6))
    print("ratio", ratio(v0, verbose=True))
