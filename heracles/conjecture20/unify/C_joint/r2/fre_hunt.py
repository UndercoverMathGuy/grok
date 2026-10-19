import sys, glob, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_joint'); sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import EW, fields_x
def kap(p):
    F = (p-1)//2; return 1 - np.cos(2*np.pi*np.outer(np.arange(1, p), np.arange(1, F+1))/p)
worst = 0; cnt = 0
for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/C_hunt/out/best_b_*.npz')):
    d = np.load(f); net = EW.Net(d['U'], d['W']); h = net.margin()
    if h <= 0: continue
    Phi = fields_x(net); FRE = (kap(net.p) @ Phi.real.reshape(Phi.shape[0], -1)).min()
    worst = max(worst, (FRE - h)/h); cnt += 1
    print(f"{f.split('/')[-1]}: h={h:.6e}  FRE={FRE:.6e}  (FRE-h)/h={(FRE-h)/h:.2e}")
print(f"{cnt} HUNT networks: max (FRE-h)/h = {worst:.3e}")
