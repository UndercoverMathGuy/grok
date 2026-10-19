import numpy as np
r = 1/1.8552
def brute(P, T, n=161):
    g = np.linspace(-1, 3, n)
    L1, L2, L3 = np.meshgrid(g, g, g, indexing='ij')
    L4 = -(L1 + L2 + L3)
    ok = (L4 >= -1) & (L1**2+L2**2+L3**2+L4**2 <= 2*P) & (L1**3+L2**3+L3**3+L4**3 >= T)
    obj = np.where(ok, np.clip(1+L1,0,None)**r + np.clip(1+L2,0,None)**r + np.clip(1+L3,0,None)**r + np.clip(1+L4,0,None)**r, np.inf)
    i = np.unravel_index(np.argmin(obj), obj.shape)
    return obj[i], (L1[i], L2[i], L3[i], L4[i])
for P, T in [(1.017, -2.616), (2.437, -2.983), (1.17, -0.147), (0.26, -0.094)]:
    v, l = brute(P, T)
    print(P, T, "brute min", round(v, 5), np.round(sorted(l, reverse=True), 3))
