import numpy as np
exec(open('v3_combined2.py').read().split('worst=np.inf')[0])
for d in (0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.7):
    row=[]
    for e1 in (1.0,1.02,1.05,1.1,1.2,1.3,1.4,1.5,1.6,1.7):
        t=3**(1-r)*(1-d/2)**r
        row.append(max(boundC(d,e1),boundH(d,e1),e1**r)/t)
    print(f"delta={d:.2f}: "+" ".join(f"{v:.3f}" for v in row))
print("columns e1 =",(1.0,1.02,1.05,1.1,1.2,1.3,1.4,1.5,1.6,1.7))
