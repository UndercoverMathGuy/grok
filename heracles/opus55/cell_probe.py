import time
exec(open('v3_certify.py').read().split('fails=[]')[0])
for cell in [(0.2,0.25,1.01,1.1),(0.2,0.2125,1.01,1.0325),(0.3,0.31,1.3,1.35),(0.45,0.5,1.1,1.3)]:
    t0=time.time(); da,db,ea,eb=cell
    t=TGT(da); tr=ipow(iv.mpf(ea),R); hb=hbound(*cell); cb=cbound(*cell)
    print(cell, "target",float(t.b),"trivial",float(tr.a),"H",None if hb is None else float(hb.a),"C",None if cb is None else float(cb.a), f"{time.time()-t0:.2f}s")
