import time,collections
src=open('v3_certify.py').read().split('fails=[]')[0]
exec(src)
calls=collections.Counter(); deep=[]
def check(da,db,ea,eb,depth=0):
    calls[depth]+=1
    if sum(calls.values())>4000: raise SystemExit
    if db<=0.05 and eb<=1.01: return
    t=TGT(da)
    if ipow(iv.mpf(ea),R).a>=t.b: return
    hb=hbound(da,db,ea,eb)
    if hb is not None and hb.a>=t.b: return
    cb=cbound(da,db,ea,eb)
    if cb is not None and cb.a>=t.b: return
    if depth>=12: deep.append((da,db,ea,eb,float(t.b),None if hb is None else float(hb.a),None if cb is None else float(cb.a))); return
    if (db-da)/2 >= (eb-ea):
        m=(da+db)/2; check(da,m,ea,eb,depth+1); check(m,db,ea,eb,depth+1)
    else:
        m=(ea+eb)/2; check(da,db,ea,m,depth+1); check(da,db,m,eb,depth+1)
try: check(0.2,0.5,1.01,1.1)
except SystemExit: pass
print("calls by depth",sorted(calls.items()))
for d in deep[:12]: print(d)
