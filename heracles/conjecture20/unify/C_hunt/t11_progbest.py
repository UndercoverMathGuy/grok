import json, time, hlib2 as H2
sup = json.load(open('/home/user/grok/heracles/conjecture20/mechanism_scripts/relations3/supports.json'))
for p in [29, 31, 37, 41, 43, 47, 53]:
    t = time.time(); R = H2.prog_best(p, 1.5)
    print(f"b=2 p={p}: prog_best={R['G']:.12f} supp={R['supp']} | stored={sup[str(p)]['Gamma']:.12f} {sup[str(p)]['S']} ({time.time()-t:.0f}s)", flush=True)
for p in [29, 31, 37, 41, 43]:
    t = time.time(); R = H2.prog_best(p, 1.2)
    print(f"b=2.5 p={p}: prog_best={R['G']:.12f} K*={len(R['supp'])} supp={R['supp']} ({time.time()-t:.0f}s)", flush=True)
