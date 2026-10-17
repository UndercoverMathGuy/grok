import json, glob
D='./'
def gamma_table(s=1.5):
    out={}
    fns=glob.glob(D+f'gam_s{s}_*.json')
    if s==1.5: fns.append('../relations2/supports_s1.5.json')
    for fn in fns:
        try: d=json.load(open(fn))
        except Exception: continue
        for p,v in d.items():
            if p not in out or v['Gamma']>out[p]['Gamma']: out[p]=v
    return {int(k):v for k,v in out.items()}
