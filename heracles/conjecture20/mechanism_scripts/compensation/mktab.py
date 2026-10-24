# LaTeX tables from the computed JSON/outputs.
import json,numpy as np,glob,sys,re,subprocess
def progtab(s,fn):
    rows=[]
    for line in open(fn):
        d=json.loads(line); rows.append(f"{d['p']} & {d['Gamma']:.5f} & $\\{{{','.join(map(str,d['S']))}\\}}$ & $\\{{{','.join(map(str,d['Bhalf']))}\\}}$ & {', '.join(f'{v:.3f}' for v in d['lam'])}\\\\")
    return "\n".join(rows)
def parse_all(s,var='base'):
    out=subprocess.run(['python3','sigma_all.py',str(s),var],capture_output=True,text=True).stdout
    res={}
    for line in out.splitlines():
        m=re.match(r'\s*(\d+)\s+(\d+)\s+\|\s+([\d.inf]+).*?\|\s+([\d.naninf]+).*?\|\s+([\d.naninf]+).*?\|\s+([\d.naninf]+).*?\|\s+([\d.inf]+)',line)
        if m: res[(int(m.group(1)),int(m.group(2)))]=[float(m.group(k)) for k in range(3,8)]
    return res
def parse_orig(fn):
    res={}
    for line in open(fn):
        m=re.match(r'p=\s*(\d+) l=\s*(\d+) sigma_orig_fine=([\d.]+)',line)
        if m: res[(int(m.group(1)),int(m.group(2)))]=float(m.group(3))
    return res
if __name__=="__main__":
    s=float(sys.argv[1]); c1=1-3**(1-s)
    cur=json.load(open(f'curves2_s{s}_base.json'))
    A=parse_all(s); O=parse_orig(f'refine{"15" if s==1.5 else "12"}.txt')
    J={}
    for fn in glob.glob(f'joint_s{s}*.out'):
        for line in open(fn):
            m=re.match(r'p=(\d+) l=(\d+): sigma_joint=([\d.]+)',line)
            if m: k_=(int(m.group(1)),int(m.group(2))); J[k_]=min(J.get(k_,9),float(m.group(3)))
    for k,c in sorted(cur.items(),key=lambda kv:(kv[1]['p'],kv[1]['l'])):
        p,l=c['p'],c['l']; a=A.get((p,l),[np.nan]*5)
        j=f"{J[(p,l)]:.3f}" if (p,l) in J else "--"
        print(f"{p} & {l} & {c['h1']:.3f} & {abs(c['h12']):.3f} & {c['h1']/0.5:.2f} & {c['h1']/c1:.2f} & {O.get((p,l),np.nan):.3f} & {a[0]:.3f} & {j} & {a[1]:.2f} & {a[3]:.2f} & {a[4]:.2f}\\\\")
