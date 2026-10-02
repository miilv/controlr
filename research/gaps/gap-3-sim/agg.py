import json, numpy as np, csv
R=json.load(open('results.json'))
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; w=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*p,100*(c-w),100*(c+w)
tests=['sig0','sig2','sig5','sig10','stale2','stale5','stale10','erase','shift']
groups={}
for k,v in R.items():
    base=k.split('|seed')[0]
    groups.setdefault(base,[]).append(v)
rows=[]
for base,vs in groups.items():
    row={'model':base}
    for t in tests:
        s=sum(v[t]['succ'] for v in vs); n=sum(v[t]['n'] for v in vs)
        row[t]=s; row['n']=n
    fs=sum(v['shift']['follow'] for v in vs); row['shift_follow']=fs
    rows.append(row)
with open('summary.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['model','n_per_cell']+[t+'_succ' for t in tests]+['shift_follow'])
    for r in rows: w.writerow([r['model'],r['n']]+[r[t] for t in tests]+[r['shift_follow']])
def pct(k,n): p,l,h=wil(k,n); return f'{p:.0f} [{l:.0f}-{h:.0f}]'
print('| model | n | σ0 | σ2mm | σ5mm | σ10mm | stale 2s (1cm) | 5s (2.5cm) | 10s (5cm) | erased | shift: follows LLM | Δerase pp |')
print('|'+'---|'*12)
for r in rows:
    n=r['n']
    print(f"| {r['model']} | {n} | "+' | '.join(f"{100*r[t]/n:.0f}" for t in tests[:8])+f" | {100*r['shift_follow']/n:.0f} | {100*(r['sig0']-r['erase'])/n:+.0f} |")
print()
for r in rows:
    n=r['n']; print(r['model'], 'sig5', pct(r['sig5'],n), 'stale5', pct(r['stale5'],n), 'stale10', pct(r['stale10'],n),'shift_follow',pct(r['shift_follow'],n))
