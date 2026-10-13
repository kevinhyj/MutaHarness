from pathlib import Path
import json,re,numpy as np
from scipy.spatial.distance import cdist
c=json.load(open('case.json'));wt=c['reference_sequence'];aa='ACDEFGHIKLMNPQRSTVWY'
seqs=[];cur=''
for line in next(Path('raw_evidence').glob('*.a3m')).read_text().splitlines():
 if line.startswith('>'):
  if cur:seqs.append(re.sub('[a-z.]','',cur))
  cur=''
 else:cur+=line.strip()
if cur:seqs.append(re.sub('[a-z.]','',cur))
assert seqs[0]==wt
seqs=list(dict.fromkeys(s for s in seqs[1:] if s!=wt and len(s)==len(wt)))
a=np.array([list(s) for s in seqs]);core=a[:,8:38]
keep=(a[:,12]=='W') & np.isin(a[:,23],list('FY')) & (a[:,27]=='H') & (a[:,34]=='W') & (np.isin(core,list(aa)).mean(1)>=.9)
a=a[keep];core=a[:,8:38];enc=np.array([[ord(v) for v in s] for s in core],dtype=np.uint8)
w=np.zeros(len(a))
for start in range(0,len(a),256):
 d=cdist(enc[start:start+256],enc,metric='hamming');w[start:start+256]=1/np.maximum(1,(d<=.2+1e-9).sum(1))
print('binding-class-filtered rows',len(a),'redundancy-weighted depth',w.sum())
f=np.zeros((46,20))
for p in range(46):
 count=np.array([(w*(a[:,p]==v)).sum() for v in aa]);f[p]=(count+.05)/(count.sum()+1)
for p in [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]:
 orig=next(s['msa']['frequencies'] for s in c['evidence']['sites'] if s['position']==p)
 print(p,wt[p-1],','.join(f'{aa[k]}:{f[p-1,k]:.2f}' for k in np.argsort(-f[p-1])[:4]),'selected alts',','.join(f'{v}:{orig[v]:.3g}->{f[p-1,aa.index(v)]:.3g}' for v in sorted(set(q['sequence'][p-1] for q in c['query'])-{wt[p-1]})))
np.savez('raw_profile_diagnostics.npz',frequencies=f,weights=w,sequences=a)
