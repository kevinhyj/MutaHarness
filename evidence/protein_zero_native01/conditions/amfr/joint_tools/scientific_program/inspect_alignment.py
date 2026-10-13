"""Reproduce supplied natural-homolog selection/weights; summarize contact-site covariation."""
import json,numpy as np
from collections import Counter
records=[];seq=''
for line in open('raw_evidence/AMFR_HUMAN_2023-08-07_b04.a2m'):
 line=line.strip()
 if line.startswith('>'):
  if seq:records.append(seq)
  seq=''
 else:seq+=line
if seq:records.append(seq)
aligned=[''.join(a for a in s if a.isupper() or a=='-') for s in records]
unique=sorted(set(aligned))[:2048]
x=np.array([list(s) for s in unique]);valid=x!='-';neighbors=np.zeros(len(x))
for start in range(0,len(x),64):
 xx=x[start:start+64];vv=valid[start:start+64]
 shared=vv[:,None,:]&valid[None,:,:]
 depth=shared.sum(2);same=((xx[:,None,:]==x[None,:,:])&shared).sum(2)
 neighbors[start:start+64]=((same>=.8*np.maximum(depth,1))&(depth>=.5*x.shape[1])).sum(1)
w=1/np.maximum(1,neighbors)
print('neff',w.sum(),'target',631.201790072901)
profile=json.load(open('raw_evidence/msa_profile.json'))
for p in [9,28,32,38,41,46]:
 j=p-6;counts=Counter()
 for a,ww in zip(x[:,j],w):
  if a!='-':counts[a]+=ww
 dep=sum(counts.values());freq={a:(counts[a]+.05)/(dep+1) for a in 'ACDEFGHIKLMNPQRSTVWY'}
 print('site',p,'depth',round(dep,3),'maxfreqerror',max(abs(freq[a]-profile['positions'][str(p)]['frequency'][a]) for a in freq))
for p,q in [(28,38),(28,41),(32,38),(32,41),(38,41),(41,46)]:
 counts=Counter()
 for a,b,ww in zip(x[:,p-6],x[:,q-6],w):
  if a!='-' and b!='-':counts[a+b]+=ww
 print('pair',p,q,'effective coverage',round(sum(counts.values()),2),'top',[(ab,round(v,1)) for ab,v in counts.most_common(12)])
np.savez('alignment_reconstructed.npz',x=x,weights=w)
