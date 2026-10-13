import json,collections
from pathlib import Path
import numpy as np
c=json.load(open('case.json'));ref=c['reference_sequence']
rows=[];seq='';head=None;names=[]
with open('raw_evidence/UBE4B_HUMAN_2023-08-07_b04.a2m') as f:
 for line in f:
  line=line.strip()
  if line.startswith('>'):
   if head is not None:rows.append(seq);names.append(head)
   head=line[1:];seq=''
  else:seq+=line
 if head is not None:rows.append(seq);names.append(head)
query=rows[0];aligned_query=''.join(x for x in query if x.isupper() or x=='-')
map_positions=[];p=0
for x in query:
 if x.isalpha():p+=1
 if x.isupper() or x=='-':map_positions.append(p if x!='-' else None)
assert query.replace('-','').upper()==ref
aligned=[''.join(x for x in r if x.isupper() or x=='-') for r in rows]
assert len(set(map(len,aligned)))==1
counts=collections.Counter(aligned[1:]);unique=list(counts)
arr=np.frombuffer(''.join(unique).encode(),dtype=np.uint8).reshape(len(unique),len(aligned_query))
qarr=np.frombuffer(aligned_query.encode(),dtype=np.uint8)
observed=arr!=ord('-');coverage=observed.mean(1);ident=((arr==qarr)&observed).sum(1)/np.maximum(observed.sum(1),1)
idx=np.argsort(-ident)
print('rows',len(rows),'unique',len(unique),'columns',len(aligned_query),'mapped positions',map_positions)
print('query',aligned_query)
for t in [.2,.3,.35,.4,.5,.6,.7,.8,.9]:
 sel=(ident>=t)&(coverage>=.8)
 print('Threshold',t,'unique',sel.sum(),'identity median',np.median(ident[sel]) if sel.any() else None)
for i in idx[:20]:print('near',round(ident[i],3),round(coverage[i],3),counts[unique[i]],unique[i])
# Frequency summaries for a conservative homolog selection (>=40% id, >=80% coverage).
sel=(ident>=.4)&(coverage>=.8)
print('close PROFILE unweighted exploratory')
for j,p in enumerate(map_positions):
 freq=collections.Counter(chr(x) for x in arr[sel,j] if x!=ord('-'));tot=sum(freq.values())
 print(p,ref[p-1],tot,' '.join(f'{a}:{n/tot:.2f}' for a,n in freq.most_common(4)))
np.savez_compressed('alignment_inspection.npz',alignment=arr,identity=ident,coverage=coverage,mapping=np.array(map_positions),multiplicity=np.array([counts[u] for u in unique]))
