"""Build a closer-homolog prior from supplied public A2M; no phenotype data.
The supplied lexical subset is broad and includes metal-dependent RING domains.
Use >=40% amino-acid identity, >=80% covered aligned reference, then standard
inverse >=80% identity-neighbor weighting. Query itself is excluded; identical
homolog rows are collapsed. Query lowercase insertions have absent evidence.
"""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
ref=json.loads((P/'case.json').read_text())['reference_sequence'];AA='ACDEFGHIKLMNPQRSTVWY'
rows=[];seq='';head=None
with (P/'raw_evidence/UBE4B_HUMAN_2023-08-07_b04.a2m').open() as f:
 for line in f:
  line=line.strip()
  if line.startswith('>'):
   if head is not None:rows.append(seq)
   head=line[1:];seq=''
  else:seq+=line
 if head is not None:rows.append(seq)
query=rows[0];assert query.replace('-','').upper()==ref
positions=[];pos=0
for x in query:
 if x.isalpha():pos+=1
 if x.isupper() or x=='-':positions.append(pos if x!='-' else None)
assert None not in positions
aligned_query=''.join(x for x in query if x.isupper() or x=='-')
unique=sorted(set(''.join(x for x in r if x.isupper() or x=='-') for r in rows[1:]))
assert len(set(map(len,unique)))==1
arr=np.frombuffer(''.join(unique).encode(),dtype=np.uint8).reshape(len(unique),len(positions))
q=np.frombuffer(aligned_query.encode(),dtype=np.uint8)
canonical=np.isin(arr,np.frombuffer(AA.encode(),dtype=np.uint8))
coverage=canonical.mean(1);identity=((arr==q)&canonical).sum(1)/np.maximum(canonical.sum(1),1)
sel=(identity>=.4)&(coverage>=.8)
arr=arr[sel];canonical=canonical[sel];identity=identity[sel];coverage=coverage[sel]
n=len(arr);neighbor_counts=np.empty(n)
for start in range(0,n,64):
 # Pairwise identities over observed canonical overlap only.
 valid=canonical[start:start+64,None,:]&canonical[None,:,:]
 shared=valid.sum(2)
 identical=((arr[start:start+64,None,:]==arr[None,:,:])&valid).sum(2)
 neighbor_counts[start:start+64]=((identical>=.8*shared)&(shared>=.5*len(positions))).sum(1)
weights=1/neighbor_counts
# Weak, explicitly documented regularizer: total pseudocount 1, uniform over 20.
sites={};neff=float(weights.sum())
for j,p in enumerate(positions):
 depth=float(weights[canonical[:,j]].sum())
 counts={a:float(weights[arr[:,j]==ord(a)].sum()) for a in AA}
 sites[p]={'WT':ref[p-1],'effective_non_gap_depth':depth,'gap_fraction':1-depth/neff,'frequencies':{a:(v+.05)/(depth+1) for a,v in counts.items()},'counts':counts}
profile={'reference_sha256':hashlib.sha256(ref.encode()).hexdigest(),'msa_sha256':hashlib.sha256((P/'raw_evidence/UBE4B_HUMAN_2023-08-07_b04.a2m').read_bytes()).hexdigest(),'selection':{'identity_minimum':.4,'canonical_coverage_minimum':.8,'deduplicated':True,'selected_unique_rows':n,'all_unique_rows':len(unique),'query_excluded':True,'identity_neighbor_threshold':.8,'minimum_shared_coverage':.5,'pseudocount_total':1.,'pseudocount_background':'uniform20'},'neff':neff,'unmapped_positions':[p for p in range(1,len(ref)+1) if p not in positions],'sites':sites}
(P/'local_homolog_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
print('Selected',n,'Neff',round(neff,2),'absent',profile['unmapped_positions'])
for p in [8,13,17,20,23,28,29,31,39,41,42,45,55,61]:
 s=sites[p];f=s['frequencies'];print(p,s['WT'],'depth',round(s['effective_non_gap_depth'],1),'top',[(a,round(v,3)) for a,v in sorted(f.items(),key=lambda x:-x[1])[:4]])
