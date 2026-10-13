"""Programmatic raw-MSA coverage/phylogenetic diagnostics; no outcome data."""
import numpy as np,json
from collections import Counter
seqs=[]; seq=''
for line in open('raw_evidence/PABP_YEAST_full_11-26-2021_b07.a2m'):
 if line.startswith('>'):
  if seq: seqs.append(seq)
  seq=''
 else:seq+=line.strip()
seqs.append(seq)
c=json.load(open('case.json')); ref=c['reference_sequence']; assert seqs[0].upper()==ref
X=np.array([list(s) for s in seqs]);wt=np.array(list(ref)); valid=np.char.isupper(X)&np.isin(X,list('ACDEFGHIKLMNPQRSTVWY'))
mask=np.array([a.isupper() for a in seqs[0]])
# Condition on other folded RNA recognition motifs, excluding altered RRM2.
background=mask & np.array([(35<=p<=120) or (205<=p<=410) for p in range(1,578)])
coverage=valid[:,background].mean(1)
identity=((X[:,background]==wt[background])&valid[:,background]).sum(1)/np.maximum(1,valid[:,background].sum(1))
local=mask & np.array([126<=p<=200 for p in range(1,578)])
local_identity=((X[:,local]==wt[local])&valid[:,local]).sum(1)/np.maximum(1,valid[:,local].sum(1))
print('unique ordered-domain sequences',len(set(''.join(s[background]) for s in X)))
print('unique RRM2 sequences',len(set(''.join(s[local]) for s in X)))
print('background identity percentiles',np.percentile(identity[coverage>.8],[0,25,50,75,90,95,99,100]))
for t in [.5,.6,.7,.8,.9]:
 idx=(identity>=t)&(coverage>=.8); print('threshold',t,'rows',idx.sum(),'unique local',len(set(''.join(s[local]) for s in X[idx])),'RRM2 id mean',local_identity[idx].mean())
print('candidate selected WT frequency vs near background identity>.7')
near=(identity>.7)&(coverage>.8)
for p in [129,131,132,134,135,138,139,142,144,150,152,154,155,156,163,164,165,168,170,172,174,177,179,180,181,182,184,188,194,195,199,200]:
 idx=valid[:,p-1]&near
 z=Counter(X[idx,p-1]); den=sum(z.values());print(p,ref[p-1],den,[(a,round(n/den,3)) for a,n in z.most_common(5)])
np.savez_compressed('msa_diagnostics_arrays.npz',background_identity=identity,background_coverage=coverage,local_identity=local_identity)
