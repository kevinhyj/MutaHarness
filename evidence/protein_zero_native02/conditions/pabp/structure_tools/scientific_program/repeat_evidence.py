"""Align two other RRM domains in the supplied WT protein to RRM2.
This is internal repeat compatibility, NOT a public homolog MSA or evolutionary
fitness/depth. Rigid alignment of WT CA atoms is not mutant relaxation.
"""
import json
import numpy as np
from pathlib import Path

def build(case,evidence):
 seq=case['reference_sequence'];sites={s['position']:s for s in evidence['raw_coordinate_evidence']['sites']}
 ca={p:np.array(s['CA_coordinate_A']) for p,s in sites.items()}
 out={p:[] for p in range(126,201)};alignments=[]
 anchors=list(range(128,134))+list(range(166,174))
 for delta,lo,hi in [(93,219,296),(196,322,399)]:
  xa=np.array([ca[p+delta] for p in anchors]);ya=np.array([ca[p] for p in anchors]);xc=xa.mean(0);yc=ya.mean(0)
  u,_,vt=np.linalg.svd((xa-xc).T@(ya-yc));rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
  transformed={p:(ca[p]-xc)@rot+yc for p in range(lo,hi+1)}
  rmsd=float(np.sqrt(np.mean(np.sum(((xa-xc)@rot+yc-ya)**2,axis=1))))
  # Globally ordered structural alignment. Distance alone; no sequence
  # substitution table is used to select the correspondence.
  ts=list(range(126,201));qs=list(range(lo,hi+1));n=len(ts);m=len(qs)
  dist=np.array([[np.linalg.norm(ca[p]-transformed[q]) for q in qs] for p in ts])
  similarity=2.0*np.exp(-(dist/3)**2)-.6
  dp=np.zeros((n+1,m+1));tr=np.zeros_like(dp,dtype=int);gap=-.65
  dp[:,0]=np.arange(n+1)*gap;dp[0,:]=np.arange(m+1)*gap
  for i in range(1,n+1):
   for j in range(1,m+1):
    vals=[dp[i-1,j-1]+similarity[i-1,j-1],dp[i-1,j]+gap,dp[i,j-1]+gap]
    k=int(np.argmax(vals));dp[i,j]=vals[k];tr[i,j]=k
  i,j=n,m;pairs=[]
  while i and j:
   k=tr[i,j]
   if k==0:
    p,q=ts[i-1],qs[j-1];d=float(dist[i-1,j-1]);confidence=min(sites[p]['source_asserted_AF2_pLDDT'],sites[q]['source_asserted_AF2_pLDDT'])
    # Omit ambiguous loop matches rather than pretending full coverage.
    if d<=3.2 and confidence>=75:
     entry=dict(position=q,WT=seq[q-1],aligned_CA_distance_A=d,source_pLDDT=confidence,domain_delta=delta)
     out[p].append(entry);pairs.append([p,q,d])
    i-=1;j-=1
   elif k==1:i-=1
   else:j-=1
  alignments.append(dict(domain_start=lo,domain_end=hi,anchor_CA_RMSD_A=rmsd,pairs=pairs[::-1]))
 return dict(format='within_reference_RRM_WT_structural_repeat_v1',limitations=['Two repeats from the SAME reference sequence, not independent homologs or measured outcomes.','Repeat RNA specificity can differ; compatibility is only a weak prior.','Static WT rigid superpositions; no mutant structure or epistasis.'],alignments=alignments,sites={str(p):x for p,x in out.items()})
if __name__=='__main__':
 case=json.load(open('case.json'));ev=json.load(open('structure_evidence.json'));r=build(case,ev)
 Path('internal_repeat_evidence.json').write_text(json.dumps(r,indent=2)+'\n')
 print('RMSDs',[(a['domain_start'],round(a['anchor_CA_RMSD_A'],3),len(a['pairs'])) for a in r['alignments']])
 for p,x in r['sites'].items():
  print(p,case['reference_sequence'][int(p)-1],[(q['position'],q['WT'],round(q['aligned_CA_distance_A'],1)) for q in x])
