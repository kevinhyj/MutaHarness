import numpy as np,json,collections
c=json.load(open('case.json'));wt=c['reference_sequence'];query=wt[3:67];res=[]
with open('raw_evidence/UBE4B_HUMAN_2023-08-07_b04.a2m') as f:
 for line in f:
  if line.startswith('>'):continue
  s=''.join(a for a in line.strip() if a.isupper() or a=='-')
  assert len(s)==64
  cov=sum(a!='-' for a in s)/64
  ident=sum(a==b for a,b in zip(s,query))/64
  if ident>=.35 and cov>=.85:res.append((s,ident))
for t in [.35,.4,.5,.6,.8]:
 ss=sorted({s for s,ident in res if ident>=t});print('threshold',t,'raw',sum(ident>=t for s,ident in res),'unique',len(ss),'cys',collections.Counter(s.count('C') for s in ss))
ss=sorted({s for s,ident in res if ident>=.4})
print('unique >=40pct frequencies')
for p in [4,7,8,13,17,20,23,25,28,29,31,33,35,37,39,41,42,45,55,57,61,63,66]:
 co=collections.Counter(s[p-4] for s in ss);n=sum(co.values())-co['-']; print(p,wt[p-1],'WT',round(co[wt[p-1]]/n,3),'top',[(a,round(nc/n,3)) for a,nc in co.most_common(4) if a!='-'])
json.dump({'query':query,'sequences':ss,'threshold':.4,'coverage_threshold':.85},open('homolog_subset.json','w'))
