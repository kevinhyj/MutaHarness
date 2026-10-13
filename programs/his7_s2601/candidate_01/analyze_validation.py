from experiment import *
import itertools
c=json.load(open('case.json'));rs=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in rs]);multi=np.array([len(mutations(r,ref))>1 for r in rs]); g=np.array([group(mutations(r,ref)) for r in rs])
P={};S={};C={}
for stem in ['validation2','validation']:
 try:
  d=json.load(open(stem+'_results.json')); z=np.load(stem+'_predictions.npz');P.update({k:z[k][:,multi] for k in z.files}); S.update(d['scores']);C.update(d['configs'])
 except FileNotFoundError:pass
ranked=sorted(S,key=lambda k:np.mean(S[k]),reverse=True)
for k in ranked[:15]:print('top',round(np.mean(S[k]),4),k)
choices=[]
for meth in ['kernel','en','poly','rbf','rf','et','gb']:
 ks=[k for k in ranked if C[k][0]==meth]
 if ks:choices.append(ks[0])
yp=y[multi]
R={k:np.array([rankdata(p)/len(p) for p in P[k]]) for k in choices}
for k in choices:print('CLASS',round(np.mean(S[k]),4),k)
for a,b in itertools.combinations(choices,2):
 for weight in [.25,.5,.75]:
  pp=weight*R[a]+(1-weight)*R[b];score=np.mean([metric(yp,p) for p in pp])
  print('BLEND',round(score,4),weight,a,b)
# save shortlisted configurations for independent split sensitivity checks.
json.dump([C[k] for k in choices],open('shortlist.json','w'),indent=2)
