import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));ss=c['support'];si=[r for r in ss if r['mutation_order']==1];mi=[r for r in ss if r['mutation_order']>1]
# Validation with the actual production entry point and unlabeled query pool.
for training_size in [48,32,16]:
 vals=[];lo=[];hi=[]
 for seed in [17413,17627,18401,19991]:
  rng=np.random.default_rng(seed);ix=rng.permutation(len(mi));tr=ix[:training_size];te=ix[training_size:]
  cc=dict(c);cc['support']=si+[mi[i] for i in tr];cc['query']=[mi[i] for i in te]+c['query']
  r=proposal.propose(cc,{})['scores'];ys=np.array([mi[i]['fitness'] for i in te]);pr=np.array([r[mi[i]['id']] for i in te]);o=np.array([mi[i]['mutation_order'] for i in te])
  vals.append(float(spearmanr(ys,pr).statistic));lo.append(float(spearmanr(ys[o<10],pr[o<10]).statistic));hi.append(float(spearmanr(ys[o>20],pr[o>20]).statistic))
 print('feedback train',training_size,'rho',vals,'mean',np.mean(vals),'low',np.mean(lo),'high',np.mean(hi),flush=True)
t=time.time();a=proposal.propose(c,{});b=proposal.propose(c,{})
assert a==b and len(a['scores'])==len(c['query']) and set(a['scores'])==set(r['id'] for r in c['query'])
assert a['selected_ids']==[] and all(np.isfinite(v) and -1<=v<=1 for v in a['scores'].values())
rev=dict(c);rev['support']=list(reversed(c['support']));rev['query']=list(reversed(c['query']));assert proposal.propose(rev,{})==a
print('contract, repeated-call and record-order determinism passed; three full fits seconds',time.time()-t,flush=True)
