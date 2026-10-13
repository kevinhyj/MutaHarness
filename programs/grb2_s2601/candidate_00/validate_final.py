import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import solve
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);fb=np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);orig=np.array([i for i,r in enumerate(s) if r['id'].startswith('S')]);report=[]
for setting,seeds,folds in [('feedback_4fold',[809,919,1021],4),('feedback_2fold',[1123,1229,1327],2),('mixed_4fold',[1429,1531],4)]:
 for seed in seeds:
  pool=np.arange(len(s)) if setting.startswith('mixed') else fb
  pp=np.zeros(len(s));bp=np.zeros(len(s));fold_scores=[]
  for tr,te in KFold(folds,shuffle=True,random_state=seed).split(pool):
   tr=pool[tr];te=pool[te]
   if not setting.startswith('mixed'):tr=np.r_[orig,tr]
   subset={**c,'support':[s[i] for i in tr],'query':[{k:v for k,v in s[i].items() if k!='fitness'} for i in te]}
   pred=proposal.predict_raw(subset);pp[te]=pred;fold_scores.append(float(spearmanr(y[te],pred).statistic))
   f=proposal.features(subset['support'],c['reference_sequence']);g=proposal.features(subset['query'],c['reference_sequence']);K=f['site']@f['site'].T+.2*f['order']@f['order'].T;Q=g['site']@f['site'].T+.2*g['order']@f['order'].T;bp[te]=proposal.regression(K,Q,y[tr],.4,'raw')
  report.append({'setting':setting,'seed':seed,'pooled_spearman':float(spearmanr(y[pool],pp[pool]).statistic),'fold_spearman':fold_scores,'position_only_spearman':float(spearmanr(y[pool],bp[pool]).statistic)})
json.dump(report,open('final_validation.json','w'),indent=2)
for setting in sorted({r['setting'] for r in report}):
 rs=[r for r in report if r['setting']==setting];print(setting,'ensemble:',[round(r['pooled_spearman'],4) for r in rs],'position baseline:',[round(r['position_only_spearman'],4) for r in rs])
# Contract checks on deliberately changed supplied supports.
a=proposal.propose(c,{});b=proposal.propose(c,{})
assert a==b and a['selected_ids']==[] and set(a['scores'])=={r['id'] for r in c['query']}
assert all(np.isfinite(x) and -1<=x<=1 for x in a['scores'].values())
small={**c,'support':s[:32]};assert proposal.propose(small,{})!=a
rev={**c,'support':list(reversed(s)),'query':list(reversed(c['query']))};assert proposal.propose(rev,{})==a
for records in [[],s[:1]]:
 assert all(np.isfinite(x) for x in proposal.propose({**c,'support':records},{})['scores'].values())
print('Determinism, support refitting, order invariance, and output contracts passed.')
