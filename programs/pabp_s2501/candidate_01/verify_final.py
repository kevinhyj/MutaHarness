import json,time,numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));s=c['support'];fb=np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);orig=np.setdiff1d(np.arange(len(s)),fb)
start=time.time();results=[]
for seed in [1031,2051]:
 corr=[]
 for tr,va in KFold(4,shuffle=True,random_state=seed).split(fb):
  train=np.r_[orig,fb[tr]];valid=fb[va]
  sub={**c,'support':[s[i] for i in train],'query':[s[i] for i in valid]}
  result=proposal.propose(sub,{})
  corr.append(float(spearmanr([result['scores'][s[i]['id']] for i in valid],[s[i]['fitness'] for i in valid]).statistic))
 results.append({'seed':seed,'fold_spearman':corr,'mean':float(np.mean(corr))})
print(results);print('runtime',time.time()-start)
json.dump(results,open('final_implementation_validation.json','w'),indent=2)
