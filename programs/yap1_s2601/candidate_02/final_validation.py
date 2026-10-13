import json,time
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import proposal
c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);out={}
t=time.monotonic()
for tag,seeds in [('development',[17,41,73]),('feedback_assessment',[103,139,197,251]),('all_support_assessment',[113,181,277])]:
 vals=[]
 for seed in seeds:
  indices=np.arange(n) if tag.startswith('all') else np.arange(32,n)
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(indices):
   tr=indices[tr];te=indices[te]
   if not tag.startswith('all'):tr=np.r_[np.arange(32),tr]
   cc=dict(c,support=[s[i] for i in tr],query=[s[i] for i in te]+c['query'])
   p=proposal.propose(cc,{})['scores'];vals.append(spearmanr([p[s[i]['id']] for i in te],y[te]).statistic)
 out[tag]={'mean_spearman':float(np.mean(vals)),'sd_folds':float(np.std(vals)),'folds':vals};print(tag,out[tag],flush=True)
cc=dict(c,support=s[:32],query=s[32:]);p=proposal.propose(cc,{})['scores'];print('original_singles_to_feedback',spearmanr([p[r['id']] for r in s[32:]],y[32:]).statistic,flush=True)
start=time.monotonic();a=proposal.propose(c,{});b=proposal.propose(c,{})
assert a==b and set(a['scores'])=={r['id'] for r in c['query']} and a['selected_ids']==[]
assert all(np.isfinite(v) and -1<=v<=1 for v in a['scores'].values())
print('determinism_and_contract_ok',len(a['scores']),'full_fit_seconds',(time.monotonic()-start)/2,'validation_seconds',time.monotonic()-t,flush=True)
json.dump(out,open('final_validation_results.json','w'),indent=2)
