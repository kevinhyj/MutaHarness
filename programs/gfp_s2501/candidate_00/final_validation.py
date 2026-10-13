import json,numpy as np,time
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));s=c['support'];ref=c['reference_sequence'];idx=np.array([i for i,r in enumerate(s) if len(proposal.mutations(r,ref))>1]);single=np.setdiff1d(np.arange(len(s)),idx)
results=[]
for seed in [209,317,421]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(idx):
  train=np.r_[single,idx[tr]];test=idx[te];cc=dict(c);cc['support']=[s[i] for i in train];cc['query']=[{k:v for k,v in s[i].items() if k!='fitness'} for i in test]
  p=proposal.propose(cc,{});corr=float(spearmanr([p['scores'][s[i]['id']] for i in test],[s[i]['fitness'] for i in test]).statistic);results.append(corr)
report={'split_seeds':[209,317,421],'n_folds':4,'multi_feedback_spearman':results,'mean':float(np.mean(results)),'std':float(np.std(results))}
json.dump(report,open('final_validation_results.json','w'),indent=2);print(report)
