import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,collections
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));s=c['support'];fb=np.arange(32,len(s));r={}
t=time.time()
for kind,seeds in [('feedback',[2026,578,341,937,1311]),('mixed',[2044,721,1711])]:
 vals=[]
 for seed in seeds:
  population=fb if kind=='feedback' else np.arange(len(s))
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(population):
   tr=population[tr];te=population[te]
   if kind=='feedback':tr=np.r_[np.arange(32),tr]
   cc=dict(c,support=[s[i] for i in tr],query=[{k:v for k,v in s[i].items() if k!='fitness'} for i in te])
   out=proposal.propose(cc,{})['scores']
   vals.append(spearmanr([s[i]['fitness'] for i in te],[out[s[i]['id']] for i in te]).statistic)
 r[kind]={'mean_spearman':float(np.mean(vals)),'fold_sd':float(np.std(vals)),'folds':len(vals)}
# Mutation-disjoint stress test: every test record has at least one mutation
# token absent from training. This is harder than ordinary variant holdout.
tokens=sorted({m for row in s for m in row['mutation'].split(':')})
vals=[];train_n=[];test_n=[]
for seed in range(6):
 rng=np.random.default_rng(810+seed);held=set(rng.choice(tokens,size=20,replace=False))
 te=[i for i,row in enumerate(s) if held.intersection(row['mutation'].split(':'))];tr=[i for i in range(len(s)) if i not in te]
 cc=dict(c,support=[s[i] for i in tr],query=[{k:v for k,v in s[i].items() if k!='fitness'} for i in te])
 out=proposal.propose(cc,{})['scores'];vals.append(spearmanr([s[i]['fitness'] for i in te],[out[s[i]['id']] for i in te]).statistic)
 train_n.append(len(tr));test_n.append(len(te))
r['novel_mutation_stress']={'mean_spearman':float(np.mean(vals)),'fold_sd':float(np.std(vals)),'splits':len(vals),'train_sizes':train_n,'test_sizes':test_n}
o=proposal.propose(c,{});o2=proposal.propose(c,{})
assert o==o2
assert set(o['scores'])=={q['id'] for q in c['query']}
assert all(np.isfinite(v) and -1<=v<=1 for v in o['scores'].values())
assert o['selected_ids']==[]
r['total_validation_seconds']=time.time()-t
print(json.dumps(r,indent=2));json.dump(r,open('final_validation.json','w'),indent=2)
