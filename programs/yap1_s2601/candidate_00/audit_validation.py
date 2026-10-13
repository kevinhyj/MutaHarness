import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));S=c['support'];y=np.array([r['fitness'] for r in S]);B=proposal.kernel_blocks(S,c['reference_sequence'])
configs=list(proposal.CONFIGURATIONS)+[(.1,.05,.3,0,.1,.2,0)]
Ks=[proposal.model_kernel(B,cfg) for cfg in configs]
protocols={}
for name,seeds,parts,feedback in [('feedback_75pct',[151,197,263],4,True),('all_75pct',[151,197,263],4,False),('feedback_50pct',[151,197,263],2,True),('all_50pct',[151,197,263],2,False)]:
 fs=[]
 for seed in seeds:
  inds=np.arange(32,96) if feedback else np.arange(96)
  for tr,te in KFold(parts,shuffle=True,random_state=seed).split(inds):
   fs.append((np.r_[np.arange(32),inds[tr]] if feedback else inds[tr],inds[te]))
 protocols[name]=fs
report={}
for name,fs in protocols.items():
 scores={str(i):[] for i in range(len(configs))};scores['ensemble']=[];scores['ensemble_mix']=[];scores['ensemble_rank']=[];scores['additive_log']=[]
 for tr,te in fs:
  predictions=[];mixes=[];ranks=[]
  for i,(cfg,K) in enumerate(zip(configs,Ks)):
   fac=cho_factor(K[np.ix_(tr,tr)]+cfg[4]*np.eye(len(tr)),check_finite=False)
   p=[]
   for mode in ['mix','rank']:
    p.append(K[np.ix_(te,tr)]@cho_solve(fac,proposal.target_values(y[tr],mode),check_finite=False))
   pr=np.mean(p,axis=0);scores[str(i)].append(spearmanr(y[te],pr).statistic)
   if i<4:predictions.append(pr);mixes.append(p[0]);ranks.append(p[1])
  scores['ensemble'].append(spearmanr(y[te],np.mean(predictions,axis=0)).statistic)
  scores['ensemble_mix'].append(spearmanr(y[te],np.mean(mixes,axis=0)).statistic)
  scores['ensemble_rank'].append(spearmanr(y[te],np.mean(ranks,axis=0)).statistic)
  K=.3*B[0]+.5*B[1]+.2*B[2]+10
  fac=cho_factor(K[np.ix_(tr,tr)]+.2*np.eye(len(tr)),check_finite=False)
  pr=K[np.ix_(te,tr)]@cho_solve(fac,proposal.target_values(y[tr],'log'),check_finite=False)
  scores['additive_log'].append(spearmanr(y[te],pr).statistic)
 report[name]={k:{'mean_spearman':float(np.mean(v)),'fold_sd':float(np.std(v))} for k,v in scores.items()}
 print(name,report[name],flush=True)
json.dump(report,open('validation_audit.json','w'),indent=2)
# Contract and determinism on varied public training subsets.
for n in [1,16,32,48,72,96]:
 sub=dict(c); sub['support']=S[:n];a=proposal.propose(sub,{});b=proposal.propose(sub,{})
 assert a==b and len(a['scores'])==len(c['query']) and a['selected_ids']==[]
 assert all(np.isfinite(v) and -1<=v<=1 for v in a['scores'].values())
print('Subset contracts and determinism passed.')
