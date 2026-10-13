import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));S=c['support'];y=np.array([r['fitness'] for r in S]);B=proposal.kernel_blocks(S,c['reference_sequence']);Ks=[proposal.model_kernel(B,cfg) for cfg in proposal.CONFIGURATIONS[:3]]
result={}
for name,parts,feedback in [('feedback_75pct',4,True),('all_75pct',4,False),('feedback_50pct',2,True),('all_50pct',2,False)]:
 scores=[]
 for seed in [151,197,263]:
  inds=np.arange(32,96) if feedback else np.arange(96)
  for a,b in KFold(parts,shuffle=True,random_state=seed).split(inds):
   tr=np.r_[np.arange(32),inds[a]] if feedback else inds[a];te=inds[b];preds=[]
   for cfg,K in zip(proposal.CONFIGURATIONS[:3],Ks):
    target=.25*proposal.target_values(y[tr],'mix')+.75*proposal.target_values(y[tr],'rank')
    coef=cho_solve(cho_factor(K[np.ix_(tr,tr)]+cfg[4]*np.eye(len(tr)),check_finite=False),target,check_finite=False)
    preds.append(K[np.ix_(te,tr)]@coef)
   pred=.6*preds[0]+.2*preds[1]+.2*preds[2]
   scores.append(spearmanr(y[te],pred).statistic)
 result[name]={'mean_spearman':float(np.mean(scores)),'fold_sd':float(np.std(scores)),'folds':len(scores)}
print(result);json.dump(result,open('validation_final.json','w'),indent=2)
