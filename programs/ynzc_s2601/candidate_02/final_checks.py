"""Exercise the actual reusable artifact, including fits on altered supports."""
import copy,json,time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
import proposal

c=json.load(open('case.json'));a=json.load(open('assignment.json'));t=time.time()
r1=proposal.propose(c,a);r2=proposal.propose(c,a)
assert r1==r2
assert set(r1['scores'])=={r['id'] for r in c['query']}
assert r1['selected_ids']==[]
assert all(np.isfinite(v) and -1<=v<=1 for v in r1['scores'].values())
rev=copy.deepcopy(c);rev['query'].reverse();rev['support'].reverse()
assert proposal.propose(rev,a)==r1
changed=copy.deepcopy(c)
for r in changed['support']:r['fitness']=-r['fitness']
assert proposal.propose(changed,a)['scores']!=r1['scores']
empty=copy.deepcopy(c);empty['support']=[]
assert all(v==0 for v in proposal.propose(empty,a)['scores'].values())
s=c['support'];y=np.array([r['fitness'] for r in s]);order=np.array([r['mutation_order'] for r in s]);foldscores=[]
for seed in [101,137,191,229,271,307]:
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(y,order):
        sub={'reference_sequence':c['reference_sequence'],'support':[s[i] for i in tr],
             'query':[{k:v for k,v in s[i].items() if k!='fitness'} for i in te]}
        out=proposal.propose(sub,{'batch_size':0})
        pred=np.array([out['scores'][s[i]['id']] for i in te])
        foldscores.append(float(spearmanr(y[te],pred).statistic))
res={'deterministic':True,'order_invariant':True,'changed_labels_change_predictions':True,
     'all_query_ids':len(r1['scores']),'fold_spearman_mean':float(np.mean(foldscores)),
     'fold_spearman_sd':float(np.std(foldscores)),
     'elapsed_seconds':time.time()-t}
json.dump(res,open('final_checks.json','w'),indent=2);print(res)
