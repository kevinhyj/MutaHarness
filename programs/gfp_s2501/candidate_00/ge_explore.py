import json,numpy as np
from scipy.stats import spearmanr,rankdata
from model_dev import *
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);x,z,ms=featurize(s,c['reference_sequence'])
v=json.load(open('validation_results.json'));splits=[(np.array(a),np.array(b)) for a,b in v['splits']];preds=v['predictions'];res=[]
for ceiling in [3.85,4.0,4.2]:
 for alpha in [.03,.1,.3,1,3]:
  for zw in [0.,.3,1.]:
   cfg=('ge',alpha,zw,ceiling);out=[];cors=[]
   for tr,te in splits:
    p=ge_pred(x[tr],x[te],z[tr],z[te],y[tr],alpha,zw,ceiling)
    out.append(p.tolist());cors.append(spearmanr(p,y[te]).statistic)
   preds[str(cfg)]=out;res.append((np.mean(cors),np.std(cors),cfg));print(round(np.mean(cors),4),cfg,flush=True)
# Nonlinear smooth kernels on shared features and optional normalization.
for mode in ['raw','rank','latent']:
 for alpha in [.03,.3,3]:
  for rbf in [1,10]:
   cfg=('rbf',alpha,mode,rbf);out=[];cors=[]
   for tr,te in splits:
    p=kernel_pred(x[tr],x[te],z[tr],z[te],y[tr],alpha,mode,.3,rbf=rbf)
    out.append(p.tolist());cors.append(spearmanr(p,y[te]).statistic)
   preds[str(cfg)]=out;res.append((np.mean(cors),np.std(cors),cfg));print(round(np.mean(cors),4),cfg,flush=True)
print('TOP',sorted(res,reverse=True)[:20])
json.dump({'predictions':preds,'results':v['results']+res,'splits':v['splits']},open('validation_ge.json','w'))
