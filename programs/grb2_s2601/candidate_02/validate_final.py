import json,copy,time,numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from proposal import propose
c=json.load(open('case.json'));s=c['support'];idx=np.array([i for i,r in enumerate(s) if r['mutation_order']==2]);singles=np.array([i for i,r in enumerate(s) if r['mutation_order']==1]);all_corr=[]
start=time.time()
for seed in [2601,731,1187]:
 pred={}
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(idx):
  cc=copy.deepcopy(c);cc['support']=[s[i] for i in np.r_[singles,idx[tr]]];cc['query']=[{k:v for k,v in s[i].items() if k!='fitness'} for i in idx[te]]
  pred.update(propose(cc,{})['scores'])
 cor=float(spearmanr([s[i]['fitness'] for i in idx],[pred[s[i]['id']] for i in idx]).statistic);all_corr.append(cor)
 print('Outer feedback holdout seed',seed,'Spearman',cor)
print('mean',np.mean(all_corr),'seconds',time.time()-start)
# Stress test without any feedback; never consult feedback fitness in fitting.
cc=copy.deepcopy(c);cc['support']=[s[i] for i in singles];cc['query']=[{k:v for k,v in s[i].items() if k!='fitness'} for i in idx];pred=propose(cc,{})['scores'];print('Singles only',spearmanr([s[i]['fitness'] for i in idx],[pred[s[i]['id']] for i in idx]).statistic)
