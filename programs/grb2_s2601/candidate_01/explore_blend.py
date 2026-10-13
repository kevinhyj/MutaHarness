import pickle,itertools,numpy as np,json
from scipy.stats import rankdata,spearmanr
c=json.load(open('case.json'));y=np.array([r['fitness'] for r in c['support']]);spl=np.load('cv_splits.npz');splits=[(spl[f'tr{i}'],spl[f'te{i}']) for i in range(15)]
tr=pickle.load(open('transfer_results.pkl','rb'));pool=pickle.load(open('pool_results.pkl','rb'));ref=pickle.load(open('refine_results.pkl','rb'))
candidates=[tr[0],next(r for r in tr if r[3]==('base',4,0.,0.,.1,.1)),next(r for r in tr if r[3]==('wt',4,0.,0.,2,0,2)),pool[0],pool[1],next(r for r in ref if r[3]==(.1,.25,.5,'wc',1,.3,.15))]
ranked=[[(rankdata(p)-.5)/len(p) for p in r[4]] for r in candidates]
res=[]
for w in itertools.product(range(5),repeat=6):
 if sum(w)!=8 or w[0]<2:continue
 ss=[]
 for i,(train,test) in enumerate(splits):
  p=sum(w[j]*ranked[j][i] for j in range(6));ss.append(spearmanr(p,y[test]).statistic)
 res.append((.75*np.mean(ss[:10])+.25*np.mean(ss[10:]),np.mean(ss[:10]),np.mean(ss[10:]),w))
res.sort(reverse=True)
for r in res[:10]:print(r)
print('candidate configs',[r[3] for r in candidates])
