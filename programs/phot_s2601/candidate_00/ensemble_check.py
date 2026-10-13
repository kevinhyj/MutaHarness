exec(open('explore3.py').read().split('pd={};')[0])
from sklearn.model_selection import ShuffleSplit
cfgs=[('allele',.003,1),('allele',.01,1),('mixed',.003,1),('rbf',10,1),('allele',.003,1.5),('allele',.01,1.5),('add',0,1)]
results={}
for size in [48,32,24]:
 folds2=[(idx[tr],idx[te]) for tr,te in ShuffleSplit(20,train_size=size,random_state=918).split(idx)]
 for reg in [.01,.03,.1,.3,1]:
  for sw in [0,.1,1]:
   rr=[]
   for tr,te in folds2:
    tm=tr
    if sw:tm=np.r_[tr,[i for i in singles if any(x[i])]]
    a=x-x[tr].mean(0);b=xp-xp[tr].mean(0);ka=a@a.T;kb=b@b.T
    preds=[]
    for kind,strength,power in cfgs:
     k=ka.copy()
     if kind=='allele':k+=strength*ka*ka
     elif kind=='mixed':k+=strength*ka*kb
     elif kind=='rbf':k+=strength*np.exp(-((x[:,None]-x[None,:])**2).sum(2)/32)
     km=k[:,tm].mean(1);k=k-km[:,None]-km[None,:]+k[np.ix_(tm,tm)].mean()
     yt=np.maximum(y,0)**power;noise=np.full(len(tm),reg);noise[len(tr):]=reg/sw if sw else reg
     p=yt[tm].mean()+k[np.ix_(te,tm)]@np.linalg.solve(k[np.ix_(tm,tm)]+np.diag(noise),yt[tm]-yt[tm].mean())
     # monotone extension avoids artificial ties below zero
     p=np.sign(p)*abs(p)**(1/power)
     preds.append(p)
    ens=np.mean(preds[:6],axis=0)
    rr.append([spearmanr(y[te],p).statistic for p in preds]+[spearmanr(y[te],ens).statistic])
   results[str((size,reg,sw))]=np.mean(rr,axis=0).tolist()
   print(size,reg,sw,np.round(np.mean(rr,axis=0),4),flush=True)
json.dump(results,open('ensemble_check_results.json','w'),indent=2)
