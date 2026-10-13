exec(open('explore3.py').read().split('configs=[]')[0])
from sklearn.linear_model import LogisticRegression
results=[];started=time.time()
for f in ['context','pair']:
 for penalty in [.1,1,10]:
  preds=np.full((3,len(s)),np.nan)
  for split,(tr,te) in enumerate(splits):
   xt=features[f][tr];xq=features[f][te];p=np.zeros(len(te))
   thresholds=np.quantile(Y[tr],np.arange(.1,1.,.1))
   for cut in thresholds:
    yy=(Y[tr]>cut).astype(int)
    if len(np.unique(yy))<2:p+=yy.mean()
    else:p+=LogisticRegression(C=penalty,solver='lbfgs',max_iter=250).fit(xt,yy).predict_proba(xq)[:,1]
   preds[split//4,te]=p
  corrs=[spearmanr(Y[mid],p[mid]).statistic for p in preds];results.append((np.mean(corrs),np.std(corrs),(f,penalty),preds)); print(results[-1][:3],round(time.time()-started,2),flush=True)
np.savez('ordinal_predictions.npz',predictions=np.array([r[3] for r in results]))
json.dump([{'rho':float(r[0]),'sd':float(r[1]),'config':r[2]} for r in results],open('ordinal_results.json','w'),indent=2)
