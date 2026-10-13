exec(open('explore.py').read().split('results=[]')[0])
# Generalized least-squares intercept estimated without assuming WT fitness.
def evaluate3(K,alpha,mode,offset=.1,intercept='gls',return_pred=False):
 scores=[];oof=np.full((len(splits),len(s)),np.nan)
 for z,(tr,te) in enumerate(splits):
  yy=y[tr].copy()
  if mode=='log':yy=np.log(np.maximum(yy,0)+offset)
  elif mode=='rank':yy=(rankdata(yy)-.5)/len(yy)
  elif mode=='sqrt':yy=np.sqrt(np.maximum(yy,0))
  elif mode=='asinh':yy=np.arcsinh(yy/.2)
  cf=cho_factor(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),check_finite=False)
  m=yy.mean()
  if intercept=='gls':
   iz=cho_solve(cf,np.column_stack([yy,np.ones(len(tr))]),check_finite=False)
   m=iz[:,0].sum()/iz[:,1].sum()
  wt=cho_solve(cf,yy-m,check_finite=False);pred=m+K[np.ix_(te,tr)]@wt
  scores.append(spearmanr(y[te],pred).statistic);oof[z,te]=pred
 scores=np.array(scores)
 if return_pred:return scores,oof
 return round(scores[:12].mean(),5),round(scores[12:].mean(),5),round(scores.std(),5)
res=[]
for chem,exact,smooth,alpha,mode,offset,intercept in itertools.product([1.5,5,15],[.5,2],[.2,.7],[.2,1,4],['log','sqrt','raw'],[.03,.1,.3],['mean','gls']):
 if mode!='log' and offset!=.1:continue
 K=kernels['pos']+chem*kernels['chem']+exact*kernels['exact']+smooth*kernels['smooth2'];sc=evaluate3(K,alpha,mode,offset,intercept)
 cfg=dict(chem=chem,exact=exact,glob=0,smooth=smooth,alpha=alpha,mode=mode,offset=offset,intercept=intercept)
 res.append((.7*sc[0]+.3*sc[1],sc,cfg))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('explore3_results.json','w'),indent=2)
for r in res[:25]:print(r,flush=True)
