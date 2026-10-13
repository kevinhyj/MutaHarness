exec(open('explore.py').read().split('results=[]; predictions={}')[0])
# redefine fold storage for six independent repeats
def evaluate(name,predict):
 vals=[];sm=[];oof=np.zeros((6,n))
 for i,(tr,te) in enumerate(splits):
  p=predict(tr,te);vals.append(spearmanr(y[te],p).statistic);oof[i//4,te]=p
 for tr,te in small:sm.append(spearmanr(y[te],predict(tr,te)).statistic)
 result={'name':name,'cv':float(np.mean(vals)),'small':float(np.mean(sm)),'std':float(np.std(vals))}
 results.append(result);predictions[name]=oof
 return result
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import rankdata
splits=[]
for seed in [103,127,151,173,197,223]:splits.extend(list(KFold(4,shuffle=True,random_state=seed).split(x)))
small=list(ShuffleSplit(12,train_size=32,random_state=227).split(x))
results=[];predictions={};t=time.time()
# fresh splits to measure regularized small models and alternative targets.
for loci in [('46',),('46','59'),('46','60'),('46','6'),('46','4'),('59',),()]:
 mask=[(v[i][1:-1] in loci or v[j][1:-1] in loci) for i,j in pairs]
 pp=xp[:,mask]
 for ratio in [.1,.3,1,3]:
  k=x@x.T+ratio*(pp@pp.T)
  for power in [.25,.5,.75,1,1.5]:
   for alpha in [.1,.3,1,3]:
    evaluate(f'focus_{"+".join(loci)}_{ratio}_{power}_{alpha}',lambda tr,te,k=k,power=power,alpha=alpha:kernelpred(tr,te,k,power,alpha))
# sparse Bayesian feature selection and sparse ridge+lasso
for ratio in [.1,.3,1]:
 feat=np.column_stack([x,xp*np.sqrt(ratio)])
 for power in [.5,1]:
  for alpha in [.001,.003,.01,.03]:
   for l1 in [.5,1]:
    def pred(tr,te):
     fit=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=3000).fit(feat[tr],transform(y[tr],power))
     return fit.predict(feat[te])
    evaluate(f'elastic_{ratio}_{power}_{alpha}_{l1}',pred)
  for threshold in [1e3,1e4]:
   def pred(tr,te):
    return ARDRegression(threshold_lambda=threshold,max_iter=100).fit(feat[tr],transform(y[tr],power)).predict(feat[te])
   evaluate(f'ard_{ratio}_{power}_{threshold}',pred)
# pairwise logistic rank regression, focusing on supported interaction family
for family in ['46','all','none']:
 pp=xp if family=='all' else xp[:,[v[i][1:-1]=='46' or v[j][1:-1]=='46' for i,j in pairs]] if family=='46' else np.zeros((n,0))
 for ratio in [.1,1]:
  feat=np.column_stack([x,pp*np.sqrt(ratio)])
  for alpha in [.0001,.001,.01,.1]:
   for weighted in [False,True]:
    def pred(tr,te):
     a,b=np.triu_indices(len(tr),1); dif=feat[tr[a]]-feat[tr[b]]; sign=np.sign(y[tr[a]]-y[tr[b]]); dif=dif*sign[:,None]
     w=np.abs(y[tr[a]]-y[tr[b]]) if weighted else np.ones(len(a));w=w/w.sum()
     def fun(beta):
      h=dif@beta; loss=np.logaddexp(0,-h); sig=1/(1+np.exp(np.clip(h,-500,500)))
      return w@loss+alpha*(beta@beta)/2, -dif.T@(w*sig)+alpha*beta
     beta=minimize(fun,np.zeros(feat.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':150}).x
     return feat[te]@beta
    evaluate(f'rank_{family}_{ratio}_{alpha}_{weighted}',pred)
# trees provide a different nonlinear hypothesis
for power in [.5,1]:
 for depth in [2,3,4]:
  for leaf in [2,5,10]:
   def pred(tr,te):
    fit=GradientBoostingRegressor(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,learning_rate=.04,random_state=7).fit(x[tr],transform(y[tr],power));return fit.predict(x[te])
   evaluate(f'boost_{power}_{depth}_{leaf}',pred)
np.savez('exploration2_oof.npz',**predictions)
json.dump(results,open('exploration2_results.json','w'),indent=2)
print('Time',time.time()-t,'models',len(results))
for r in sorted(results,key=lambda r:-(.7*r['cv']+.3*r['small']))[:35]:print(r)
