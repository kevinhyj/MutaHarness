exec(open('explore.py').read().split('results=[];')[0])
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import time
out=[]; pd={}
def run(name,fun):
 start=time.time(); scores=[]; pred=np.zeros((6,len(idx)))
 for j,(tr,te) in enumerate(folds):
  p=fun(tr,te);scores.append(spearmanr(y[te],p).statistic);pred[j//4,np.searchsorted(idx,te)]=p
 out.append((np.mean(scores),name,np.std(scores)));pd[name]=pred
 print(out[-1],'seconds',round(time.time()-start,2),flush=True)

for main_scale in [1,3,10]:
 for threshold in [100,1000,10000]:
  def f(tr,te):
   xt=x-x[tr].mean(0);pairs=np.array([xt[:,a]*xt[:,b] for a in range(20) for b in range(a+1,20)]).T
   z=np.c_[main_scale*xt,pairs];model=ARDRegression(max_iter=500,threshold_lambda=threshold,tol=.001)
   return model.fit(z[tr],y[tr]).predict(z[te])
  run('ard_%s_%s'%(main_scale,threshold),f)
for main_scale in [3,10,30]:
 for al in [.0003,.001,.003,.01]:
  for l1 in [.8,1]:
   def f(tr,te):
    xt=x-x[tr].mean(0);pairs=np.array([xt[:,a]*xt[:,b] for a in range(20) for b in range(a+1,20)]).T
    z=np.c_[main_scale*xt,pairs];model=ElasticNet(alpha=al,l1_ratio=l1,max_iter=15000,tol=1e-5)
    return model.fit(z[tr],y[tr]).predict(z[te])
   run('elastic_%s_%s_%s'%(main_scale,al,l1),f)
for length in [2,4,8,16,32]:
 for variance in [.01,.1,1,10]:
  for reg in [.01,.1]:
   def f(tr,te):
    xt=x-x[tr].mean(0);k=xt@xt.T; dist=((x[:,None]-x[None,:])**2).sum(2)
    k+=variance*np.exp(-dist/length)
    return y[tr].mean()+k[np.ix_(te,tr)]@np.linalg.solve(k[np.ix_(tr,tr)]+reg*np.eye(len(tr)),y[tr]-y[tr].mean())
   run('rbf_%s_%s_%s'%(length,variance,reg),f)
np.savez('exploration2_predictions.npz',y=y[idx],**pd)
json.dump(sorted(out,reverse=True),open('exploration2_results.json','w'),indent=2)
