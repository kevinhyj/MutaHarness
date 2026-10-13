exec(open('explore3.py').read().split('pd={};')[0])
from sklearn.linear_model import ElasticNet
from scipy.linalg import solve
out=[];pd={}
def run(name,fun):
 scores=[]; pred=np.zeros((6,len(idx)))
 for j,(tr,te) in enumerate(folds):
  p=fun(tr,te);scores.append(spearmanr(y[te],p).statistic);pred[j//4,np.searchsorted(idx,te)]=p
 out.append((np.mean(scores),name,np.std(scores)));pd[name]=pred
for power in [1.25,1.5,2,3,4]:
 for strength in [0,.001,.003,.01,.03]:
  for reg in [.01,.1]:
   def f(tr,te):
    a=x-x[tr].mean(0);k=a@a.T;k+=strength*k*k
    yt=np.maximum(y,1e-6)**power
    p=yt[tr].mean()+k[np.ix_(te,tr)]@np.linalg.solve(k[np.ix_(tr,tr)]+reg*np.eye(len(tr)),yt[tr]-yt[tr].mean())
    return np.maximum(p,0)**(1/power)
   run(f'power_{power}_{strength}_{reg}',f)
for kind in ['site','mixed','allele']:
 for lam in [.001,.003,.01,.03]:
  for scale in [3,10]:
   def f(tr,te):
    a=x-x[tr].mean(0);b=xp-xp[tr].mean(0)
    if kind=='site':pairs=np.array([b[:,i]*b[:,j] for i in range(b.shape[1]) for j in range(i+1,b.shape[1])]).T
    elif kind=='allele':pairs=np.array([a[:,i]*a[:,j] for i in range(a.shape[1]) for j in range(i+1,a.shape[1])]).T
    else:pairs=np.array([a[:,i]*b[:,j] for i in range(a.shape[1]) for j in range(b.shape[1]) if int(alpha[i][1:-1])!=positions[j]]).T
    z=np.c_[scale*a,pairs]
    return ElasticNet(alpha=lam,l1_ratio=1,max_iter=15000,tol=1e-5).fit(z[tr],y[tr]).predict(z[te])
   run(f'lasso_{kind}_{lam}_{scale}',f)
for row in sorted(out,reverse=True)[:25]:print(row,flush=True)
np.savez('exploration4_predictions.npz',y=y[idx],**pd)
json.dump(sorted(out,reverse=True),open('exploration4_results.json','w'),indent=2)
