from explore import *
start=time.time(); results=[]; preds={}
G=['DE','KR','FWY','FWYLIVM','P','G','STNQ']
for bins in [1,2,3,4,6,11]:
 ed=np.linspace(0,44,bins+1,dtype=int)
 features['small'+str(bins)]=np.stack([P[g][:,lo:hi].sum(1) for lo,hi in zip(ed[:-1],ed[1:]) for g in G],1)
features['smooth']=np.stack([P[g]@np.cos(np.arange(44)*np.pi*k/44) for g in G for k in range(5)],1)
features['domain']=np.stack([P[g][:,lo:hi].sum(1) for lo,hi in [(0,19),(19,24),(24,44)] for g in G],1)
features['posgroup']=np.stack([P[g] for g in G],2).reshape(N,-1)

def strat_ridge(X,alpha,kind,sw,method,kernel,gamma):
 pred=np.zeros((3,N))
 for j,(tr,te) in enumerate(folds):
  for group in [0,1]:
   test=te[(order[te]>15)==group]
   if not len(test):continue
   train=tr if method=='joint' else tr[(order[tr]>15)==group]
   x=X[train]; q=X[test]; mu=x.mean(0); sd=np.maximum(x.std(0),.5); x=(x-mu)/sd; q=(q-mu)/sd
   if kernel=='linear': k=x@x.T; kt=q@x.T
   else:
    norm=(x*x).sum(1); qnorm=(q*q).sum(1)
    k=np.exp(-gamma*np.maximum(0,norm[:,None]+norm[None,:]-2*x@x.T)/x.shape[1]); kt=np.exp(-gamma*np.maximum(0,qnorm[:,None]+norm[None,:]-2*q@x.T)/x.shape[1])
   y=target(Y[train],kind); weights=np.where(order[train]==1,sw,1.)
   ym=np.average(y,weights=weights); a=solve(k+np.diag(alpha/weights),y-ym,assume_a='pos')
   pred[j//4,test]=kt@a+ym
 return pred
for combo in [['small1'],['small3'],['small1','small3'],['small1','domain'],['smooth'],['small1','smooth'],['small1','posgroup'],['small1','small3','pairs']]:
 X=np.concatenate([features[a] for a in combo],1)
 for method in ['joint','separate']:
  for kind in ['rank','robust','raw']:
   for sw in [.3,1.]:
    for kernel,alpha,gamma in [('linear',a,1) for a in [1,10,100]]+[('rbf',a,g) for a in [.1,1] for g in [.3,1]]:
     name=('+'.join(combo),kind,alpha,sw,method+'_'+kernel,gamma)
     p=strat_ridge(X,alpha,kind,sw,method,kernel,gamma); v=val(p); results.append((v,name)); preds[str(name)]=p
results.sort(reverse=True)
print('ALL')
for v,name in results[:20]:print(tuple(round(z,4) for z in v),name)
print('HIGH')
for v,name in sorted(results,key=lambda a:a[0][3],reverse=True)[:20]:print(tuple(round(z,4) for z in v),name)
np.savez('validation_predictions3.npz',**preds)
json.dump([{'metrics':list(v),'config':list(name)} for v,name in results],open('validation_results3.json','w'),indent=2)
print('seconds',time.time()-start,flush=True)
