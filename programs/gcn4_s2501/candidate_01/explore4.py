from explore import *
start=time.time(); G=['DE','KR','FWY','FWYLIVM','P','G','STNQ']; results=[];preds={}

def run(modes,comp,alpha,gamma,method,globalrank,blend=0):
 X=np.stack([P[g]@np.cos(np.arange(44)*np.pi*k/44) for g in G for k in range(modes)],1)
 if comp:X=np.concatenate([X,np.stack([P[g].sum(1) for g in G],1)],1)
 out=np.zeros((3,N))
 for j,(tr,te) in enumerate(folds):
  yt=target(Y[tr],'rank')
  for group in [False,True]:
   test=te[(order[te]>15)==group]
   if not len(test):continue
   sel=np.ones(len(tr),bool) if method=='joint' else ((order[tr]>15)==group)
   train=tr[sel]; x=X[train]; q=X[test]; mu=x.mean(0); sd=np.maximum(x.std(0),.5); x=(x-mu)/sd; q=(q-mu)/sd
   if gamma==0: k=x@x.T; kt=q@x.T
   else:
    norm=(x*x).sum(1); qnorm=(q*q).sum(1)
    k=np.exp(-gamma*np.maximum(0,norm[:,None]+norm[None,:]-2*x@x.T)/x.shape[1]); kt=np.exp(-gamma*np.maximum(0,qnorm[:,None]+norm[None,:]-2*q@x.T)/x.shape[1])
   y=yt[sel] if globalrank else target(Y[train],'rank'); ym=y.mean()
   a=solve(k+alpha*np.eye(len(train)),y-ym,assume_a='pos'); out[j//4,test]=kt@a+ym
 return out
for modes in [3,4,5,6,8]:
 for comp in [False,True]:
  for method in ['joint','separate']:
   for alpha,gamma in [(a,0) for a in [30,100,300]]+[(a,g) for a in [.3,1,3] for g in [.3,1]]:
    name=(modes,comp,alpha,gamma,method,True)
    p=run(*name); v=val(p);results.append((v,name));preds[str(name)]=p
results.sort(reverse=True)
for v,name in results[:20]:print(tuple(round(z,4) for z in v),name)
# Compare ensembles of robust defaults, on existing validation folds.
joint=np.mean([preds[str((5,c,1,g,'joint',True))] for c in [False,True] for g in [.3,1]],0)
local=np.mean([preds[str((5,False,a,g,'separate',True))] for a,g in [(100,0),(300,0),(1,.3)]],0)
for loww in [0,.25,.5,1]:
 for highw in [0,.25,.5,.75,1]:
  w=np.where(order>15,highw,loww); p=joint*(1-w)+local*w
  print('BLEND',loww,highw,tuple(round(z,4) for z in val(p)))
np.savez('validation_predictions4.npz',**preds)
json.dump([{'metrics':list(v),'config':list(name)} for v,name in results],open('validation_results4.json','w'),indent=2)
print('seconds',time.time()-start,flush=True)
