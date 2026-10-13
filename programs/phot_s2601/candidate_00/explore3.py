exec(open('explore.py').read().split('results=[];')[0])
import time
positions=sorted(set(int(m[1:-1]) for m in alpha))
xp=np.array([[sum(m in s for m in alpha if int(m[1:-1])==pos)>0 for pos in positions] for s in sets],float)
pd={};out=[]
def run(name,fun):
 scores=[]; pred=np.zeros((6,len(idx)))
 for j,(tr,te) in enumerate(folds):
  p=fun(tr,te); scores.append(spearmanr(y[te],p).statistic);pred[j//4,np.searchsorted(idx,te)]=p
 out.append((np.mean(scores),name,np.std(scores)));pd[name]=pred
for centering in [True,False]:
 for kind in ['site','both','mixed','allele']:
  for strength in [.001,.003,.01,.03,.1,.3,1]:
   for reg in [.01,.03,.1]:
    def f(tr,te):
     a=x-x[tr].mean(0) if centering else x
     b=xp-xp[tr].mean(0) if centering else xp
     ka=a@a.T;kb=b@b.T
     pairs={'site':kb*kb,'both':ka*ka+kb*kb,'mixed':ka*kb,'allele':ka*ka}[kind]
     k=ka+strength*pairs
     # ensure fitted intercept also if uncentered
     km=k[:,tr].mean(1);k=k-km[:,None]-km[None,:]+k[np.ix_(tr,tr)].mean()
     return y[tr].mean()+k[np.ix_(te,tr)]@np.linalg.solve(k[np.ix_(tr,tr)]+reg*np.eye(len(tr)),y[tr]-y[tr].mean())
    name=f'{centering}_{kind}_{strength}_{reg}';run(name,f)
for row in sorted(out,reverse=True)[:30]:print(row,flush=True)
np.savez('exploration3_predictions.npz',y=y[idx],**pd)
json.dump(sorted(out,reverse=True),open('exploration3_results.json','w'),indent=2)
