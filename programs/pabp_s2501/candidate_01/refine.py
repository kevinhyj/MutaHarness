exec(open('explore.py').read().split('rng=np.random')[0])
import advanced
results=json.load(open('advanced_cv_results.json'));ek,_=advanced.extra_kernels(s,ref)
count=np.array([len(t) for t in ms]);kk=np.outer(count,count)
def cv2(k,target,alpha,center):
 out=np.zeros((5,n))
 for f,(tr,va) in enumerate(splits):
  kt=k[np.ix_(tr,tr)].copy();kv=k[np.ix_(va,tr)].copy()
  if center:
   m=kt.mean(axis=0);tot=kt.mean();kt=kt-m[:,None]-m[None,:]+tot;kv=kv-m[None,:]-kv.mean(axis=1)[:,None]+tot
  mean=target[tr].mean();coef=solve(kt+alpha*np.eye(len(tr)),target[tr]-mean,assume_a='pos');out[f//4 if f<16 else 4,va]=kv@coef+mean
 scores=[spearmanr(out[j,feedback],y[feedback]).statistic for j in range(4)];sc,sd=np.mean(scores),np.std(scores);allsc=spearmanr(out[4],y).statistic
 return .8*sc+.2*allsc-.1*sd,sc,sd,allsc,out
rr=[]
for idx,r in enumerate(results[:30]):
 k=np.einsum('k,kij->ij',r[6],kernels)+np.einsum('k,kij->ij',r[7],ek)
 if r[8]:k+=r[8]*k*k/np.mean(np.diag(k))
 for center in [False,True]:
  for ct in [0,.1,1,10,100]:
   for tf in ['log01','log05','log15','raw','rank']:
    target={'raw':y,'log05':np.log(y+.05),'log15':np.log(y+.15),'log01':np.log(y+.01),'rank':(rankdata(y)-.5)/n}[tf]
    merit,sc,sd,allsc,out=cv2(k+ct*kk,target,r[5],center)
    rr.append([merit,sc,sd,allsc,idx,center,ct,tf,out.tolist()])
rr.sort(reverse=True,key=lambda v:v[0]);json.dump(rr,open('refine_cv_results.json','w'))
for r in rr[:15]:print(r[:8])
