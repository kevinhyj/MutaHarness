exec(open('explore.py').read().split('configs=[]')[0])
from sklearn.linear_model import ElasticNet
allrecords=r+c['query'];xa,vocab=_features(allrecords,ref);x=xa[:len(r)]
# Conventional residue properties, scaled approximately to unit ranges.
aa='ACDEFGHIKLMNPQRSTVWY'
hydro=[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
vol=[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
prop={a:np.array([hydro[j]/4.5,(vol[j]-140)/90, float(a in 'KR')-float(a in 'DE'),float(a in 'FYWH'),float(a=='P'),float(a=='G'),float(a in 'STNQ')])for j,a in enumerate(aa)}
chem=np.zeros((len(r),len(positions)*7));agg=np.zeros((len(r),8*7))
for i,ms in enumerate(muts):
 for p,a in ms:
  if p in positions:
   delta=prop[a]-prop[ref[p-1]];chem[i,positions.index(p)*7:(positions.index(p)+1)*7]=delta;agg[i,block(p)*7:(block(p)+1)*7]+=delta
bases={'full':np.column_stack([x,px*.5,bm,bc*.25]),'small':np.column_stack([x,bm]),'gated':np.column_stack([x,x*mult[:,None],bm,bc*.25]),'chem':np.column_stack([x,px*.5,bm,bc*.25,chem*.4,agg*.15])}
configs=[]
for name in bases:
 for a in [.001,.003,.006,.01,.02]:
  for ratio in [.1,.5,.9]:
   for target in ['normal','uniform']:
    configs.append((name,a,ratio,target))
allpred={str(cfg):np.zeros((5,len(mi)))for cfg in configs}
# Retain promising nonlinear models and independently recheck.
zz=x@x.T;dist=px.sum(1)[:,None]+px.sum(1)[None,:]-px@px.T-zz
kernels={}
for gamma in [.2,.4,.8,1.6]:
 for target in ['normal','blend','uniform']:
  for a in [.1,1]:
   kernels[str(('rbf',gamma,a,target))]=(np.exp(-gamma*dist),a,target)
for gate in [0,3]:
 for deg in [1,2]:
  k=zz+gate*((x*mult[:,None])@(x*mult[:,None]).T)+2*bm@bm.T+1
  if deg==2:k=k+.25*(zz*zz-zz)/2
  kernels[str(('poly',gate,deg,.3,'normal'))]=(k,.3,'normal')
for k in kernels:allpred[k]=np.zeros((5,len(mi)))
start=time.time()
for rep in range(5):
 for tr,te in KFold(4,shuffle=True,random_state=711+rep).split(mi):
  train=np.r_[si,mi[tr]];test=mi[te];yt=y[train];u=(rankdata(yt)-.5)/len(yt)
  targets={'normal':norm.ppf(u),'uniform':u,'blend':norm.ppf(u)*.5+(yt-yt.mean())/yt.std()*.5}
  for cfg in configs:
   name,a,ratio,target=cfg
   model=ElasticNet(alpha=a,l1_ratio=ratio,max_iter=3000,tol=1e-5)
   model.fit(bases[name][train],targets[target]);allpred[str(cfg)][rep,te]=model.predict(bases[name][test])
  for name,(k,a,target) in kernels.items():
   t=targets[target];allpred[name][rep,te]=t.mean()+k[np.ix_(test,train)]@solve(k[np.ix_(train,train)]+a*np.eye(len(train)),t-t.mean(),assume_a='pos')
 print('rep',rep,'seconds',time.time()-start,flush=True)
res=[]
for k,p in allpred.items():
 sc=[spearmanr(y[mi],pp).statistic for pp in p];res.append((np.mean(sc),np.std(sc),k,sc))
for row in sorted(res,reverse=True)[:80]:print(row)
np.savez('explore3_predictions.npz',y=y[mi],**allpred)
