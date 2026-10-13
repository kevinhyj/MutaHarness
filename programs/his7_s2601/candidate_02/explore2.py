exec(open('explore.py').read().split('configs=[]')[0])
from sklearn.linear_model import ElasticNet
allrecords=r+c['query'];xa,vocab=_features(allrecords,ref);x=xa[:len(r)]
px=np.array([[float(any(p==pp for p,a in ms)) for pp in positions]for ms in muts])
z=x@x.T;zg=(x*mult[:,None])@(x*mult[:,None]).T
# Categorical Hamming distance on substituted positions (two different mutants at one site count once).
dist=px.sum(1)[:,None]+px.sum(1)[None,:]-px@px.T-z
configs=[];kernels={}
for gate in [0,1,3]:
 for degree in [1,2]:
  for a in [.1,.3,1,3]:
   name=('poly',gate,degree,a)
   k=z+gate*zg+2*bm@bm.T+1
   if degree==2:k=k+.25*(z*z-z)/2
   kernels[str(name)]=(k,a)
for gamma in [.05,.1,.2,.4,.8]:
 for a in [.01,.1,.3,1]:
  for local in [False,True]:
   name=('rbf',gamma,a,local)
   k=np.exp(-gamma*dist)
   if local:k=k*(bm@bm.T)+(z*.1)+1
   kernels[str(name)]=(k,a)
# Pool two kernels; additive generalizes at larger distances; RBF models combinations.
for gamma in [.1,.2,.4]:
 for w in [1,3,10]:
  for a in [.1,.3,1]:
   name=('mixed',gamma,w,a)
   k=z+2*bm@bm.T+1+w*np.exp(-gamma*dist)*(bm@bm.T)
   kernels[str(name)]=(k,a)
targets=['normal','uniform','raw','logit','blend']
allpred={k+' '+t:np.zeros((4,len(mi))) for k in kernels for t in targets}
for alpha in [.003,.01,.03,.1]:
 for ratio in [.1,.5,.9]:
  for t in ['normal','uniform']:
   allpred[str(('elastic',alpha,ratio,t))]=np.zeros((4,len(mi)))
feat=np.column_stack([x,px*.5,bm,bc*.25])
start=time.time()
for rep in range(4):
 for fold,(tr,te) in enumerate(KFold(4,shuffle=True,random_state=101+rep).split(mi)):
  train=np.r_[si,mi[tr]];test=mi[te];yt=y[train]
  u=(rankdata(yt)-.5)/len(yt)
  target={'normal':norm.ppf(u),'uniform':u,'raw':yt,'logit':np.log(np.clip(yt/1.2,.02,.98)/(1-np.clip(yt/1.2,.02,.98))),'blend':norm.ppf(u)*.5+(yt-yt.mean())/yt.std()*.5}
  for name,(k,a) in kernels.items():
   T=np.column_stack(list(target.values()));means=T.mean(0)
   pred=means+k[np.ix_(test,train)]@solve(k[np.ix_(train,train)]+a*np.eye(len(train)),T-means,assume_a='pos')
   for j,t in enumerate(targets):allpred[name+' '+t][rep,te]=pred[:,j]
  for alpha in [.003,.01,.03,.1]:
   for ratio in [.1,.5,.9]:
    for t in ['normal','uniform']:
     model=ElasticNet(alpha=alpha,l1_ratio=ratio,max_iter=3000)
     model.fit(feat[train],target[t]);allpred[str(('elastic',alpha,ratio,t))][rep,te]=model.predict(feat[test])
 print('rep',rep,'seconds',time.time()-start,flush=True)
res=[]
for k,p in allpred.items():
 sc=[spearmanr(y[mi],pp).statistic for pp in p]
 res.append((np.mean(sc),np.std(sc),k,sc))
for row in sorted(res,reverse=True)[:70]:print(row)
np.savez('explore2_predictions.npz',y=y[mi],**allpred)
