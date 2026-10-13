from exploration_utils import *
# Freeze the candidate family and blended weights before these additional splits.
for one in [0,.1]:
 k=one*kernels['one']+.5*kernels['phys'];dist=np.maximum(np.diag(k)[:,None]+np.diag(k)[None,:]-2*k,0)
 kernels[f'smooth{one}']=np.exp(-dist)
configs=[]
for aa,alpha,tgt in [('smooth0',.3,'rank'),('smooth0.1',.3,'rank'),('rbf2',1,'rank'),('diff2',1,'rank'),('smooth0.1',.3,'adaptfloor')]:
 configs.append(dict(aa=aa,pos=.5,phys=.25,transfer=0,global_=.25,inter=.1 if aa in ['rbf2','diff2'] else 0,alpha=alpha,target=tgt))
new=[]
for seed in [3971,6833,10007,17041]:
 for tr,te in KFold(5,shuffle=True,random_state=seed).split(y):new.append(('all',tr,te))
 feed=np.arange(32,n)
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feed):new.append(('feedback',np.r_[np.arange(32),feed[tr]],feed[te]))
tokens=[set(r['mutation'].split(':')) for r in records]
rng=np.random.RandomState(42197)
for _ in range(50):
 held=set()
 for j in [28,32,38,41]:
  opts=sorted({t for ts in tokens for t in ts if int(t[1:-1])==j});held.add(opts[rng.randint(len(opts))])
 te=np.array([i for i,ts in enumerate(tokens) if ts&held]);tr=np.array([i for i,ts in enumerate(tokens) if not ts&held]);new.append(('novel',tr,te))
for floorname in ['fixed','adaptive']:
 out=[];within=[]
 for kind,tr,te in new:
  ps=[]
  for q in configs:
   K=matrix(q)
   if q['target']=='adaptfloor':
    floor=-4 if floorname=='fixed' else y[tr].min()-.8
    z=np.log(np.expm1(np.clip(y[tr]-floor,.001,50)))
   else:z=target(q['target'],tr)
   p=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+q['alpha']*np.eye(len(tr)),z,assume_a='pos')
   ps.append((rankdata(p)-.5)/len(p))
  seen=set().union(*(tokens[ix] for ix in tr));nov=np.array([bool(tokens[ix]-seen) for ix in te])
  w=np.where(nov,.5,0)
  p=.6*((1-w)*(ps[0]+ps[1])/2+w*(ps[2]+ps[3])/2)+.4*ps[4]
  out.append((kind,spearmanr(p,y[te]).statistic))
  dd=counts[te]==2
  if dd.sum()>=5:within.append((kind,spearmanr(p[dd],y[te][dd]).statistic))
 print(floorname,{kind:float(np.mean([v for k,v in out if k==kind])) for kind in ['all','feedback','novel']},flush=True)
 print('within double', {kind:float(np.mean([v for k,v in within if k==kind])) for kind in ['all','feedback','novel']},flush=True)
