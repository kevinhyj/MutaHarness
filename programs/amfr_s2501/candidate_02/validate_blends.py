from exploration_utils import *
D=json.load(open('validation_design.json'));splits=[(k,np.array(tr),np.array(te)) for k,tr,te in D['splits']]
for one,phys in [(0,.5),(.1,.5),(0,.2),(.2,.5)]:
 k=one*kernels['one']+phys*kernels['phys'];dist=np.maximum(np.diag(k)[:,None]+np.diag(k)[None,:]-2*k,0)
 kernels[f'smooth{one},{phys}']=np.exp(-dist)
cs=[]
for aa,alpha,tgt in [('smooth0.1,0.5',.3,'rank'),('smooth0,0.5',.3,'rank'),('rbf2',1,'rank'),('diff2',1,'rank'),('smooth0.1,0.5',.3,'floor4'),('smooth0.1,0.5',.3,'raw')]:
 cs.append(dict(aa=aa,pos=.5,phys=.25,transfer=0,global_=.25,inter=.1 if aa in ['rbf2','diff2'] else 0,alpha=alpha,target=tgt))
prs=[];tokens=[set(r['mutation'].split(':')) for r in records]
for q in cs:
 K=matrix(q);ps=[];out=[]
 for kind,tr,te in splits:
  pred=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+q['alpha']*np.eye(len(tr)),target(q['target'],tr),assume_a='pos')
  ps.append((rankdata(pred)-.5)/len(pred));out.append((kind,spearmanr(pred,y[te]).statistic))
 print(q,{kind:float(np.mean([v for k,v in out if k==kind])) for kind in ['all','feedback','novel']},flush=True);prs.append(ps)
res=[]
for aaw,rawfloorw,adapt in itertools.product([.2,.4,.6],[0,.2,.4],[False,True]):
 out=[]
 for j,(kind,tr,te) in enumerate(splits):
  seen=set().union(*(tokens[ix] for ix in tr));nov=np.array([bool(tokens[ix]-seen) for ix in te])
  w=np.full(len(te),aaw)
  if adapt:w=np.where(nov,np.minimum(aaw+.3,1),max(aaw-.2,0))
  seq=(prs[0][j]+prs[1][j])/2;aa=(prs[2][j]+prs[3][j])/2
  p=(1-w)*seq+w*aa;p=(1-rawfloorw)*p+rawfloorw*prs[4][j]
  out.append((kind,spearmanr(p,y[te]).statistic))
 vals={kind:float(np.mean([v for k,v in out if k==kind])) for kind in ['all','feedback','novel']}
 score=.25*vals['all']+.4*vals['feedback']+.35*vals['novel'];res.append((score,aaw,rawfloorw,adapt,vals))
res.sort(reverse=True,key=lambda x:x[0])
for r in res:print('blend',r,flush=True)
