from exploration_utils import *
# Recreate smooth kernels independently of prior exploration output.
for v in [.1,.3,1]:
 k=kernels['phys']*.5+kernels['one']*.2+kernels['global']*v
 dist=np.diag(k)[:,None]+np.diag(k)[None,:]-2*k
 for gamma in [.1,.3,1]:kernels[f'seq{v}-{gamma}']=np.exp(-dist*gamma)

def cfg(aa='seq0.1-1',alpha=.3,transfer=0,target='rank',inter=0):
 return dict(aa=aa,pos=.5,phys=.25,transfer=transfer,global_=.25,inter=inter,alpha=alpha,target=target)
cs=[cfg(),cfg(alpha=.1),cfg(aa='seq0.3-1'),cfg(aa='seq0.1-0.3'),cfg(aa='rbf2',alpha=1,inter=.1),cfg(aa='diff2',alpha=1,inter=.1),cfg(transfer=.5),cfg(transfer=1),cfg(aa='rbf2',alpha=1,transfer=1,inter=.1),cfg(target='raw'),cfg(target='floor4')]
# Strict mutation-token holdouts: all variants containing withheld substitutions are removed from training.
rng=np.random.RandomState(817)
tokens=[set(r['mutation'].split(':')) for r in records]
positions=[28,32,38,41]
novel=[]
for _ in range(30):
 held=set()
 for j in positions:
  options=sorted({t for ts in tokens for t in ts if int(t[1:-1])==j})
  held.add(options[rng.randint(len(options))])
 te=np.array([i for i,ts in enumerate(tokens) if ts&held]);tr=np.array([i for i,ts in enumerate(tokens) if not ts&held]);novel.append(('novel',tr,te))
# New seeds for ordinary holdout checks, separate from the exploration splits.
ordinary=[]
for seed in [1019,237,2027]:
 for tr,te in KFold(5,shuffle=True,random_state=seed).split(y):ordinary.append(('all',tr,te))
 feed=np.arange(32,n)
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feed):ordinary.append(('feedback',np.r_[np.arange(32),feed[tr]],feed[te]))

predictions=[]
for q in cs:
 K=matrix(q);outs=[];prs=[]
 for kind,tr,te in ordinary+novel:
  p=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+q['alpha']*np.eye(len(tr)),target(q['target'],tr),assume_a='pos')
  prs.append((rankdata(p)-.5)/len(p))
  outs.append((kind,spearmanr(p,y[te]).statistic))
 result={kind:float(np.mean([v for k,v in outs if k==kind])) for kind in ['all','feedback','novel']}
 print(q,result,flush=True);predictions.append(prs)
for choices,weights in [([0,4],[.7,.3]),([0,4,5],[.6,.2,.2]),([0,4,6],[.5,.25,.25]),([0,4,6,9],[.5,.2,.2,.1]),([0,1,2,3,4,5],[.3,.2,.2,.1,.1,.1])]:
 outs=[]
 for k,(kind,tr,te) in enumerate(ordinary+novel):
  p=sum(w*predictions[ix][k] for ix,w in zip(choices,weights));outs.append((kind,spearmanr(p,y[te]).statistic))
 print('ENSEMBLE',choices,weights,{kind:float(np.mean([v for k,v in outs if k==kind])) for kind in ['all','feedback','novel']},flush=True)
json.dump({'candidates':cs,'splits':[(k,tr.tolist(),te.tolist()) for k,tr,te in ordinary+novel]},open('validation_design.json','w'),indent=2)
