# Reuses validation folds and base kernels from explore.py without rerunning search.
exec(open('explore.py').read().split('rng=np.random')[0])
from proposal import PROPS,AA
from scipy.spatial.distance import cdist
# Novel chemistry kernels: smooth amino-acid similarity at each position,
# and sparse interaction covariance. WT subtraction anchors no change to zero.
seq=[a['sequence'] for a in records]
codes=np.array([[AA.index(a) for a in s] for s in seq]);wt=np.array([AA.index(a) for a in ref])
new={}
for bw in [1.,3.,10.]:
 sim=np.exp(-cdist(PROPS,PROPS,'sqeuclidean')/(2*bw))
 gram=np.zeros((n,n))
 for j in range(len(ref)):
  a=codes[:,j];w=wt[j]
  gram+=sim[a[:,None],a[None,:]]-sim[a,w][:,None]-sim[a,w][None,:]+1
 new['aa_rbf_'+str(bw)]=gram
 new['ep_rbf_'+str(bw)]=gram**2/4
# RBF of whole sequence chemistry allows response curvature, also length-scale variation.
D=cdist(f['phys'],f['phys'],'sqeuclidean')
for bw in [2.,8.,32.]:new['whole_rbf_'+str(bw)]=np.exp(-D/(2*bw))
oldkeys=keys.copy();keys+=list(new);kernels=np.concatenate([kernels,np.array(list(new.values()))])
cache=[]
for tr,te in splits:
 kt=kernels[:,tr][:,:,tr];qt=kernels[:,te][:,:,tr];mean=kt.mean(axis=1);grand=mean.mean(axis=1)
 cache.append((kt-mean[:,:,None]-mean[:,None,:]+grand[:,None,None],qt-qt.mean(axis=2,keepdims=True)-mean[:,None,:]+grand[:,None,None],tr,te))
base=json.load(open('cv_search.json'))[:30];rng=np.random.default_rng(582);results=[]
for t in range(650):
 rr=base[t%len(base)];w=np.array([rr['weights'].get(k,0) for k in keys]);alpha=rr['alpha']
 if t>=30:
  w[:len(oldkeys)]*=10**rng.uniform(-.5,.5,len(oldkeys));alpha*=10**rng.uniform(-.5,.5)
  ix=rng.choice(len(new),rng.integers(1,4),replace=False)+len(oldkeys)
  w[ix]=10**rng.uniform(-1.5,1.5,len(ix))
 target='raw' if t%4 else ['rank','asinh'][t%8==0]
 score=evaluate(w,alpha,target)
 results.append(dict(objective=score[0],overall=score[1],double=score[2],weights=dict(zip(keys,w.tolist())),alpha=alpha,target=target))
 if t%100==0:print(t,max(r['objective'] for r in results),flush=True)
results.sort(key=lambda r:r['objective'],reverse=True);json.dump(results,open('kernel_results.json','w'),indent=2)
for rr in results[:15]:print(rr,flush=True)
np.savez('kernel_data.npz',kernels=kernels,keys=np.array(keys))
