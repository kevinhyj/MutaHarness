exec(open('explore.py').read().split('configs=[]')[0])
from scipy.special import ndtr
# Cross-site transfer conditional on reference residue chemistry.
x=proposal.encode(s,c['reference_sequence']);wt=np.array([proposal.INDEX[a] for a in c['reference_sequence']]);z=x-x.sum(2)[:,:,None]*np.eye(20)[wt][None,:,:]
# tensor: source physicochemistry x mutation physicochemistry.
delta=z@proposal.PROP
src=np.c_[np.ones(len(wt)),proposal.PROP[wt]]
f=np.einsum('npd,pe->nde',delta,src).reshape(n,-1)
ktransfer=f@f.T/f.shape[1]
# Mutation magnitude allows shared nonlinear costs of drastic substitutions.
mag=np.abs(delta).sum(1); kmag=mag@mag.T/mag.shape[1]
# Pair descriptors, pooled across all interacting positions (nonlinear chemistry).
pf=np.einsum('npd,npe->nde',delta,delta)
psum=delta.sum(1);pf=(np.einsum('nd,ne->nde',psum,psum)-pf)/2
pf=pf.reshape(n,-1);kchem=pf@pf.T/pf.shape[1]
# Transfer using categorical original amino acid rather than its chemistry.
f2=np.einsum('npa,pb->nab',delta,np.eye(20)[wt]).reshape(n,-1);kref=f2@f2.T/proposal.PROP.shape[1]
# Position sharing based on sequence predicted amphipathic environment; hypothesis only.
h=np.array([proposal.HYD[proposal.INDEX[a]] for a in c['reference_sequence']]);context=[]
for p in range(len(wt)):
 near=[h[j] for j in range(max(0,p-4),min(len(wt),p+5)) if abs(p-j) in [3,4]]
 context.append(np.mean(near) if near else 0)
context=np.array(context);context=(context-context.mean())/context.std();fc=(delta*context[None,:,None]).sum(1);kenv=fc@fc.T/proposal.PROP.shape[1]
extra=[ktransfer,kmag,kchem,kref,kenv]
def ev2(cfg,fs=folds):
 base,weights=cfg;l,ident,pos,glob,count,pair,noise,transform=base
 p=parts[l];k=p[0]+ident*p[1]+pos*p[2]+glob*p[3]+count*p[4]+pair*p[5]+1
 for w,ex in zip(weights,extra):k+=w*ex
 pred=np.zeros((len(fs)//4,n))
 for j,(tr,va) in enumerate(fs):
  co=cho_solve(cho_factor(k[np.ix_(tr,tr)]+noise*np.eye(len(tr)),lower=True,check_finite=False),trans[transform](y[tr]),check_finite=False)
  pred[j//4,va]=k[np.ix_(va,tr)]@co
 a=np.mean([spearmanr(y[32:],pr[32:]).statistic for pr in pred]);b=np.mean([spearmanr(y[order==2],pr[order==2]).statistic for pr in pred]);mask=(order==1)&(np.arange(n)>=32);d=np.mean([spearmanr(y[mask],pr[mask]).statistic for pr in pred]);return .75*b+.25*a,a,b,d,pred
configs=[]
for _ in range(1500):
 base=(rng.choice(list(parts)),rng.choice([0,.03,.1,.3]),rng.choice([.1,.3,1,3]),rng.choice([.1,.3,1,3,10]),rng.choice([0,.3,1]),rng.choice([0,.1,.3,1]),rng.choice([.03,.1,.3,1,3]),rng.choice(['raw','rank','asinh']))
 weights=[rng.choice([0,0,.1,.3,1,3]) for _ in extra]
 configs.append((base,weights))
results=[];t=time.time()
for cfg in configs:
 score,a,b,d,p=ev2(cfg);results.append((score,a,b,d,cfg))
results.sort(reverse=True)
print('seconds',time.time()-t)
for r in results[:30]: print(r)
json.dump(results,open('cv2_results.json','w'),indent=2)
for num in [3,5,10,20,40]:
 prs=[ev2(r[-1])[-1] for r in results[:num]];pred=np.mean([np.array([rankdata(p[32:]) for p in pr]) for pr in prs],0)
 print('ensemble',num,'all',np.mean([spearmanr(y[32:],p).statistic for p in pred]),'double',np.mean([spearmanr(y[32:][order[32:]==2],p[order[32:]==2]).statistic for p in pred]))
