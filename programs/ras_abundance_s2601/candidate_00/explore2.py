exec(open('explore.py').read().split('configs=[]')[0])
from scipy.special import ndtri
# Additional regional main effects and mutation amino acid in regional context.
from proposal import _mutations
mm=[_mutations(r,c['reference_sequence']) for r in rows]
reg=np.zeros((n,12));regaa=np.zeros((n,12*40)); coarse=np.zeros((n,8*24));norder=np.zeros((n,2))
edges=[0,9,17,25,40,58,76,86,104,120,138,166,len(c['reference_sequence'])]
from model_features import AA,chem
for i,m in enumerate(mm):
 for p,a,z in m:
  r=max(0,min(11,np.searchsorted(edges,p,side='left')-1));reg[i,r]+=1;regaa[i,40*r+AA.index(a)]+=1;regaa[i,40*r+20+AA.index(z)]+=1
  r=min(7,(p-1)//24);desc=np.r_[chem(z)-chem(a),abs(chem(z)-chem(a)),chem(z)*chem(a)];coarse[i,24*r:24*r+24]+=desc
K['regmean']=reg@reg.T;K['regaa']=regaa@regaa.T;K['coarse']=coarse@coarse.T
# Separate search and genuinely fresh confirmation splits.
splits=[]
for seed,number,size,pool in [(470,20,20,np.arange(32,n)),(1980,30,20,np.arange(32,n)),(2980,15,24,np.arange(n))]:
 for rep in range(number):
  rng=np.random.default_rng(seed+rep);va=rng.choice(pool,size=size,replace=False);tr=np.setdiff1d(np.arange(n),va);splits.append((tr,va))
def evaluate(k,alpha,trans):
 out=[];met=[]
 for tr,va in splits:
  yt=y[tr].copy();rr=(rankdata(yt)-.5)/len(yt)
  if trans=='rank':yt=rr
  elif trans=='normal':yt=ndtri(rr)
  elif trans=='mix':yt=.5*(yt-yt.mean())/yt.std()+.5*(rr-rr.mean())/rr.std()
  kt=k[np.ix_(tr,tr)];kv=k[np.ix_(va,tr)];mu=kt.mean(0);g=mu.mean()
  kt=kt-mu[:,None]-mu[None,:]+g;kv=kv-kv.mean(1)[:,None]-mu[None,:]+g
  p=kv@solve(kt+alpha*np.eye(len(tr)),yt-yt.mean(),assume_a='pos')
  out.append(p);met.append(spearmanr(y[va],p).statistic)
 return out,met
configs=[]
for a in [.03,.1,.3,1,3]:
 for pw in [.1,.3,1]:
  for rw in [.01,.03,.1]:
   configs.append((f'reg_a{a}_p{pw}_w{rw}',K['exact']+pw*K['pos']+rw*K['region']+.1*K['count'],a,'rank'))
base=K['exact']+.3*K['pos']+.03*K['region']+.1*K['count']
for t in ['raw','normal','mix']:
 for a in [.1,1]:configs.append((f'trans_{t}_{a}',base,a,t))
for block in ['regmean','regaa','coarse','struct','aa','subst','local','chem']:
 for w in [.01,.1,1]:configs.append((f'extra_{block}_{w}',base+w*K[block],.1,'rank'))
for w in [.01,.1,1,10]:
 configs.append((f'poly_{w}',base+w*(K['exact']+.3*K['pos'])**2,.1,'rank'))
for ell in [1,3,10,30,100]:
 kk=K['region'];dd=np.maximum(0,np.diag(kk)[:,None]+np.diag(kk)[None,:]-2*kk)
 for w in [.1,1,10]:configs.append((f'rbf_{ell}_{w}',base+w*np.exp(-dd/(2*ell)),.1,'rank'))
allp={};allm={};res=[]
for name,k,a,t in configs:
 p,m=evaluate(k,a,t);allp[name]=p;allm[name]=m;res.append((np.mean(m[:20]),np.mean(m[20:50]),np.mean(m[50:]),name))
res.sort(reverse=True)
for x in res[:25]:print('best',x,flush=True)
for x in res:
 if not x[3].startswith('reg_'):print('other',x,flush=True)
# Selection only initial search results, evaluate ensemble on fresh confirm.
for nn in [3,5,10,20]:
 names=[x[3] for x in res[:nn]];metrics=[]
 for j,(tr,va) in enumerate(splits):
  pred=np.mean([(rankdata(allp[name][j])-.5)/len(va) for name in names],axis=0)
  metrics.append(spearmanr(y[va],pred).statistic)
 print('ensemble',nn,np.mean(metrics[:20]),np.mean(metrics[20:50]),np.mean(metrics[50:]),names,flush=True)
json.dump(res,open('explore2_results.json','w'),indent=2)
