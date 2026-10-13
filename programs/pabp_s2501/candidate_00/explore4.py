exec(open('explore3.py').read().split('\nres=[]')[0])
extras=dict(np.load('explore_extras.npz'))
# Local chemistry kernels and sequence-derived motif classes.
L=len(c['reference_sequence']);loc=np.arange(L)+1;dist=np.abs(loc[:,None]-loc[None,:]);
Q=np.zeros((L,7))
# RNP1/RNP2 motifs are sequence hypotheses; the remaining sites are not presumed functional.
for j in range(L):
 pos=j+1;aa=c['reference_sequence'][j]
 Q[j,0]=float(127<=pos<=133);Q[j,1]=float(166<=pos<=173)
 Q[j,2]=float(aa in 'FYW' and (127<=pos<=133 or 166<=pos<=173))
 Q[j,3]=float(aa in 'ILVM' and (127<=pos<=133 or 166<=pos<=173))
 Q[j,4]=float(aa=='G' and 166<=pos<=173)
 Q[j,5]=float(aa in 'KRH' and (127<=pos<=133 or 166<=pos<=173))
 Q[j,6]=float(179<=pos<=190)
context=Q@Q.T
for j in range(7):
 T=np.zeros((len(s),8))
 for i,r in enumerate(s):
  for pos,a,b in p.mutations(r,c['reference_sequence']):T[i]+=Q[pos-1,j]*(p.PROP[p.INDEX[b]]-p.PROP[p.INDEX[a]])/np.sqrt(8)
 extras['motif'+str(j)]=T@T.T
extras['motifall']=sum(extras['motif'+str(j)] for j in range(7))
extras['motifpos']=P@context@P.T
for w in [1,2,4,8]:
 local=np.exp(-dist**2/(2*w*w))
 extras['localchem'+str(w)]=sum(D[:,a::8]@local@D[:,a::8].T for a in range(8))
# Smooth RBF mutation kernel borrowing between neighboring sites.
for bw in [5,12]:
 for width in [2,6]:
  M=np.zeros((len(s),len(s)))
  for i,r in enumerate(s):
   for j,t in enumerate(s[:i+1]):
    for pos,a,b in p.mutations(r,c['reference_sequence']):
     d=p.PROP[p.INDEX[b]]-p.PROP[p.INDEX[a]]
     for po,aa,bb in p.mutations(t,c['reference_sequence']):
      e=p.PROP[p.INDEX[bb]]-p.PROP[p.INDEX[aa]]
      M[i,j]+=np.exp(-(pos-po)**2/(2*width*width)-np.sum((d-e)**2)/bw)
    M[j,i]=M[i,j]
  extras['localrbf'+str(bw)+'w'+str(width)]=M
np.savez('explore4_extras.npz',**extras)
basecfgs=[dict(chem=0,exact=2,smooth=.2,alpha=.6,mode='log',offset=.1,intercept='mean',rbf=2),dict(chem=1.5,exact=.5,smooth=.2,alpha=1,mode='log',offset=.03,intercept='mean',rbf=0)]
res=[]
for cfg,name,weight in itertools.product(basecfgs,[k for k in extras if k.startswith(('local','motif'))],[.05,.2,1]):
 K=kernels['pos']+cfg['chem']*kernels['chem']+cfg['exact']*kernels['exact']+cfg['smooth']*kernels['smooth2']+cfg['rbf']*extras['rbf5']+weight*extras[name]
 sc=evaluate3(K,cfg['alpha'],cfg['mode'],cfg['offset'],cfg['intercept']);res.append((.7*sc[0]+.3*sc[1],sc,dict(cfg,extra=name,weight=weight)))
# RBF tuning with target, intercept, smoothness, regularization.
for rbf,bw,exact,smooth,alpha,offset,intercept in itertools.product([1,4],[2,5,12],[.5,2],[.2,.7],[.3,2],[.03,.1],['mean','gls']):
 K=kernels['pos']+exact*kernels['exact']+smooth*kernels['smooth2']+rbf*extras['rbf'+str(bw)]
 sc=evaluate3(K,alpha,'log',offset,intercept);cfg=dict(chem=0,exact=exact,smooth=smooth,alpha=alpha,mode='log',offset=offset,intercept=intercept,rbf=rbf,bw=bw)
 res.append((.7*sc[0]+.3*sc[1],sc,cfg))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('explore4_results.json','w'),indent=2)
for r in res[:25]:print(r,flush=True)
