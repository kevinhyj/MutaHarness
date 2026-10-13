exec(open('explore.py').read().split('results=[];t=time.time()')[0])
# New hierarchical levels: common effects of mutating a reference residue,
# reference residue x chemical change, and shared mutation categories.
W=np.zeros((n,20)); WP=np.zeros((n,20*10)); C=np.zeros((n,P.shape[1]*10));WC=np.zeros((n,20*10));dest=np.zeros((n,20));origdest=np.zeros((n,20*20));regionpos=np.zeros((n,6))
cats=['AVLIM','FWY','STNQ','DE','KRH','P','G','C','AGST','NQ']
position_list=sorted({p for mi in mut for p,w,a in mi});pi={p:i for i,p in enumerate(position_list)}
for i,mi in enumerate(mut):
 for p,w,a in mi:
  ai=proposal.AA.index(a);wi=proposal.AA.index(w);j=pi[p];r=region(p)
  W[i,wi]+=1
  WP[i,wi*10:(wi+1)*10]+=(proposal.PROPS[ai]-proposal.PROPS[wi])/np.sqrt(10)
  dest[i,ai]+=1
  for k,cat in enumerate(cats):
   delta=float(a in cat)-float(w in cat)
   C[i,j*10+k]+=delta; WC[i,wi*10+k]+=delta
  origdest[i,wi*20+ai]+=1
  regionpos[i,r]+=1
components.update(wt=W@W.T,wtprop=WP@WP.T,cat=C@C.T,wtcat=WC@WC.T,dest=dest@dest.T,regionpos=regionpos@regionpos.T)
prev=json.load(open('explore_results.json'))
bases=[prev[i]['config'] for i in [0,1,2,6,16]]
results=[];t=time.time()
for base in bases:
 base=dict(base); iw=base.pop('interaction',0)
 K0=sum(components[k]*v for k,v in base.items() if k in components)
 if iw:K0+=iw*K0*K0/np.diag(K0).mean()
 for key in ['wt','wtprop','cat','wtcat','dest','regionpos']:
  for weight in [.03,.1,.3,1.,3.]:
   cfg=dict(base,**{key:weight})
   if iw:cfg['interaction']=iw
   a,b,p=validate(K0+weight*components[key],cfg['alpha'],cfg['target'])
   results.append((.7*a+.3*b,a,b,cfg,p))
results.sort(key=lambda r:r[0],reverse=True)
for r in results[:20]:print(r[:4])
print('elapsed',time.time()-t)
np.savez('explore2_results.npz',predictions=np.array([np.concatenate(r[4]) for r in results]),configs=np.array([json.dumps(r[3]) for r in results]),metrics=np.array([r[:3] for r in results]))
json.dump([{'metric':r[0],'fb':r[1],'all':r[2],'config':r[3]} for r in results],open('explore2_results.json','w'),indent=2)
