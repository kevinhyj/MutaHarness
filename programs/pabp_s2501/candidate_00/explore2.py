exec(open('explore.py').read().split('results=[]')[0])
results=json.load(open('explore_results.json'));top=results[0][2]
def base(cfg):return kernels['pos']+cfg['chem']*kernels['chem']+cfg['exact']*kernels['exact']+cfg['glob']*kernels['global']+cfg['smooth']*kernels['smooth2']
# Extra pooled and site-specific substitution properties.
L=len(c['reference_sequence']);AB=np.zeros((len(s),L*8));Dest=np.zeros_like(AB);SP=np.zeros((len(s),L*5));SE=np.zeros((len(s),11*17))
for i,r in enumerate(s):
 for pos,a,b in p.mutations(r,c['reference_sequence']):
  d=p.PROP[p.INDEX[b]]-p.PROP[p.INDEX[a]];j=pos-1
  AB[i,j*8:(j+1)*8]=abs(d)/np.sqrt(8);Dest[i,j*8:(j+1)*8]=p.PROP[p.INDEX[b]]/np.sqrt(8)
  sev=np.array([np.linalg.norm(d)/3, np.linalg.norm(d)**2/8,float(b=='P'),float(b=='G'),float(a in 'FYW' and b not in 'FYW')])
  SP[i,j*5:(j+1)*5]=sev
  group_id=group(pos)
  if group_id>=0:SE[i,group_id*17:(group_id+1)*17]+=np.r_[d/np.sqrt(8),abs(d)/np.sqrt(8),1]
extras={'abs':AB@AB.T,'dest':Dest@Dest.T,'severity':SP@SP.T,'se':SE@SE.T}
# Position-kernel times radial similarity of AA substitutions.
mi=[]
for r in s:mi.append([(po,p.PROP[p.INDEX[b]]-p.PROP[p.INDEX[a]]) for po,a,b in p.mutations(r,c['reference_sequence'])])
for bw in [2,5,12]:
 M=np.zeros((len(s),len(s)))
 for i in range(len(s)):
  for j in range(i+1):
   for pos,d in mi[i]:
    for po,e in mi[j]:
     if pos==po: M[i,j]+=np.exp(-np.sum((d-e)**2)/bw)
   M[j,i]=M[i,j]
 extras['rbf'+str(bw)]=M
np.savez('explore_extras.npz',**extras)
res=[]
# Second-stage check each augmentation across top distinct configurations.
bases=[results[i][2] for i in [0,4,12,30,60]]
for cfg,name,weight in itertools.product(bases,list(extras)+['group','dgroup','smooth4','smooth8','aa'],[.1,.5,2]):
 K=base(cfg)+weight*(extras[name] if name in extras else kernels[name]);sc=evaluate(K,cfg['alpha'],cfg['mode']);res.append((.7*sc[0]+.3*sc[1],sc,dict(cfg,extra=name,weight=weight)))
# Kernel square models pair effects with shrinkage.
for cfg in bases:
 K=base(cfg)
 for w,alpha in itertools.product([.02,.1,.5],[.2,1,3]):
  KK=K+w*K*K;sc=evaluate(KK,alpha,cfg['mode']);res.append((.7*sc[0]+.3*sc[1],sc,dict(cfg,poly=w,alpha=alpha)))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('explore2_results.json','w'),indent=2)
for r in res[:25]:print(r,flush=True)
