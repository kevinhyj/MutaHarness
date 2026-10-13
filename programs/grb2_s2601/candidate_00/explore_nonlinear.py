exec(open('explore.py').read().split('rng=np.random.default_rng')[0])
results=json.load(open('validation_grid.json'))
configs=[]
seen=set()
for r in results:
 key=json.dumps(r['cfg'],sort_keys=True)
 if key not in seen:configs.append(r['cfg']);seen.add(key)
 if len(configs)==10:break
res=[]
for cfg in configs:
 K=sum(v*bases[k] for k,v in cfg.items() if k!='alpha')
 for lo in [-1.55,-1.7,-2.,-2.5,-4.]:
  for hi in [.3,.5,1.,2.,10.]:
   z=np.log((y-lo)/(hi-y));score,r,m,p=evaluate(K,cfg['alpha'],z)
   res.append({'score':score,'feedback':r,'mixed':m,'cfg':cfg,'scale':[lo,hi]})
 for w in [.01,.05,.2,.6,2.]:
  # A tensor feature kernel allows softly regularized interactions.
  K2=K+w*K*K/np.mean(np.diag(K))
  for scale in ['raw','floor2','rank']:
   z=y if scale=='raw' else np.log(y+2) if scale=='floor2' else norm.ppf((rankdata(y)-.5)/n)
   score,r,m,p=evaluate(K2,cfg['alpha'],z)
   res.append({'score':score,'feedback':r,'mixed':m,'cfg':cfg,'scale':scale,'interaction':w})
res.sort(key=lambda r:-r['score']);json.dump(res,open('validation_nonlinear.json','w'),indent=2)
print('Best transformed/interaction',res[:8],flush=True)
# Similarity based on smoothly shared chemical states at each mutated position.
ms=[proposal.mutations(r,c['reference_sequence']) for r in s]
Ds=np.stack([proposal.DESC[proposal.AA.index(m)] for r in ms for p,w,m in r]);pos=np.array([p for r in ms for p,w,m in r]);wt=np.stack([proposal.DESC[proposal.AA.index(w)] for r in ms for p,w,m in r]);owner=np.array([i for i,r in enumerate(ms) for _ in r]);A=np.eye(n)[owner].T
physdist=((Ds[:,None]-Ds[None,:])**2).sum(2)/8
wtdist=((wt[:,None]-wt[None,:])**2).sum(2)/8
same=(pos[:,None]==pos[None,:]);pd=abs(pos[:,None]-pos[None,:])
regions=[(159,165),(166,174),(175,180),(181,190),(191,194),(195,200),(201,205),(206,208),(209,214)]
reg=np.array([next((i for i,(a,b) in enumerate(regions) if a<=p+1<=b),9) for p in pos]);strand=np.isin(reg,[0,2,4,6,8]);parity=pos%2
shared=(reg[:,None]==reg[None,:])*((parity[:,None]==parity[None,:])|~strand[:,None])
res2=[]
for bandwidth in [.2,.5,1.,2.,5.]:
 sim=np.exp(-physdist/bandwidth)
 for wtshare in [0.,.03,.1,.3]:
  for local in [0.,.1,.3]:
   for structural in [0.,.1,.3]:
    if local and structural:continue
    mutkernel=sim*(same+wtshare*np.exp(-wtdist)+local*np.exp(-pd/3)+structural*shared)
    K=A@mutkernel@A.T+.02*bases['global']+.2*bases['order']
    for alpha in [.1,.4,1.,3.]:
     for scale in ['raw','floor2','rank']:
      z=y if scale=='raw' else np.log(y+2) if scale=='floor2' else norm.ppf((rankdata(y)-.5)/n)
      score,r,m,p=evaluate(K,alpha,z)
      res2.append({'score':score,'feedback':r,'mixed':m,'cfg':{'bandwidth':bandwidth,'wtshare':wtshare,'local':local,'structural':structural,'alpha':alpha},'scale':scale})
res2.sort(key=lambda r:-r['score']);json.dump(res2,open('validation_similarity.json','w'),indent=2)
print('Best chemical similarity',res2[:8])
