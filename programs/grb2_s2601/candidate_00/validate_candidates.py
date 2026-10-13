exec(open('explore_nonlinear.py').read().split('results=json.load')[0])
# Fresh split seeds and training-size sensitivity, for a short candidate list.
splits=[]
for seed in [101,223,347]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback):splits.append((np.r_[originals,feedback[tr]],feedback[te]))
for tr,te in KFold(4,shuffle=True,random_state=457).split(np.arange(n)):splits.append((tr,te))
for seed in [571,683]:
 for tr,te in KFold(2,shuffle=True,random_state=seed).split(feedback):splits.append((np.r_[originals,feedback[tr]],feedback[te]))
ms=[proposal.mutations(r,c['reference_sequence']) for r in s]
Ds=np.stack([proposal.DESC[proposal.AA.index(m)] for r in ms for p,w,m in r]);pos=np.array([p for r in ms for p,w,m in r]);wt=np.stack([proposal.DESC[proposal.AA.index(w)] for r in ms for p,w,m in r]);owner=np.array([i for i,r in enumerate(ms) for _ in r]);A=np.eye(n)[owner].T
physdist=((Ds[:,None]-Ds[None,:])**2).sum(2)/8;wtdist=((wt[:,None]-wt[None,:])**2).sum(2)/8;same=(pos[:,None]==pos[None,:]);pd=abs(pos[:,None]-pos[None,:])
regions=[(159,165),(166,174),(175,180),(181,190),(191,194),(195,200),(201,205),(206,208),(209,214)]
reg=np.array([next((i for i,(a,b) in enumerate(regions) if a<=p+1<=b),9) for p in pos]);strand=np.isin(reg,[0,2,4,6,8]);parity=pos%2;shared=(reg[:,None]==reg[None,:])*((parity[:,None]==parity[None,:])|~strand[:,None])
models=[]
for bw in [1.,2.,4.]:
 for alpha in [.4,1.]:
  for scale in ['raw','floor2','rank']:
   for structure in [0.,.1]:
    cfg={'bandwidth':bw,'wtshare':.3,'structural':structure,'alpha':alpha}
    K=A@(np.exp(-physdist/bw)*(same+.3*np.exp(-wtdist)+structure*shared))@A.T+.02*bases['global']+.2*bases['order']
    models.append({'type':'similarity','cfg':cfg,'scale':scale,'K':K})
for entry in json.load(open('validation_nonlinear.json'))[:8]+json.load(open('validation_grid.json'))[:4]:
 if entry['scale']=='rank':continue
 cfg=entry['cfg'];K=sum(v*bases[k] for k,v in cfg.items() if k!='alpha');weight=entry.get('interaction',0.)
 if weight:K=K+weight*K*K/np.mean(np.diag(K))
 models.append({'type':'linear','cfg':cfg,'scale':entry['scale'],'interaction':weight,'K':K})

def target(z,scale):
 if scale=='raw':return z
 if scale=='floor2':return np.log(z-min(-2.,z.min()-.1))
 if scale=='floor16':return np.log(z-min(-1.6,z.min()-.1))
 if scale=='rank':return norm.ppf((rankdata(z)-.5)/len(z))
 if isinstance(scale,list):return np.log((z-scale[0])/(scale[1]-z))
 raise ValueError(scale)
preds=[];stats=[]
for model in models:
 ps=[];K=model['K']
 for tr,te in splits:
  z=target(y[tr],model['scale']);k=K[np.ix_(tr,tr)];q=K[np.ix_(te,tr)];mean=k.mean(0);qm=q.mean(1);km=k.mean();k=k-mean[:,None]-mean[None,:]+km;q=q-mean[None,:]-qm[:,None]+km
  p=q@solve(k+np.eye(len(tr))*model['cfg']['alpha'],z-z.mean(),assume_a='pos')
  ps.append(p/max(z.std(),1e-8))
 preds.append(ps)
 def metrics(ps):
  out=[]
  for start,end in [(0,4),(4,8),(8,12),(12,16),(16,18),(18,20)]:
   pp=np.zeros(n);idx=[]
   for i in range(start,end):te=splits[i][1];pp[te]=ps[i];idx.extend(te)
   out.append(spearmanr(y[idx],pp[idx]).statistic)
  return out
 m=metrics(ps);score=np.mean(m[:3])*.6+m[3]*.2+np.mean(m[4:])*.2
 stats.append({'score':score,'metrics':m,**{k:v for k,v in model.items() if k!='K'}})
order=np.argsort([-r['score'] for r in stats]);json.dump([stats[i] for i in order],open('validation_confirmatory.json','w'),indent=2)
for i in order[:12]:print(i,stats[i],flush=True)
# Broad unweighted ensembles reduce selection variance.
for inds in [[i for i,m in enumerate(models) if m['type']=='similarity'],list(order[:6]),list(order[:12]),[i for i,m in enumerate(models) if m['type']=='similarity' and m['cfg']['bandwidth']==2.],list(range(len(models)))]:
 ps=[np.mean([preds[i][j] for i in inds],axis=0) for j in range(len(splits))];print('ENSEMBLE',inds,metrics(ps),flush=True)
np.savez('confirmatory_predictions.npz',**{str(i):np.concatenate(p) for i,p in enumerate(preds)})
