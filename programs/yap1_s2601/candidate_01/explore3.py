exec(open('explore2.py').read().split('prev=json.load')[0])
# Share sidechain category changes conditional on original chemical category.
OC=np.zeros((n,100));OP=np.zeros((n,100))
for i,mi in enumerate(mut):
 for p,w,a in mi:
  ai=proposal.AA.index(a);wi=proposal.AA.index(w)
  for k,cat in enumerate(cats):
   if w not in cat:continue
   OP[i,k*10:(k+1)*10]+=(proposal.PROPS[ai]-proposal.PROPS[wi])/np.sqrt(10)
   for l,other in enumerate(cats):OC[i,k*10+l]+=float(a in other)-float(w in other)
components.update(origcat=OC@OC.T,origprop=OP@OP.T)
def kernel(cfg):
 core=sum(components[k]*cfg.get(k,0) for k in ['pos','exact','prop','global'])
 K=core+cfg.get('interaction',0)*core**2/max(np.diag(core).mean(),1e-8)
 for k in components:
  if k not in ['pos','exact','prop','global']:K+=cfg.get(k,0)*components[k]
 return K
base={'exact':.15,'pos':1.,'prop':.4,'global':.2,'alpha':.03,'target':'log','wtcat':1.,'interaction':.2}
configs=[base]
for key in ['wt','wtprop','origcat','origprop','cat','substitution']:
 for weight in [.03,.1,.3,1.]:configs.append(dict(base,**{key:weight}))
for alpha in [.01,.1,.3,1.]:configs.append(dict(base,alpha=alpha))
for iw in [0.,.05,.7,2.]:configs.append(dict(base,interaction=iw))
for scale in ['log.1','sqrt','rank']:configs.append(dict(base,target=scale))
for wc in [.1,.3,3.]:configs.append(dict(base,wtcat=wc))
results=[]
for cfg in configs:
 a,b,p=validate(kernel(cfg),cfg['alpha'],cfg['target']);results.append((.7*a+.3*b,a,b,cfg,p))
results.sort(key=lambda r:r[0],reverse=True)
print('DEVELOPMENT')
for r in results[:15]:print(r[:4])
# New seeds withheld from the earlier grids; their results are diagnostic and
# their outcomes are not used to search a new hyperparameter grid.
splits=[]
for seed in [2026,578,341,937,1311]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback):splits.append((np.r_[np.arange(32),feedback[tr]],feedback[te],'fb'))
for seed in [2044,721,1711]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(s):splits.append((tr,te,'all'))
chosen=[r[3] for r in results[:4]]
# diverse, slightly more regularized log(.1), and nonlinear alternatives.
chosen.extend([dict(base,alpha=.15,target='log.1',exact=.5),dict(base,wtcat=.3,wtprop=.3),dict(base,origprop=.3,interaction=0)])
old=json.load(open('explore_results.json'))
chosen.append(old[0]['config'])
allp=[]
print('CONFIRMATORY NEW SPLITS')
for cfg in chosen:
 ps=[];metrics=[]
 for tr,te,typ in splits:
  pred=fitpred(kernel(cfg),y,tr,te,cfg['alpha'],cfg['target']);ps.append(pred);metrics.append(spearmanr(pred,y[te]).statistic)
 allp.append(ps)
 print(np.mean(metrics[:20]),np.mean(metrics[20:]),cfg)
for inds in [list(range(4)),list(range(7)),[0,4,5,6],[0,4],[0,7]]:
 metrics=[]
 for i,(tr,te,typ) in enumerate(splits):
  pred=sum(rankdata(allp[j][i])/len(te) for j in inds)/len(inds)
  metrics.append(spearmanr(pred,y[te]).statistic)
 print('ensemble',inds,np.mean(metrics[:20]),np.mean(metrics[20:]))
json.dump(chosen,open('candidate_configs.json','w'),indent=2)
