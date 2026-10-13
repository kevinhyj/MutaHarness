exec(open('explore.py').read().split('results=[];start=time.time()')[0])
from scipy.optimize import nnls
# Transform target ranks USING TRAINING LABELS ONLY.
def fit_predict(K,kind,rg,tr,va):
 z=y[tr].copy()
 if kind=='rank':z=norm.ppf((rankdata(z)-.5)/len(z))
 elif kind.startswith('floor'):
  f,t=kind[5:].split('t');f=float(f);t=float(t)
  z=f+t*np.log(np.expm1(np.maximum(z-f,.02)/t))
 mean=z.mean();return K[np.ix_(va,tr)]@cho_solve(cho_factor(K[np.ix_(tr,tr)]+rg*np.eye(len(tr))),z-mean)+mean

def evaluate(cfg,rg,kind,ss=splits,ret=False):
 K=sum(kernels[k]*v for k,v in cfg.items());vals=[];pred=[];obs=[];dp=[];dy=[]
 for tr,va in ss:
  p=fit_predict(K,kind,rg,tr,va);pred.extend(p);obs.extend(y[va]);vals.append(spearmanr(p,y[va]).statistic)
  m=np.array([rows[i]['mutation_order']==2 for i in va]);dp.extend(p[m]);dy.extend(y[va][m])
 return (np.mean(vals),spearmanr(pred,obs).statistic,spearmanr(dp,dy).statistic),np.array(pred)

# Global substitution features condition the biochemical change on the native
# residue. This is a generic transfer model, not a structural annotation.
d=proposal.descriptors();native=d[[proposal.AA.index(a) for a in ref]]
props=prop
cross=np.einsum('nlj,lk->njk',props,native).reshape(n,-1)/9
absprop=np.abs(props).sum(1)/3
dest=one.sum(1)
nativeone=np.array([[int(a==w) for a in proposal.AA] for w in ref])
natives=site@nativeone
kernels['X']=cross@cross.T
kernels['ABS']=absprop@absprop.T
kernels['DEST']=dest@dest.T
kernels['NAT']=natives@natives.T
kernels['C2']=site.sum(1)[:,None]**2@site.sum(1)[None,:]**2/4
results=[];start=time.time()
# Fixed first-stage grid, reevaluated with proper fold-specific ranks.
for aa in ['O','P','R0.3','R1.0','R3.0']:
 for s in [.2,1.]:
  for g in [.2,1.,3.]:
   for rg in [.1,.5,2.]:
    for u in [0.,1.]:
     cfg={aa:1.,'S':s,'G':g,'U':u}
     for t in ['raw','rank','floor-3.5t0.5','floor-3.4t0.8']:
      met,p=evaluate(cfg,rg,t);results.append((met,cfg,rg,t))
print('STAGE A')
for r in sorted(results,key=lambda a:a[0][0],reverse=True)[:10]:print(r)
# Extend only the better distinct configurations (avoid enormous search).
bases=[]
for r in sorted(results,key=lambda a:a[0][0],reverse=True):
 if len(bases)>=12:break
 if not any(r[3]==b[3] and next(iter(r[1]))==next(iter(b[1])) for b in bases):bases.append(r)
for met,cfg,rg,t in bases:
 for key in ['X','ABS','DEST','NAT','pairS','pairR']:
  for wt in [.03,.1,.3,1.,3.]:
   cc=dict(cfg);cc[key]=wt;m,p=evaluate(cc,rg,t);results.append((m,cc,rg,t))
print('STAGE B')
ordered=sorted(results,key=lambda a:a[0][0],reverse=True)
for r in ordered[:25]:print(r)
json.dump([{'metric':list(m),'config':cfg,'reg':rg,'target':t} for m,cfg,rg,t in ordered],open('cv_refined.json','w'),indent=2)
np.savez('exploration_kernels.npz',**kernels)
print('elapsed',time.time()-start)
