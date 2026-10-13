exec(open('explore3.py').read().split('\nres=[]')[0])
extras=dict(np.load('explore4_extras.npz'))
configs=[
 dict(name='linear_local',chem=1.5,exact=.5,smooth=.2,alpha=1,mode='log',offset=.03,intercept='mean',rbf=0,local=1,localbw=12),
 dict(name='rbf_local',chem=0,exact=2,smooth=.2,alpha=.6,mode='log',offset=.1,intercept='mean',rbf=2,local=1,localbw=5),
 dict(name='rbf_gls',chem=0,exact=.5,smooth=.2,alpha=2,mode='log',offset=.03,intercept='gls',rbf=4,local=0,localbw=5),
 dict(name='rbf_gls_low',chem=0,exact=2,smooth=.2,alpha=.3,mode='log',offset=.03,intercept='gls',rbf=4,local=0,localbw=5)]
Ks=[]
for cfg in configs:Ks.append(kernels['pos']+cfg['chem']*kernels['chem']+cfg['exact']*kernels['exact']+cfg['smooth']*kernels['smooth2']+cfg['rbf']*extras['rbf5']+cfg['local']*extras['localrbf'+str(cfg['localbw'])+'w2'])
def predictions(K,cfg,split_list):
 out=[]
 for tr,te in split_list:
  yy=np.log(y[tr]+cfg['offset']);cf=cho_factor(K[np.ix_(tr,tr)]+cfg['alpha']*np.eye(len(tr)),check_finite=False);m=yy.mean()
  if cfg['intercept']=='gls':
   iz=cho_solve(cf,np.c_[yy,np.ones(len(tr))],check_finite=False);m=iz[:,0].sum()/iz[:,1].sum()
  pr=m+K[np.ix_(te,tr)]@cho_solve(cf,yy-m,check_finite=False)
  out.append((pr-yy.mean())/max(yy.std(),1e-8))
 return out
pred=[predictions(K,cfg,splits) for K,cfg in zip(Ks,configs)]
res=[]
for a in np.arange(0,1.01,.1):
 for b in np.arange(0,1.01-a,.1):
  for cc in np.arange(0,1.01-a-b,.1):
   w=np.array([a,b,cc,1-a-b-cc]);sc=[]
   for z,(tr,te) in enumerate(splits):sc.append(spearmanr(y[te],sum(w[i]*pred[i][z] for i in range(4))).statistic)
   sc=np.array(sc);res.append((.7*sc[:12].mean()+.3*sc[12:].mean(),sc[:12].mean(),sc[12:].mean(),w.tolist()))
res.sort(reverse=True,key=lambda x:x[0]);print('Best blend',res[:8],flush=True)
# Fresh splits: preserve exactly 32 original records, random all, smaller 48-point fits.
fresh=[];types=[]
for seed in [401,809,1327]:
 fb=np.arange(32,len(s))
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(fb):fresh.append((np.r_[np.arange(32),fb[tr]],fb[te]));types.append('feedback')
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(idx):fresh.append((tr,te));types.append('all')
 rng=np.random.default_rng(seed)
 for _ in range(4):
  ix=rng.permutation(len(s));fresh.append((ix[:48],ix[48:]));types.append('small')
pf=[predictions(K,cfg,fresh) for K,cfg in zip(Ks,configs)]
weights=[res[0][3],[.4,.3,.3,0],[.25,.25,.25,.25],[.5,.5,0,0]]
report={}
for name,w in [(cfg['name'],np.eye(4)[i]) for i,cfg in enumerate(configs)]+[('blend'+str(i),w) for i,w in enumerate(weights)]:
 sc=np.array([spearmanr(y[te],sum(w[i]*pf[i][z] for i in range(4))).statistic for z,(tr,te) in enumerate(fresh)])
 report[name]={t:float(sc[np.array(types)==t].mean()) for t in ['feedback','all','small']}
 report[name]['sd']=float(sc.std());print(name,w,report[name],flush=True)
json.dump({'configs':configs,'blend_search':res[:20],'fresh_validation':report,'weights':weights},open('ensemble_validation.json','w'),indent=2)
