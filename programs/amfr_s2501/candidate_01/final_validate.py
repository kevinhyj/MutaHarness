exec(open('rank_explore.py').read().split('\nbasecfgs=')[0])
configs=[({'R3.0':1.,'S':1.,'G':1.,'pairS':.1},'ridge',2.),({'P':1.,'S':1.,'G':1.,'pairS':.3,'DEST':.3},'ridge',2.),({'R3.0':1.,'B':.1,'S':1.,'G':1.,'pairS':.3},'ridge',2.),({'P':1.,'S':1.,'G':1.,'pairS':.3},'pair',.01),({'R3.0':1.,'B':.5,'S':1.,'G':1.,'pairS':.3},'pair',.003)]
ss=[]
for seed in [1901,1902,1903,1904]:
 for tr,va in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,n)):ss.append((np.r_[np.arange(32),32+tr],32+va))
ss+=all_splits
metrics=[];dp=[];dy=[];pools=[];observed=[]
for tr,va in ss:
 ps=[]
 for cfg,kind,rg in configs:
  K=sum(kernels[k]*v for k,v in cfg.items())
  p=fit_predict(K,'rank',rg,tr,va) if kind=='ridge' else rankfit(K,tr,va,rg,'all')
  ps.append((rankdata(p)-.5)/len(p))
 ps=np.array(ps);variants=np.r_[ps,ps[:3].mean(0)[None,:],ps.mean(0)[None,:],(.7*ps[:3].mean(0)+.3*ps[3:].mean(0))[None,:]]
 metrics.append([spearmanr(p,y[va]).statistic for p in variants]);m=np.array([rows[i]['mutation_order']==2 for i in va]);dp.append(variants[:,m]);dy.extend(y[va][m]);pools.append(variants);observed.extend(y[va])
print('Feedback-only metrics',np.mean(metrics[:16],axis=0))
print('All-support folds',np.mean(metrics[16:],axis=0))
print('Pooled doubles',[spearmanr(p,dy).statistic for p in np.concatenate(dp,axis=1)])
json.dump({'feedback_mean':list(np.mean(metrics[:16],axis=0)),'all_support_mean':list(np.mean(metrics[16:],axis=0)),'pooled_double':[spearmanr(p,dy).statistic for p in np.concatenate(dp,axis=1)],'models':configs},open('cv_confirmation.json','w'),indent=2)
