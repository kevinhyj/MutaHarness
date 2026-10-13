from explore import *
c=json.load(open('case.json'));recs=c['support'];y=np.array([r['fitness'] for r in recs]);orders=np.array([r['mutation_order'] for r in recs]);ks=kernels(recs,c['reference_sequence'])
base=ks['p']+.3*ks['s1.5']+3*ks['chem']+.1*ks['global']
d=np.maximum(np.diag(base)[:,None]+np.diag(base)[None,:]-2*base,0)
models=[(base,.25,'sqrt'),(base,1,'log0.1'),(base+3*np.exp(-d/6),.05,'sqrt'),(base+3*np.exp(-d/20),.25,'log0.3'),(ks['p']+ks['s1.5']+10*ks['chem']+.1*ks['global'],.5,'sqrt')]
sp=[];single=np.where(orders==1)[0];double=np.where(orders>1)[0]
for seed in [123,321,271]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(y):sp.append((tr,te,'all'))
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(double):sp.append((np.r_[single,double[tr]],double[te],'feedback'))
 for tr,te in KFold(2,shuffle=True,random_state=seed).split(y):sp.append((tr,te,'thin'))
ps=[]
for tr,te,kind in sp:
 ps.append(np.array([fitpred(k,y,tr,te,a,mo)/np.std(transform(y[tr],mo)) for k,a,mo in models]))
scores={}
for name,weights in [('additive',[1,0,0,0,0]),('rbf',[0,0,1,0,0]),('additive_ensemble',[1,1,0,0,1]),('nonlinear_ensemble',[0,0,1,1,0]),('all_equal',[1,1,1,1,1]),('chosen',[1,1,2,2,1])]:
 vals=[spearmanr(y[te],np.average(p,axis=0,weights=weights)).statistic for (tr,te,kind),p in zip(sp,ps)]
 scores[name]={kind:float(np.mean([v for v,ss in zip(vals,sp) if ss[2]==kind])) for kind in ['all','feedback','thin']}
 scores[name]['sd']=float(np.std(vals));scores[name]['mean']=float(np.mean(vals))
print(json.dumps(scores,indent=2),flush=True)
json.dump(scores,open('validation_results.json','w'))
for name,weights in [('all_equal',[1,1,1,1,1]),('chosen',[1,1,2,2,1])]:
 ps0=np.array([fitpred(k,y,single,double,a,mo)/np.std(transform(y[single],mo)) for k,a,mo in models])
 print('SINGLES ONLY',name,spearmanr(y[double],np.average(ps0,axis=0,weights=weights)).statistic,flush=True)
