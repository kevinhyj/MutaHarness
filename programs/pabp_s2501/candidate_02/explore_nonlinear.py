from explore import *
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor

c=json.load(open('case.json')); recs=c['support'];y=np.array([r['fitness'] for r in recs]);orders=np.array([r['mutation_order'] for r in recs]);ks=kernels(recs,c['reference_sequence']);sp=splits(y,orders)
base=ks['p']+.3*ks['s1.5']+3*ks['chem']+.1*ks['global']
results=[]
for typ in ['rbf','poly']:
 for chem in [0,1,3]:
  k=ks['p']+.3*ks['s1.5']+chem*ks['chem']+.1*ks['global']
  for length in [.3,1,3,10]:
   if typ=='rbf':
    d=np.maximum(np.diag(k)[:,None]+np.diag(k)[None,:]-2*k,0)
    nonlin=np.exp(-d/(2*length))
   else:nonlin=k*k/length
   for v in [.3,1,3]:
    kk=k+v*nonlin
    for a in [.05,.25,1,4]:
     for mode in ['raw','sqrt','log0.1','log0.3']:
      val=evaluate(kk,y,sp,a,mode)
      results.append((val,typ,chem,length,v,a,mode))
results.sort(reverse=True);print('NONLINEAR',results[:20],flush=True)
json.dump(results,open('nonlinear_results.json','w'))

p,t,pos,ms=basic_features(recs,c['reference_sequence'])
# Trees use ordered mutation slots, each with site, source/target descriptors,
# and changes. Duplicate a singleton slot to avoid an artificial missing-site code.
x=[]
for m in ms:
 m=sorted(m)
 slots=[]
 for po,a,b in m:
  pp,qq=props[AA.index(a)],props[AA.index(b)]
  slots.append(np.r_[po,region(po),pp,qq,qq-pp,abs(qq-pp)])
 if len(slots)==1:slots=slots*2
 x.append(np.r_[len(m),slots[0],slots[-1],slots[0]+slots[-1],abs(slots[0]-slots[-1])])
x=np.array(x)
trees=[]
for kind in ['extra','rf']:
 for leaf in [1,2,3,5]:
  for feat in [.5,1.]:
   for mode in ['raw','sqrt','log0.1']:
    vals=[];preds=[]
    for tr,te,k in sp:
     ctor=ExtraTreesRegressor if kind=='extra' else RandomForestRegressor
     model=ctor(n_estimators=150,min_samples_leaf=leaf,max_features=feat,random_state=1977,n_jobs=1)
     model.fit(x[tr],transform(y[tr],mode));pred=model.predict(x[te]);preds.append(pred);vals.append(spearmanr(y[te],pred).statistic)
    trees.append((np.mean(vals),kind,leaf,feat,mode,preds))
trees.sort(key=lambda r:-r[0]);print('TREES',[(s,k,l,f,m) for s,k,l,f,m,_ in trees[:15]],flush=True)
json.dump([(s,k,l,f,m,[v.tolist() for v in ps]) for s,k,l,f,m,ps in trees],open('tree_results.json','w'))
