exec(open('explore3.py').read().split('res=[]')[0])
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
extras=dict(np.load('explore_extras.npz'))
K=kernels['pos']+2*kernels['exact']+.2*kernels['smooth2']+2*extras['rbf5']
res=[]
for reg in [.0003,.001,.003,.01,.03]:
 scores=[]
 for tr,te in splits:
  ki=K[np.ix_(tr,tr)]+.5*np.eye(len(tr));cf=cho_factor(ki,check_finite=False);inv=cho_solve(cf,np.eye(len(tr)),check_finite=False)
  ii,jj=np.triu_indices(len(tr),1);sgn=np.sign(y[tr][ii]-y[tr][jj]);
  def objective(z):
   dz=sgn*(z[ii]-z[jj]);loss=np.logaddexp(0,-dz).mean()+reg*.5*z@inv@z
   g=-sgn*expit(-dz)/len(ii);grad=np.bincount(ii,g,minlength=len(tr))-np.bincount(jj,g,minlength=len(tr))+reg*inv@z
   return loss,grad
  opt=minimize(objective,np.zeros(len(tr)),jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10})
  pred=K[np.ix_(te,tr)]@inv@opt.x;scores.append(spearmanr(y[te],pred).statistic)
 sc=(np.mean(scores[:12]),np.mean(scores[12:]),np.std(scores));res.append((.7*sc[0]+.3*sc[1],sc,{'rankreg':reg}))
# Tree inputs give both sorted sites and pooled chemistry. A missing second mutation duplicates first with a presence flag.
X=[]
for r in s:
 ms=sorted(p.mutations(r,c['reference_sequence']));row=[]
 for pos,a,b in (ms+ms)[:2]:
  d=p.PROP[p.INDEX[b]]-p.PROP[p.INDEX[a]]
  row.extend([pos/200,group(pos)/10]+list(d)+list(p.PROP[p.INDEX[b]])+list(abs(d)))
 row.extend([len(ms),ms[-1][0]-ms[0][0]]);X.append(row)
X=np.array(X)
for typ,leaf,feat,mode in itertools.product(['extra','rf'],[2,4,7],[.5,1.],['raw','log']):
 scores=[]
 for tr,te in splits:
  yy=y[tr] if mode=='raw' else np.log(y[tr]+.03)
  cls=ExtraTreesRegressor if typ=='extra' else RandomForestRegressor
  est=cls(n_estimators=180,min_samples_leaf=leaf,max_features=feat,random_state=55,n_jobs=1).fit(X[tr],yy)
  pred=est.predict(X[te]);scores.append(spearmanr(y[te],pred).statistic)
 sc=(np.mean(scores[:12]),np.mean(scores[12:]),np.std(scores));res.append((.7*sc[0]+.3*sc[1],sc,dict(tree=typ,leaf=leaf,feat=feat,mode=mode)))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('explore_rank_tree_results.json','w'),indent=2)
for r in res[:25]:print(r,flush=True)
