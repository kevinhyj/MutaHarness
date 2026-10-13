exec(open('explore.py').read().split('for scale in [')[0])
from scipy.optimize import minimize
from scipy.special import expit
positions=np.array([int(m[1:-1]) for m in vocab]); ii,jj=np.triu_indices(x.shape[1],1); different=positions[ii]!=positions[jj];ii,jj=ii[different],jj[different]
# Smooth pair shrinkage using main-effect importance at each sequence position.
for method in ['equal','strong','product','specific','order']:
 for pw in [.1,.3,1.]:
  for a in [.01,.03,.1,.3]:
   def pred(tr,va,method=method,pw=pw,a=a):
    xm=x[tr].mean(0); xt=x[tr]-xm;yt=y[tr]-y[tr].mean()
    co=xt.T@solve(xt@xt.T+.1*np.eye(len(tr)),yt,assume_a='pos')
    poss=sorted(set(positions));imp={p:np.linalg.norm(co[positions==p]) for p in poss};v=np.array([imp[p] for p in positions]);v/=v.max()
    xc=x-xm
    if method=='specific':
     mask=(np.isin(positions[ii],[46,59])&np.isin(positions[jj],[46,59]));w=np.ones(mask.sum());z2=xc[:,ii[mask]]*xc[:,jj[mask]]
    elif method=='order':
     z2=np.column_stack([xc*x.sum(1)[:,None],x.sum(1)**2]);w=np.ones(z2.shape[1])
    else:
     z2=xc[:,ii]*xc[:,jj]
     w=np.ones(len(ii)) if method=='equal' else np.maximum(v[ii],v[jj]) if method=='strong' else np.sqrt(v[ii]*v[jj])
    z=np.column_stack([x,pw*z2*w]);return ridge_pred(z,tr,va,a,'raw')
   evaluate(f'adaptive/{method}/{pw}/{a}',pred)
# Direct pairwise ordinal loss, no target scale assumption.
for pw in [0.,.1,.3]:
 for lam in [.0001,.0003,.001,.003,.01]:
  for gap_weight in [False,True]:
   def pred(tr,va,pw=pw,lam=lam,gap_weight=gap_weight):
    xc=x-x[tr].mean(0);z=np.column_stack([xc,pw*xc[:,ii]*xc[:,jj]]) if pw else xc
    zt=z[tr]; yt=y[tr];a,b=np.triu_indices(len(tr),1);hi=np.where(yt[a]>yt[b],a,b);lo=np.where(yt[a]>yt[b],b,a)
    w=np.abs(yt[a]-yt[b]) if gap_weight else np.ones(len(a));w/=w.sum()
    def fun(co):
     p=zt@co;d=p[hi]-p[lo];g=-expit(-d)*w
     dp=np.bincount(hi,weights=g,minlength=len(tr))-np.bincount(lo,weights=g,minlength=len(tr))
     return np.dot(w,np.logaddexp(0,-d))+.5*lam*np.dot(co,co),zt.T@dp+lam*co
    out=minimize(fun,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-9})
    return z[va]@out.x
   evaluate(f'ordinal/{pw}/{lam}/{gap_weight}',pred)
for name,score in sorted(results.items(),key=lambda z:z[1][0],reverse=True)[:40]:print(name,*np.round(score,5),flush=True)
np.savez('refine_predictions.npz',y=y,**preds);json.dump(results,open('refine_results.json','w'),indent=2)
