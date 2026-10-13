#!/usr/bin/env python3
"""Support-only deterministic GFP ranking; no external data or fitted state."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import re
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

AA='ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy and standard residue volumes (scientific constants).
HYD=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
CH={a:float(a in 'KR')-float(a in 'DE') for a in AA}; CH['H']=.15

def mutations(record,ref):
    seq=record.get('sequence')
    if seq and len(seq)==len(ref):
        return [(i+1,a,b) for i,(a,b) in enumerate(zip(ref,seq)) if a!=b]
    return [(int(p),a,b) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def features(records,ref):
    rows=[]
    for r in records:
        ms=mutations(r,ref); fs=[]
        for p,a,b in ms:
            h=HYD.get(a,0); dh=HYD.get(b,0)-h; v=(VOL.get(b,140)-VOL.get(a,140))/100
            q=CH.get(b,0)-CH.get(a,0)
            local=np.mean([HYD.get(x,0) for x in ref[max(0,p-4):min(len(ref),p+3)]])
            parity=np.mean([HYD.get(ref[j],0) for j in range(max(0,p-5),min(len(ref),p+4)) if (j-(p-1))%2==0])
            f=[1,abs(dh)/5,abs(v),abs(q),max(0,-dh)/5,max(0,dh)/5,max(0,v),max(0,-v),float(b=='P'),float(a=='P'),float(b=='G'),float(a=='G'),float(a in 'FWY' and b not in 'FWY'),float(a in 'ILVMFWY' and b in 'DEKRQN'),float(a in 'DEKR' and b in 'DEKR' and q!=0),float(a in 'ST' and b in 'ST'),float(a in 'ILVM' and b in 'ILVM'),float(a in 'FWY' and b in 'FWY'),float(65<=p<=67),float(58<=p<=75),float(p<=9 or p>=230),local/5,parity/5,h/5,local*abs(dh)/25,parity*max(0,-dh)/25]
            fs.append(f)
        ar=np.asarray(fs) if fs else np.zeros((1,26))
        n=len(ms)
        row=list(ar.sum(axis=0))+list(ar.max(axis=0))+[n*n/10,np.log1p(n)]
        # Region-wise burden supports location-specific tolerances without coordinates.
        for lo,hi in [(1,10),(11,23),(24,38),(39,50),(51,57),(58,75),(76,90),(91,110),(111,128),(129,143),(144,155),(156,174),(175,195),(196,214),(215,229),(230,999)]:
            ff=[f for (p,a,b),f in zip(ms,fs) if lo<=p<=hi]
            row.extend([len(ff),sum(f[1]+f[2]+f[8] for f in ff)])
        rows.append(row)
    return np.asarray(rows,float)


# A priori approximate secondary-structure segments and hydrophobic face.
# These are hypotheses, not structure coordinates.
STRANDS=[(11,22,0),(25,37,1),(40,50,0),(80,89,0),(94,104,0),(117,124,1),(129,142,0),(145,153,0),(165,172,1),(181,190,0),(198,207,1),(219,229,1)]
def mfeat(ms,ref):
    out=[]
    for p,a,b in ms:
        h=HYD[a];dh=HYD[b]-h;v=(VOL[b]-VOL[a])/100;q=CH[b]-CH[a]
        strand=next(((l,u,f) for l,u,f in STRANDS if l<=p<=u),None)
        core=float(strand is not None and p%2==strand[2]);helix=float(58<=p<=75);critical=float(p in (65,66,67));outer=float(strand is not None and not core)
        x=[1,abs(dh)/5,abs(v),abs(q),max(0,-dh)/5,max(0,dh)/5,max(0,v),max(0,-v),float(b=='P'),float(a=='P'),float(b=='G'),float(a=='G'),float(a in 'FWY' and b not in 'FWY'),float(a in 'ILVMFWY' and b in 'DEKRQN'),float(a in 'DEKR' and b in 'DEKR' and q!=0),float(a in 'ST' and b in 'ST'),float(a in 'ILVM' and b in 'ILVM'),float(a in 'FWY' and b in 'FWY'),critical,helix,float(p<=9 or p>=230)]
        x += [core,outer,float(strand is not None),core*abs(dh)/5,core*max(0,-dh)/5,core*abs(v),core*abs(q),core*(b=='P'),outer*(b=='P'),helix*(b=='P'),float(strand is None)*(b=='P'),core*(a=='G'),float(p in (65,66,67,96,148,203,205,222)),float(b in 'DEKR'),float(a in 'DEKR'),float(b in 'QN'),float(b in 'FWY')]
        # Wildtype and target residue pooling gives generic tolerances.
        x += [float(a==aa) for aa in AA]+[float(b==aa) for aa in AA]
        out.append(x)
    return np.array(out) if out else np.zeros((1,77))

def extra_features(records,ref):
    rows=[]
    for r in records:
        ms=mutations(r,ref); m=mfeat(ms,ref)
        rows.append(np.r_[m.sum(0),m.max(0),len(ms)**2/10,np.log1p(len(ms))])
    return np.array(rows)


from scipy.special import expit, logit
from scipy.optimize import least_squares
def ge_fit(X,Y,Z,alpha=.3):
 # Logistic fluorescence expectation with learned latent additive burden.
 low=np.quantile(Y,.04);hi=np.quantile(Y,.95)+.12
 yy=(Y-low)/(hi-low)
 # Remove constant columns and use compact SVD to optimize only supported dimensions.
 xm=X.mean(0);xc=X-xm;zc=Z-xm
 u,s,vt=np.linalg.svd(xc,full_matrices=False);keep=s>1e-8
 xt=u[:,keep]*s[keep];zt=zc@vt[keep].T
 init=Ridge(alpha=alpha*10).fit(xt,logit(np.clip(yy,.03,.97)))
 b=np.r_[init.intercept_,init.coef_]
 def fun(b):
  return np.r_[expit(b[0]+xt@b[1:])-yy,np.sqrt(alpha)*b[1:]]
 def jac(b):
  p=expit(b[0]+xt@b[1:]);g=p*(1-p)
  return np.r_[g[:,None]*np.c_[np.ones(len(Y)),xt],np.c_[np.zeros(len(b)-1),np.sqrt(alpha)*np.eye(len(b)-1)]]
 opt=least_squares(fun,b,jac=jac,max_nfev=70,ftol=1e-6,xtol=1e-6,gtol=1e-6)
 return opt.x[0]+zt@opt.x[1:]


def propose(case, assignment):
    """Refit solely from the supplied support, then return ordinal query scores.

    Configuration was selected with repeated support-only holdouts. Both trees
    use chemical sums/maxima and the second includes explicitly tentative
    beta-face annotations. Logistic regression learns a nonlinear assay response
    to pooled additive features; its latent response is used only for ranking.
    No assay values, fitted coefficients, IDs, or fitted state are stored here.
    """
    ref = case['reference_sequence']
    query = case.get('query', [])
    support = [r for r in case.get('support', []) if np.isfinite(float(r['fitness']))]
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0.0 for r in query}, 'selected_ids': []}
    y = np.asarray([r['fitness'] for r in support], dtype=float)
    x = features(support, ref)
    z = features(query, ref)
    a = extra_features(support, ref)
    b = extra_features(query, ref)
    predictions = []
    for train, test in ((x, z), (a, b)):
        model = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=1, max_features=0.8,
            random_state=1729, n_jobs=1)
        model.fit(train, y)
        predictions.append(model.predict(test))
    if len(support) >= 4 and np.ptp(y) > 1e-8:
        predictions.append(ge_fit(a, y, b, alpha=0.3))
    # Equal rank blending avoids calibration assumptions across model families.
    ranks = np.mean([rankdata(np.nan_to_num(p), method='average') / len(query)
                     for p in predictions], axis=0)
    scores = 2.0 * (rankdata(ranks, method='average') - 0.5) / len(query) - 1.0
    return {'scores': {r['id']: float(s) for r, s in zip(query, scores)},
            'selected_ids': []}
