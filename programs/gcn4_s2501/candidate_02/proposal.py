"""Support-only ranking for the Gcn4 101--144 activation-domain assay.
All fitted state is local to propose(); no stored outcomes or prediction lists.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import rankdata
from scipy.linalg import solve

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Fixed, sequence-derived descriptors. Regions are hypotheses, not structures.
GROUPS = ['DE', 'KR', 'H', 'FWY', 'FWYLIMV', 'LIMV', 'P', 'G', 'STNQ', 'A', 'C', 'F', 'W', 'Y']
KD = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))

def features(records, reference):
    seqs=[r['sequence'][100:144] for r in records]
    ref=reference[100:144]
    comp=[]; regional=[]; position=[]; exact=[]; interactions=[]
    windows=[(0,44),(0,19),(19,25),(25,44),(0,10),(10,19),(19,32),(32,44),(0,25),(15,30),(0,22),(22,44)]
    for s in seqs:
        a=np.array(list(s)); n=len(a)
        g=np.array([[float(x in group) for group in GROUPS] for x in s])
        total=g.sum(0)
        hyd=np.array([KD.get(x,0) for x in s])
        mutation=np.array([x!=y for x,y in zip(s,ref)],float)
        c=[s.count(x) for x in AA]+list(total)+[total[0]-total[1],hyd.sum(),mutation.sum()]
        reg=[]
        for lo,hi in windows[1:]:
            reg.extend(g[lo:hi].sum(0));reg.extend([hyd[lo:hi].sum(),mutation[lo:hi].sum()])
        # Acidity around aromatic / bulky residues, and local clustering.
        inter=[]
        for radius in [2,4,7,12]:
            for hidx in [3,4,5]:
                pair=[sum(g[j,0] for j in range(max(0,i-radius),min(n,i+radius+1)) if j!=i) for i in range(n)]
                inter.append(float(np.dot(g[:,hidx],pair)))
                inter.append(float(sum(g[i,hidx]*sum(g[j,hidx] for j in range(i+1,min(n,i+radius+1))) for i in range(n))))
        for width in [5,9,15]:
            for idx in [0,1,3,4,5,6]:
                vals=np.convolve(g[:,idx],np.ones(width),mode='valid')
                inter.extend([vals.max(),vals.min(),vals.std()])
        for idx in [0,1,3,4,5]:
            inter.extend([float(np.dot(g[:,idx],np.arange(n))/max(1,total[idx])),float(np.dot(g[:,idx],(np.arange(n)-21.5)**2)/max(1,total[idx]))])
        # Anchor hydrophobic positions are permitted scientific annotations.
        anchor=g[[7,9,12,19,22,23],:]
        inter.extend([total[3]*total[0],total[4]*total[0],total[3]*total[1],total[4]**2,total[0]**2])
        for lo,hi in [(0,19),(25,44),(0,10),(10,19)]:
            inter.extend([g[lo:hi,0].sum()*g[19:25,4].sum(),g[lo:hi,0].sum()*g[19:25,3].sum()])
        p=np.concatenate([g, hyd[:,None]/3.,mutation[:,None]],axis=1).ravel()
        oh=np.array([[float(x==y) for y in AA] for x in s]).ravel()
        comp.append(c);regional.append(reg);position.append(p);exact.append(oh);interactions.append(inter)
    return {k:np.asarray(v,float) for k,v in [('comp',comp),('regional',regional),('position',position),('exact',exact),('interactions',interactions)]}

def target_values(y, kind='rank'):
    if kind=='raw': return (y-np.mean(y))/max(np.std(y),1e-6)
    return (rankdata(y)-.5)/len(y)*2-1

def kernel_predict(fs, train, test, y, alpha=1., weights=(1.,.5,.1,.1,.1), kind='rank', single_weight=.5, orders=None, rbf=0.):
    K=np.zeros((len(train),len(train)));Q=np.zeros((len(test),len(train)))
    for key,w in zip(['comp','regional','position','exact','interactions'],weights):
        if not w: continue
        X=fs[key][train]; Z=fs[key][test]
        mean=X.mean(0); sd=X.std(0)
        # A variance floor avoids magnifying rare substitutions in the singles.
        if key in ('position','exact'): sd=np.maximum(sd,.35)
        else: sd=np.maximum(sd,1.)
        X=(X-mean)/sd;Z=(Z-mean)/sd
        denom=max(1.,np.mean(np.sum(X*X,axis=1)))
        A=X@X.T/denom;B=Z@X.T/denom
        if rbf:
            D=np.maximum(0,np.sum(X*X,1)[:,None]+np.sum(X*X,1)[None,:]-2*X@X.T)/denom
            E=np.maximum(0,np.sum(Z*Z,1)[:,None]+np.sum(X*X,1)[None,:]-2*Z@X.T)/denom
            A=np.exp(-rbf*D);B=np.exp(-rbf*E)
        K+=w*A;Q+=w*B
    t=target_values(y,kind)
    sample=np.ones(len(train))
    if orders is not None:sample[orders[train]==1]=single_weight
    intercept=np.average(t,weights=sample)
    coef=solve(K+np.diag(alpha/sample),t-intercept,assume_a='pos',check_finite=False)
    return Q@coef+intercept

# Validation-selected hyperparameters, never fitted states or observed outcomes.
CONFIGS = [
    ('extra', 'regional', 2, 'rank', 2.),
    ('extra', 'regional', 2, 'raw', .5),
    ('extra', 'regional', 4, 'rank', 2.),
    ('extra', 'regional', 4, 'raw', .5),
    ('boost', 'mix', 10, 'rank', 1.),
    ('boost', 'mix', 10, 'raw', 1.),
    ('ridge', 'regional', 1., 'raw', .5),
]

def propose(case, assignment):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    support = list(case.get('support', []))
    query = list(case.get('query', []))
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if len(support) < 2:
        return {'scores': {r['id']: 0. for r in query}, 'selected_ids': []}
    # Sorting supplies order-invariant fitting without using IDs as predictors.
    support.sort(key=lambda r: (r['sequence'], r.get('mutation', ''), str(r['id'])))
    query.sort(key=lambda r: (r['sequence'], str(r['id'])))
    fs = features(support + query, case['reference_sequence'])
    fs['mix'] = np.concatenate([fs[k] for k in ['comp','regional','interactions']], axis=1)
    n = len(support)
    y = np.asarray([r['fitness'] for r in support], dtype=float)
    ref = case['reference_sequence']
    orders = np.asarray([sum(a != b for a,b in zip(r['sequence'],ref)) for r in support])
    if np.ptp(y) < 1e-12:
        return {'scores': {r['id']: 0. for r in query}, 'selected_ids': []}
    components = []
    for model, key, par, kind, single_weight in CONFIGS:
        X = fs[key][:n]
        Z = fs[key][n:]
        sample = np.where(orders == 1, single_weight, 1.)
        target = target_values(y, kind)
        if model == 'extra':
            fit = ExtraTreesRegressor(n_estimators=320, min_samples_leaf=par,
                max_features=.8, random_state=991, n_jobs=1)
        elif model == 'boost':
            fit = GradientBoostingRegressor(n_estimators=150, max_depth=2,
                min_samples_leaf=par, learning_rate=.025, loss='huber', random_state=991)
        else:
            mu = X.mean(axis=0)
            sd = np.maximum(X.std(axis=0), 1.)
            X = (X-mu)/sd
            Z = (Z-mu)/sd
            fit = Ridge(alpha=par)
        fit.fit(X, target, sample_weight=sample)
        prediction = fit.predict(Z)
        # Borda aggregation removes arbitrary differences in component scales.
        components.append((rankdata(prediction)-.5)/len(query))
    prediction = np.mean(components, axis=0)
    scores = (rankdata(prediction)-.5)/len(query)*2.-1.
    return {'scores': {r['id']: float(v) for r,v in zip(query,scores)}, 'selected_ids': []}
