"""Deterministic support-only sequence regression for GCN4 ranking.
All fitted quantities are estimated inside propose; no training data are stored.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Sequence-only hypotheses: the assayed 44-residue activation-domain segment,
# the central WxxLF motif, and coarse chemistry. These are not measured structures.
GROUPS = ['DE','KR','FWY','WFYLIMV','LIMV','STNQ','PG','P','G','A','H','C','NQ','ST','W','F','Y']
KD = dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
HELIX = dict(zip(AA,[1.45,.77,.98,1.53,1.12,.53,1.24,1.,1.07,1.34,1.20,.73,.59,1.17,.79,.79,.82,1.14,1.14,.61]))

def sequence(record, ref):
    if record.get('sequence'): return record['sequence']
    s=list(ref)
    for m in record.get('mutation','').split(':'):
        if m: s[int(m[1:-1])-1]=m[-1]
    return ''.join(s)

def features(records, ref):
    segments=[(0,44),(0,19),(19,24),(24,44),(0,12),(12,24),(24,36),(36,44)]
    comp=[]; phys=[]; positional=[]; onehot=[]; basic=[]; interactions=[]
    wt=ref[100:144]
    for r in records:
        s=sequence(r,ref)[100:144]
        n=sum(a!=b for a,b in zip(s,wt))
        basic.append([n/44, np.sqrt(n)/np.sqrt(44),float(n>15),float(n<=1)])
        comp.append([s[a:b].count(aa) for a,b in segments for aa in AA])
        row=[]
        for a,b in segments:
            seg=s[a:b]
            row.extend(sum(c in g for c in seg) for g in GROUPS)
            row.extend([sum(KD.get(c,0) for c in seg)/4, sum(HELIX.get(c,1) for c in seg)])
        phys.append(row)
        p=np.array([[float(c in g) for c in s] for g in GROUPS[:7]]+[[KD.get(c,0)/4 for c in s],[HELIX.get(c,1) for c in s]])
        positional.append(p.ravel())
        onehot.append([float(c==aa)-float(w==aa) for c,w in zip(s,wt) for aa in AA])
        acid=np.array([c in 'DE' for c in s],float); hydro=np.array([c in 'WFYLIMV' for c in s],float)
        arom=np.array([c in 'FWY' for c in s],float); base=np.array([c in 'KR' for c in s],float)
        ints=[]
        for radius in [2,4,8]:
            near=np.convolve(acid,np.ones(2*radius+1),mode='same')
            ints.extend([np.dot(near,hydro), np.dot(near,arom),np.max(np.convolve(hydro,np.ones(2*radius+1),mode='valid')),np.max(np.convolve(acid,np.ones(2*radius+1),mode='valid'))])
        ints.extend([acid.sum()*hydro.sum()/44,acid.sum()*arom.sum()/44,base.sum()*arom.sum()/44,
                     float(s[19]=='W'),float(s[22]=='L'),float(s[23]=='F'),float(s[19]=='W' and s[22]=='L' and s[23]=='F'),
                     sum(s[i]==wt[i] for i in [6,7,9,12]), sum(s[i] in 'FWYLIMV' for i in [6,7,9,12]),
                     sum(s[i] in 'P' for i in range(18,25)),sum(s[i] in 'DE' for i in range(0,19))])
        interactions.append(ints)
    return {k:np.array(v,dtype=float) for k,v in [('comp',comp),('phys',phys),('pos',positional),('onehot',onehot),('basic',basic),('inter',interactions)]}

def focused_features(records, ref):
    rows=[]
    for r in records:
        q=sequence(r,ref)[100:144]
        a=np.array([v in 'DE' for v in q],float)
        h=np.array([v in 'WFYLIMV' for v in q],float)
        ar=np.array([v in 'FWY' for v in q],float)
        b=np.array([v in 'KR' for v in q],float)
        rows.append([a[:19].sum(),a[24:].sum(),a[14:30].sum(),h[:19].sum(),h[24:].sum(),ar[:19].sum(),ar[24:].sum(),b[:19].sum(),b[24:].sum(),q.count('P'),q.count('G'),sum(q[i] in 'FWYLIMV' for i in [19,22,23]),sum(q[i]==v for i,v in zip([19,22,23],'WLF')),float(q[19]=='W' and q[22]=='L' and q[23]=='F')])
    return np.array(rows,float)

def component_predictions(case):
    from scipy.stats import norm
    from scipy.linalg import cho_factor, cho_solve
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    support=case.get('support',[]); query=case.get('query',[])
    n=len(support);ref=case['reference_sequence'];records=support+query
    f=features(records,ref)
    global_x=np.c_[f['comp'][:,:20],f['basic']]
    local_x=np.c_[f['phys'],f['inter'],f['basic']]
    focused_x=np.c_[focused_features(records,ref),f['basic']]
    # The assay is optimized for signed Spearman correlation, not calibration.
    # Normal scores cap the influence of extreme values without reversing order.
    raw=np.array([r['fitness'] for r in support],float)
    y=norm.ppf((rankdata(raw)-.5)/n)
    out={}
    mu=local_x[:n].mean(0);sd=np.maximum(local_x[:n].std(0),1.)
    a=(local_x[:n]-mu)/sd;b=(local_x[n:]-mu)/sd
    dot=a@a.T;dist=np.maximum(np.diag(dot)[:,None]+np.diag(dot)[None,:]-2*dot,0)
    dt=np.maximum((b*b).sum(1)[:,None]+np.diag(dot)[None,:]-2*b@a.T,0)
    scale=max(np.trace(dot)/n,1e-9);mean=y.mean()
    preds=[]
    for gamma,alpha in [(1.,3.),(1.,10.),(.3,3.)]:
        k=np.exp(-gamma*dist/scale)
        coef=cho_solve(cho_factor(k+alpha*np.eye(n)),y-mean)
        preds.append(np.exp(-gamma*dt/scale)@coef+mean)
    out['kernel']=np.mean(preds,axis=0)
    for name,x in [('global_tree',global_x),('local_tree',local_x)]:
        preds=[]
        for leaf in [3,5,8]:
            model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=.8,random_state=12,n_jobs=1)
            model.fit(x[:n],y);preds.append(model.predict(x[n:]))
        out[name]=np.mean(preds,axis=0)
    mu=focused_x[:n].mean(0);sd=np.maximum(focused_x[:n].std(0),1.)
    xx=(focused_x-mu)/sd
    model=Ridge(alpha=100.).fit(xx[:n],y)
    out['focused']=model.predict(xx[n:])
    model=GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=10,learning_rate=.03,loss='huber',random_state=12)
    model.fit(focused_x[:n],y)
    out['focused_boost']=model.predict(focused_x[n:])
    return out

# Fixed, deliberately broad ensemble weights are validation-selected settings,
# not stored sequence effects. Every component is refitted on supplied supports.
ENSEMBLE_WEIGHTS={'kernel':.25,'global_tree':.25,'local_tree':.10,'focused':.15,'focused_boost':.25}

def propose(case, assignment):
    from threadpoolctl import threadpool_limits
    support=case.get('support',[]);query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if len(support)<2: return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    with threadpool_limits(limits=1):
        parts=component_predictions(case)
    z=sum(ENSEMBLE_WEIGHTS[k]*v for k,v in parts.items())
    z=2*(rankdata(z)-.5)/len(z)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,z)},'selected_ids':[]}
