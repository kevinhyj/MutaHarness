"""Support-only deterministic mutation regression; public configuration, no saved fit."""
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

def tokens(record, reference=None):
    m = record.get('mutation', '')
    if m and m.lower() not in ('wt','wildtype','wild_type'):
        return tuple(sorted(m.split(':')))
    if reference and record.get('sequence'):
        return tuple(a+str(i+1)+b for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a!=b)
    return ()

def ridge_predict(x,y,z,alpha):
    mean=x.mean(0); ym=y.mean(); xc=x-mean
    return (z-mean) @ (xc.T @ solve(xc@xc.T+alpha*np.eye(len(x)),y-ym,assume_a='pos')) + ym

def predict_records(support,query,reference=''):
    ts=[tokens(r,reference) for r in support]; tq=[tokens(r,reference) for r in query]
    if not support: return np.zeros(len(query))
    y=np.array([float(r['fitness']) for r in support])
    vocab=sorted(set(t for row in ts for t in row)); ix={t:i for i,t in enumerate(vocab)}
    def features(rows):
        x=np.zeros((len(rows),len(vocab)+1))
        for j,row in enumerate(rows):
            for t in row:
                if t in ix:x[j,ix[t]]=1
            x[j,-1]=len(row)/8
        return x
    x=features(ts); z=features(tq)
    multi=np.array([len(t)>1 for t in ts]); qm=np.array([len(t)>1 for t in tq])
    p=ridge_predict(x,y,z,2.)
    if multi.sum()>=8:p[qm]=ridge_predict(x[multi],y[multi],z[qm],1.)
    singles=np.flatnonzero(~multi)
    if len(singles):
        baseline=np.median(y[singles])
        for i,row in enumerate(tq):
            if len(row)!=1:continue
            t=row[0]; same=[j for j in singles if ts[j] and ts[j][0][1:-1]==t[1:-1]]
            p[i]=np.mean(y[same]) if same else baseline
    return p

def propose(case,assignment):
    query=case.get('query',[])
    p=predict_records(case.get('support',[]),query,case.get('reference_sequence',''))
    p=np.nan_to_num(p,nan=0.,posinf=1.,neginf=-1.)
    scores=2*(rankdata(p,method='average')-.5)/max(len(p),1)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,scores)},'selected_ids':[]}
