"""Deterministic support-only regularized sequence predictor.
All coefficients are re-estimated from the supplied support on each call.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

def mutations(record, reference):
    seq = record.get('sequence')
    if seq and len(seq) == len(reference):
        return [(i + 1, a, b) for i, (a, b) in enumerate(zip(reference, seq)) if a != b]
    import re
    return [(int(p), a, b) for a, p, b in re.findall(r'([A-Z])(\d+)([A-Z])', record.get('mutation', ''))]

def features(records, reference):
    # Sequence intervals are only statistical pooling groups, not known structures.
    out=[]
    for r in records:
        m=mutations(r,reference); n=len(m); d={'intercept':1.0,'count': n / 8.,'multi':float(n>1)}
        for p,a,b in m:
            d['mut:%s:%s'%(p,b)]=1.
            d['site:%s'%p]=0.35
            if n>1: d['context:%s:%s'%(p,b)]=1.
        out.append(d)
    keys=sorted({k for d in out for k in d}); ix={k:i for i,k in enumerate(keys)}
    x=np.zeros((len(out),len(keys)))
    for j,d in enumerate(out):
        for k,v in d.items(): x[j,ix[k]]=v
    return x

def propose(case, assignment):
    support=case.get('support',[]); query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if not support: return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    x=features(support+query,case['reference_sequence']); n=len(support)
    y=np.asarray([r['fitness'] for r in support],float)
    k=x[:n]@x[:n].T
    coef=solve(k+2*np.eye(n),y,assume_a='pos')
    pred=x[n:]@x[:n].T@coef
    scores=2*(rankdata(pred,method='average')-0.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,scores)},'selected_ids':[]}
