#!/usr/bin/env python3
"""Support-only deterministic ordinal prediction; no persisted fitted state."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

# Mutation-library windows, inferred from positions in the supplied sequences.
# These are design covariates, not established structural domains.
BLOCKS = [{7,8,9,11,18,19,20,21,22,23,24},
          {56,59,60,62,76,77,78,79,80,81,82,85},
          {66,67,73,86,87,92,93},
          {96,97,98,99,102,104,105,117,119,120,121,122},
          {137,139,140,141,142,143,144,157,159,160,161,163},
          {145,147,148,149,151,152,154,164,165,168,169,170},
          {171,172,173,175,177,178,180,195,196,198,200},
          {181,183,186,187,189,195,201,202,203,204,205,206,207,208,209,210,211}]

def mutations(record, reference):
    seq = record.get('sequence')
    if seq and len(seq) == len(reference):
        return [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,seq)) if a != b]
    return [(int(t[1:-1]),t[0],t[-1]) for t in record.get('mutation','').split(':') if len(t)>2]

def encode(records, reference, vocabulary=None):
    mm = [mutations(r,reference) for r in records]
    if vocabulary is None:
        vocabulary = sorted(set((p,b) for m in mm for p,a,b in m))
    lookup = {v:i for i,v in enumerate(vocabulary)}
    x = np.zeros((len(mm),len(vocabulary)))
    group = np.zeros((len(mm),len(BLOCKS)))
    counts = np.array([len(m) for m in mm],float)
    for i,m in enumerate(mm):
        for p,a,b in m:
            if (p,b) in lookup: x[i,lookup[p,b]]=1
        g = int(np.argmax([sum(p in block for p,a,b in m) for block in BLOCKS]))
        group[i,g]=1
    multi = (counts>1).astype(float)
    # Context interactions permit multi-mutant effects to differ from single effects.
    features = np.column_stack([x,x*multi[:,None],group*multi[:,None],counts,counts**2/10])
    return features, vocabulary, group, counts

# Standard biochemical constants, used only in a validation comparator.
_HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
_VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',[88,108,111,138,190,60,153,167,169,167,163,114,113,144,173,89,116,140,228,194]))
def _properties(a):
    return np.array([_HYDRO.get(a,0)/4,_VOLUME.get(a,150)/150,{'D':-1,'E':-1,'K':1,'R':1,'H':.2}.get(a,0),float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQ')])

def _components(support, query, reference, comparator=False):
    from sklearn.ensemble import GradientBoostingRegressor
    support=sorted(support,key=lambda r:(r.get('mutation',''),str(r.get('id',''))))
    f,v,g,n=encode(support,reference)
    fq,_,gq,nq=encode(query,reference,v)
    d=len(v); x=f[:,:d];xq=fq[:,:d]
    y=np.array([float(r['fitness']) for r in support])
    yr=(rankdata(y,method='average')-.5)/len(y)
    predictions=[Ridge(alpha=1.).fit(f,yr).predict(fq)]
    # Recurrent pair terms only; interactions seen in one support are too weak
    # to estimate separately. Each invocation reconstructs this list from supports.
    co=x.T@x
    pairs=[(i,j) for i in range(d) for j in range(i) if v[i][0]!=v[j][0] and co[i,j]>=2]
    z=np.column_stack([x[:,i]*x[:,j] for i,j in pairs]) if pairs else np.zeros((len(support),0))
    zq=np.column_stack([xq[:,i]*xq[:,j] for i,j in pairs]) if pairs else np.zeros((len(query),0))
    for alpha,scale in [(1.,1.),(.3,1.),(1.,.5)]:
        xx=np.column_stack([f,scale*z]);qq=np.column_stack([fq,scale*zq])
        predictions.append(Ridge(alpha=alpha).fit(xx,yr).predict(qq))
    positions=sorted(set(p for p,a in v))
    p=np.column_stack([x[:,[j for j,(pp,a) in enumerate(v) if pp==pos]].sum(axis=1) for pos in positions]) if positions else np.zeros((len(support),0))
    pq=np.column_stack([xq[:,[j for j,(pp,a) in enumerate(v) if pp==pos]].sum(axis=1) for pos in positions]) if positions else np.zeros((len(query),0))
    tx=np.column_stack([x,p,g,n]);tq=np.column_stack([xq,pq,gq,nq])
    tree=GradientBoostingRegressor(n_estimators=140,learning_rate=.04,max_depth=1,min_samples_leaf=2,random_state=123,loss='squared_error')
    predictions.append(tree.fit(tx,y).predict(tq))
    if comparator:
        index={p:i for i,p in enumerate(positions)}
        def chem(records,counts):
            b=np.zeros((len(records),7*len(positions)))
            for i,r in enumerate(records):
                for p,a,c in mutations(r,reference):
                    if p in index:
                        j=index[p];b[i,j*7:(j+1)*7]=_properties(c)-_properties(a)
            return np.column_stack([b*.6,b*(counts>1)[:,None]*.6])
        cx=np.column_stack([f,chem(support,n)]);cq=np.column_stack([fq,chem(query,nq)])
        predictions.append(Ridge(alpha=1.).fit(cx,yr).predict(cq))
    return np.array(predictions)

def propose(case, assignment):
    """Refit all components on supplied observations and rank every current query.

    Configuration selected using repeated held-out feedback, not hidden outcomes.
    Output scores are ordinal positions, not fitness or calibrated probabilities.
    """
    query=case.get('query',[])
    support=[r for r in case.get('support',[]) if np.isfinite(float(r['fitness']))]
    if not query:return {'scores':{},'selected_ids':[]}
    if len(support)<2:return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    pp=_components(support,query,case['reference_sequence'])
    ranks=np.array([(rankdata(p,method='average')-.5)/len(query) for p in pp])
    # Conservative averaging of nearby regularization settings; small weight for
    # an independent raw-growth stump model preserves nonlinear partition evidence.
    blended=np.array([.25,.20,.20,.20,.15])@ranks
    scores=2*(rankdata(blended,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(value) for r,value in zip(query,scores)},'selected_ids':[]}
