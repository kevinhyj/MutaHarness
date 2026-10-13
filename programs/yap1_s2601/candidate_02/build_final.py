from pathlib import Path
p=Path('proposal.py').read_text();p=p[:p.index('def basic_features')]
s=Path('scientific_model.py').read_text();s=s[s.index('# Putative WW'):]
r=Path('ranking_explore.py').read_text();r=r[r.index('def rank_fit'):r.index("if __name__")]
r=r.replace("wt=((jj-ii)/len(y))**power/len(y)","keep=y[hi] > y[lo]\n lo,hi,ii,jj=lo[keep],hi[keep],ii[keep],jj[keep]\n wt=((jj-ii)/len(y))**power/len(y)")
header='''# Fixed configurations selected using only public support cross-validation.
# Every effect and pairwise score function is estimated again on invocation.
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import norm
SEED = 2601
'''
entry='''
def propose(case, assignment):
    """Fit only supplied measurements; rank every current query; select none.

    The two-degree polynomial kernel learns regularized combination effects.
    WW-region/severity descriptors are hypotheses with fitted signs. No WT
    outcome, external sequence, stored fit, or fixed mutation effect is used.
    """
    support=list(case.get('support', []))
    query=list(case.get('query', []))
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0.0 for r in query}, 'selected_ids': []}
    # No randomized fitting is needed. Seed kept explicit for future extensions.
    rng = np.random.default_rng(SEED)
    n=len(support)
    y=np.array([float(r['fitness']) for r in support], dtype=float)
    ks=kernels(support+query,case['reference_sequence'])
    train=np.arange(n);test=np.arange(n,n+len(query))
    base={'exact':1.0,'pos':0.3,'local':0.2,'severity':1.0,'target':1.0}
    configs=[
        # weight, kernel weights, regularization, objective, rank gap exponent, epistasis
        (.30,base,.01,'pair',1.0,0.0),
        (.25,dict(base,count=1.0),.03,'pair',.5,0.0),
        (.15,dict(base,count=3.0),.01,'pair',.5,.2),
        (.15,{'exact':1.0,'pos':.3,'brbf':.2,'severity':1.0,'target':1.0},.1,'sqrt',0.0,0.0),
        (.15,dict(base,count=3.0),.1,'sqrt',0.0,.2),
    ]
    ensemble=np.zeros(len(query))
    for weight,weights,reg,objective,gap,epistasis in configs:
        K=combine(ks,weights,epistasis)
        if objective=='pair':
            pred=rank_fit(K,train,test,y,reg,gap)
        else:
            # The observed assay scores are positive. Square-root is monotone
            # and reduces leverage of large readings; this is not calibration.
            pred=predict_kernel(K,train,test,target_transform(y,objective),reg)
        ensemble+=weight*(rankdata(pred,method='average')-.5)/len(query)
    # Purely ordinal, finite scores in [-1,1]; preserve ties.
    ordinal=2*(rankdata(ensemble,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,ordinal)}, 'selected_ids':[]}
'''
Path('proposal.py').write_text(p+header+s+'\n'+r+entry)
