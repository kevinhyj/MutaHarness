import json
from pathlib import Path
p=Path('proposal.py');old=p.read_text();Path('early_proposal.py').write_text(old)
head=old.split('def kernels(')[0]
adv=Path('advanced.py').read_text();adv=adv[adv.index('def align('):adv.index('def extra_kernels(')]+adv[adv.index('def tree_features('):]
configs=json.load(open('chosen_configs.json'))
body='''
def model_kernels(records, reference):
    pos,exact,chem,reg,glob=features(records,reference)
    n=len(records);length=len(reference)
    active=np.flatnonzero(np.any(pos!=0,axis=0))
    exact=exact.reshape(n,length,20)[:,active,:].reshape(n,-1)
    chem=chem.reshape(n,length,21)[:,active,1:]
    reg=reg.reshape(n,12,21)
    pos=pos[:,active]
    out=[pos@pos.T,exact@exact.T,np.einsum('npd,mpd->nm',chem,chem),np.einsum('npd,mpd->nm',reg,reg),glob@glob.T]
    dist=active[:,None]-active[None,:]
    for width in [1.5,3,6,12]:
        spatial=np.exp(-.5*(dist/width)**2)
        out.extend([pos@spatial@pos.T,np.einsum('npd,pq,mqd->nm',chem,spatial,chem,optimize=True)])
    return np.array(out)

def nonlinear_kernels(records, reference):
    from scipy.spatial.distance import cdist
    desc=descriptors(records,reference)
    flat=[];owner=[]
    for i,row in enumerate(desc):
        for v in row:flat.append(v);owner.append(i)
    n=len(records)
    if not flat:return np.zeros((13,n,n))
    flat=np.array(flat);h=np.zeros((n,len(flat)));h[np.array(owner),np.arange(len(flat))]=1
    same=(flat[:,0,None]==flat[None,:,0]).astype(float)
    spatial=np.exp(-.5*((flat[:,0,None]-flat[None,:,0])*100/3)**2)
    dt=cdist(flat[:,9:16],flat[:,9:16],metric='sqeuclidean')
    dg=cdist(flat[:,16:],flat[:,16:],metric='sqeuclidean')
    out=[]
    for sigma in [.5,1,2,4]:
        sim=np.exp(-dt/(2*sigma*sigma))
        out.extend([h@(sim*same)@h.T,h@(sim*spatial)@h.T,h@np.exp(-dg/(2*sigma*sigma))@h.T])
    prior=h@flat[:,30:]
    out.append(prior@prior.T)
    return np.array(out)

# Six validation-selected covariance configurations, never fitted coefficients.
# Each tuple is (ridge penalty, base kernel weights, nonlinear kernel weights,
# interaction strength, response transform, fitted-intercept flag).
CONFIGS = '''+repr(configs)+'''

def _predict(case):
    from sklearn.ensemble import ExtraTreesRegressor
    support=[r for r in case.get('support',[]) if np.isfinite(float(r['fitness']))]
    query=list(case.get('query',[]))
    if not query:return {'scores':{},'selected_ids':[]}
    if len(support)<2:return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    # Sorting only the training records makes the result independent of support order.
    support=sorted(support,key=lambda r:(r.get('mutation',''),r.get('sequence',''),str(r.get('id',''))))
    records=support+query;ref=case['reference_sequence'];n=len(support)
    y=np.array([float(r['fitness']) for r in support])
    base=model_kernels(records,ref);extra=nonlinear_kernels(records,ref)
    member=[]
    for alpha,w,ew,pair,transform,center in CONFIGS:
        k=np.einsum('k,kij->ij',w,base)+np.einsum('k,kij->ij',ew,extra)
        if pair:k+=pair*k*k/max(float(np.mean(np.diag(k[:n,:n]))),1e-12)
        kt=k[:n,:n].copy();kv=k[n:,:n].copy()
        if center:
            m=kt.mean(axis=0);total=kt.mean()
            kt=kt-m[:,None]-m[None,:]+total
            kv=kv-m[None,:]-kv.mean(axis=1)[:,None]+total
        # Observed assay is positive; log offsets soften its lower boundary.
        # For a signed/recentered invocation shift by the training minimum only.
        offset=float(transform[3:])/100
        target=np.log(y-min(0.,float(y.min()))+offset)
        mean=float(target.mean())
        coef=solve(kt+alpha*np.eye(n),target-mean,assume_a='pos')
        pred=kv@coef+mean
        member.append((rankdata(pred,method='average')-.5)/len(query))
    kernel_rank=np.mean(member,axis=0)
    x=tree_features(records,ref);trees=[]
    for target in [y,(rankdata(y,method='average')-.5)/n]:
        model=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=1,max_features=.7,random_state=73,n_jobs=1)
        model.fit(x[:n],target)
        pred=model.predict(x[n:])
        trees.append((rankdata(pred,method='average')-.5)/len(query))
    # A fixed blend chosen for stable signed Spearman correlation across splits.
    ordinal=.65*kernel_rank+.35*np.mean(trees,axis=0)
    score=2*(rankdata(ordinal,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,score)},'selected_ids':[]}

def propose(case,assignment):
    """Refit every model from supplied supports; return all current query ranks."""
    from threadpoolctl import threadpool_limits
    # Limit BLAS threads to make runtime predictable within the 120 second budget.
    with threadpool_limits(limits=1):
        return _predict(case)
'''
p.write_text(head+'\n'+adv+'\n'+body)
