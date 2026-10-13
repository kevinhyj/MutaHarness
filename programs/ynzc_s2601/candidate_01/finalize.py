import json
p='proposal.py';s=open(p).read();s=s.replace('from scipy.stats import rankdata','from scipy.stats import rankdata\nfrom scipy.spatial.distance import cdist')
pos=s.index('def fit_predict')
s=s[:pos]+'''# Fixed, rounded regularization/kernel configurations selected by public
# support-only validation. These are hyperparameters, never fitted coefficients.
# Individual members use different smoothness and epistasis priors. Their fitted
# residue effects are recomputed from the supplied support on every call.
CONFIGS = '''+repr([(cf['weights'],cf['alpha']) for cf in json.load(open('chosen_configs.json'))[:6]])+'''
RANDOM_SEED = 20261013

def kernel_components(records, ref):
    f=features(records,ref)
    n=len(records); L=len(ref)
    f['amino']=f['mut'].reshape(n,L,20).sum(axis=1)
    ph=f['phys'].reshape(n,L,8)
    f['localcross']=np.concatenate([ph[:,a]*ph[:,b] for a in range(L) for b in range(a+1,L)],axis=1)
    required=set(k for weights,alpha in CONFIGS for k in weights)
    kernels={k:x@x.T for k,x in f.items() if k in required}
    # Sequence-only similarity. WT subtraction gives a zero feature vector
    # for unchanged residues and a positive semidefinite covariance.
    codes=np.array([[AA.index(a) for a in record_sequence(r,ref)] for r in records])
    wt=np.array([AA.index(a) for a in ref])
    for bw in (1.,3.,10.):
        ka='aa_rbf_'+str(bw); ke='ep_rbf_'+str(bw)
        if ka not in required and ke not in required: continue
        sim=np.exp(-cdist(PROPS,PROPS,'sqeuclidean')/(2*bw))
        gram=np.zeros((n,n))
        for j in range(L):
            a=codes[:,j];w=wt[j]
            gram+=sim[a[:,None],a[None,:]]-sim[a,w][:,None]-sim[a,w][None,:]+1
        if ka in required:kernels[ka]=gram
        if ke in required:kernels[ke]=gram**2/4
    D=cdist(f['phys'],f['phys'],'sqeuclidean')
    for bw in (2.,8.,32.):
        key='whole_rbf_'+str(bw)
        if key in required:kernels[key]=np.exp(-D/(2*bw))
    return kernels

def record_sequence(record, ref):
    if record.get('sequence'): return record['sequence']
    seq=list(ref)
    for token in record.get('mutation','').split(':'):
        if token: seq[int(token[1:-1])-1]=token[-1]
    return ''.join(seq)

def predict(train, test, ref):
    if not test: return np.zeros(0)
    if not train: return np.zeros(len(test))
    y=np.array([float(r['fitness']) for r in train])
    if not np.all(np.isfinite(y)): raise ValueError('Support fitness must be finite')
    n=len(train); kernels=kernel_components(train+test,ref)
    mean_y=y.mean();scale_y=y.std()
    if scale_y<1e-12:return np.full(len(test),mean_y)
    z=(y-mean_y)/scale_y
    predictions=[]
    for weights,alpha in CONFIGS:
        gram=sum(w*kernels[key] for key,w in weights.items())
        kt=gram[:n,:n];qt=gram[n:,:n]
        mean=kt.mean(axis=0);grand=mean.mean()
        K=kt-mean[:,None]-mean[None,:]+grand
        Q=qt-qt.mean(axis=1,keepdims=True)-mean[None,:]+grand
        # Intercept is unpenalized; all other effects have positive shrinkage.
        coef=solve(K+alpha*np.eye(n),z,assume_a='pos',check_finite=False)
        predictions.append((Q@coef)*scale_y+mean_y)
    return np.mean(predictions,axis=0)

def propose(case, assignment):
    # No files, ID-based scientific features, state cache, or query outcomes.
    # The final models have no stochastic operations; this seed also makes
    # invocation deterministic if future local components use NumPy randomness.
    np.random.seed(RANDOM_SEED)
    train=list(case.get('support',[]));query=list(case.get('query',[]))
    pred=predict(train,query,case['reference_sequence'])
    if not np.all(np.isfinite(pred)):raise ValueError('Nonfinite prediction')
    # Arbitrary bounded ordinal scores, with average ranks for exact ties.
    score=2*rankdata(pred)/(len(pred)+1)-1 if len(pred) else []
    return {'scores':{r['id']:float(s) for r,s in zip(query,score)},'selected_ids':[]}
'''
open(p,'w').write(s)
