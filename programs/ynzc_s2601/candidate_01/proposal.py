"""Deterministic, support-only sequence regression for ordinal stability ranking.
No external data, stored models, or fitness values are used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
from scipy.spatial.distance import cdist

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Ordinary physicochemical constants, not fitted assay coefficients.
KD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
# Relative helix free energies (Pace/Scholtz-style), with a proline indicator
# separately fitted: the scale is a prior feature, not an assay mechanism.
HEL = [0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]
PROPS = np.array([KD, np.array(VOL)/100., HEL,
 [0,0,-1,-1,0,0,.1,0,1,0,0,0,0,0,1,0,0,0,0,0],
 [float(a=='P') for a in AA], [float(a=='G') for a in AA],
 [float(a in 'FWY') for a in AA], [float(a in 'STNQH') for a in AA]],dtype=float).T
PROPS = (PROPS-PROPS.mean(axis=0))/PROPS.std(axis=0)

def features(records, ref):
    n,L=len(records),len(ref); p=PROPS.shape[1]
    mut=np.zeros((n,L,20)); phys=np.zeros((n,L,p)); counts=np.zeros((n,L))
    for i,r in enumerate(records):
        seq=r.get('sequence')
        if not seq:
            seq=list(ref)
            for token in r.get('mutation','').split(':'):
                if token: seq[int(token[1:-1])-1]=token[-1]
        for j,(w,a) in enumerate(zip(ref,seq)):
            if a!=w and a in AA and w in AA:
                mut[i,j,AA.index(a)]=1
                phys[i,j]=PROPS[AA.index(a)]-PROPS[AA.index(w)]
                counts[i,j]=1
    # Positional effects and pair backgrounds remain distinct from residue effects.
    pair=np.stack([counts[:,a]*counts[:,b] for a in range(L) for b in range(a+1,L)],axis=1)
    general=np.concatenate([phys.sum(axis=1),counts.sum(axis=1,keepdims=True)],axis=1)
    # Sequence-only position bases support transfer to sparsely sampled sites.
    position=np.linspace(0,1,L)
    basis=np.stack([np.ones(L),position,position**2,np.sin(2*np.pi*np.arange(L)/3.6),np.cos(2*np.pi*np.arange(L)/3.6),np.array([float(a in 'AILMVFWY') for a in ref]),np.array([float(a in 'DEKR') for a in ref])],axis=1)
    transfer=np.einsum('nlp,lb->npb',phys,basis).reshape(n,-1)
    # Symmetric chemistry products allow learned nonadditivity without coordinates.
    interactions=[]
    for a in range(p):
        for b in range(a,p):
            v=phys[:,:,a].sum(axis=1)*phys[:,:,b].sum(axis=1)-(phys[:,:,a]*phys[:,:,b]).sum(axis=1)
            interactions.append(v)
    cross=np.array(interactions).T/4
    return {'mut':mut.reshape(n,-1), 'pos':counts, 'pair':pair,
            'phys':phys.reshape(n,-1)/2, 'general':general/2,
            'transfer':transfer/2, 'cross':cross}

# Fixed, rounded regularization/kernel configurations selected by public
# support-only validation. These are hyperparameters, never fitted coefficients.
# Individual members use different smoothness and epistasis priors. Their fitted
# residue effects are recomputed from the supplied support on every call.
CONFIGS = [({'mut': 1.1, 'pos': 76.0, 'pair': 24.0, 'phys': 2.6, 'general': 0.016, 'transfer': 0.088, 'amino': 0.02, 'localcross': 0.076, 'aa_rbf_3.0': 0.036, 'whole_rbf_2.0': 31.0}, 0.21), ({'mut': 0.44, 'pos': 15.0, 'pair': 7.2, 'phys': 0.55, 'transfer': 0.2, 'amino': 0.0064, 'localcross': 0.028, 'aa_rbf_3.0': 0.2, 'ep_rbf_10.0': 2.5, 'whole_rbf_2.0': 20.0}, 0.048), ({'mut': 0.39, 'pos': 4.7, 'pair': 8.6, 'phys': 0.39, 'general': 0.0094, 'transfer': 0.075, 'amino': 0.1, 'localcross': 0.21, 'aa_rbf_3.0': 0.15, 'aa_rbf_10.0': 0.28, 'whole_rbf_2.0': 10.0}, 0.73), ({'mut': 1.7, 'pos': 51.0, 'pair': 52.0, 'phys': 1.1, 'general': 0.0074, 'transfer': 0.011, 'amino': 0.037, 'aa_rbf_3.0': 0.2, 'ep_rbf_10.0': 4.1, 'whole_rbf_8.0': 28.0}, 1.1), ({'mut': 1.3, 'pos': 16.0, 'pair': 3.9, 'phys': 0.61, 'general': 0.027, 'transfer': 0.019, 'cross': 0.006, 'localcross': 0.75, 'whole_rbf_8.0': 4.6}, 0.9), ({'mut': 0.82, 'pos': 66.0, 'pair': 56.0, 'phys': 3.5, 'amino': 2.1, 'ep_rbf_10.0': 8.5, 'whole_rbf_2.0': 19.0}, 0.5)]
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
