"""Deterministic support-only ranking of GRB2 assay variants.

No trained quantities are stored here. Configurations are chosen using repeated
held-out measured feedback; every coefficient and intercept is refit from the
case argument. Output ranks are arbitrary ordinal scores, not probabilities.
Chemical descriptors and broad residue classes are scientific priors. No MSA,
external sequences, structure coordinates, or pretrained predictor are used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
VOLUME = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]

def aa_descriptors():
    z=np.array([[HYDRO[i],VOLUME[i],float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),float(a=='G'),float(a=='P'),float(a in 'STNQ'),float(a in 'IVT')] for i,a in enumerate(AA)],dtype=float)
    z[:,:2]=(z[:,:2]-z[:,:2].mean(0))/z[:,:2].std(0)
    return z
DESC=aa_descriptors()


def mutations(record, reference):
    seq=record.get('sequence')
    if seq is not None and len(seq)==len(reference):
        return [(i,a,b) for i,(a,b) in enumerate(zip(reference,seq)) if a!=b]
    result=[]
    for token in record.get('mutation','').split(':'):
        if token: result.append((int(token[1:-1])-1,token[0],token[-1]))
    return result


def features(records,reference):
    n=len(records); L=len(reference)
    site=np.zeros((n,L)); prop=np.zeros((n,L*8)); severity=np.zeros((n,L*3))
    global_prop=np.zeros((n,16)); aa=np.zeros((n,20)); pair=np.zeros((n,400)); context=np.zeros((n,24));order=np.zeros((n,1))
    positions=[];wild=[];dest=[];owners=[]
    for j,r in enumerate(records):
        ms=mutations(r,reference);order[j,0]=len(ms)
        for p,w,m in ms:
            if w not in AA or m not in AA or not 0<=p<L:continue
            wi=AA.index(w);mi=AA.index(m);d=DESC[mi]-DESC[wi]
            site[j,p]+=1;prop[j,p*8:p*8+8]+=d
            severity[j,p*3:p*3+3]+=[np.linalg.norm(d)/3,float(m=='P'),float(m=='G')]
            global_prop[j,:8]+=d;global_prop[j,8:]+=abs(d)
            aa[j,mi]+=1;pair[j,wi*20+mi]+=1
            cls=0 if w in 'AVLIMFWY' else 1 if w in 'GP' else 2
            context[j,cls*8:cls*8+8]+=d
            positions.append(p);wild.append(DESC[wi]);dest.append(DESC[mi]);owners.append(j)
    # Aggregation maps individual substitutions into the multi-mutant variant.
    agg=np.zeros((n,len(owners)))
    if owners:agg[np.array(owners),np.arange(len(owners))]=1.
    return {'site':site,'prop':prop,'severity':severity,'global':global_prop,'aa':aa,'pair':pair,'context':context,'order':order,
            'positions':np.asarray(positions),'wild':np.asarray(wild).reshape(-1,8),'dest':np.asarray(dest).reshape(-1,8),'agg':agg}


def similarity_parts(f,g):
    # Per-position chemical smoothness plus weaker sharing across wild-type
    # chemical environments. These covariances are assumptions, not mechanisms.
    dist=((f['dest'][:,None,:]-g['dest'][None,:,:])**2).sum(2)/8.
    wdist=((f['wild'][:,None,:]-g['wild'][None,:,:])**2).sum(2)/8.
    sharing=(f['positions'][:,None]==g['positions'][None,:]).astype(float)+.3*np.exp(-wdist)
    common=.02*(f['global']@g['global'].T)+.2*(f['order']@g['order'].T)
    return dist,sharing,common


def similarity_kernel(f,g,parts,bandwidth):
    dist,sharing,common=parts
    return f['agg']@(np.exp(-dist/bandwidth)*sharing)@g['agg'].T+common


def interaction_base(f,g):
    weights={'site':1.,'prop':.1,'severity':.1,'pair':.05,'aa':.05,'context':.2,'order':.2}
    return sum(v*(f[k]@g[k].T) for k,v in weights.items())


def regression(K,Q,y,alpha,scale,weights=None):
    # A log-distance transform tests assay saturation. -2 is a modeling scale,
    # NOT an experimentally established floor; adapt below any observed value.
    z=y if scale=='raw' else np.log(y-min(-2.,float(y.min())-.1))
    if weights is None:weights=np.ones(len(y))
    A=K+np.diag(alpha/np.maximum(weights,1e-6))
    sol=solve(A,np.column_stack((z,np.ones(len(y)))),assume_a='pos',check_finite=False)
    intercept=sol[:,0].sum()/sol[:,1].sum()
    coef=sol[:,0]-intercept*sol[:,1]
    p=Q@coef+intercept
    # Comparable ensemble units; intercept remains free (no measured WT assumed).
    return (p-z.mean())/max(float(z.std()),1e-8)


def predict_raw(case):
    supports=case.get('support',[]);query=case.get('query',[])
    if not supports:return np.zeros(len(query))
    reference=case['reference_sequence']
    f=features(supports,reference);g=features(query,reference)
    y=np.array([float(r['fitness']) for r in supports])
    if len(y)<2 or np.ptp(y)<1e-12:return np.zeros(len(query))
    sp=similarity_parts(f,f);qp=similarity_parts(g,f)
    chemical=[]
    # Nearby regularization and similarity settings, equally weighted to reduce
    # tuning variance; target scales are refit within every supplied training set.
    for bandwidth in [2.,4.]:
        K=similarity_kernel(f,f,sp,bandwidth);Q=similarity_kernel(g,f,qp,bandwidth)
        for scale in ['raw','log']:
            chemical.append(regression(K,Q,y,.4,scale))
    # Second-order tensor covariances allow generic pair effects to be learned
    # from feedback, while shrinking unobserved combinations strongly.
    K0=interaction_base(f,f);Q0=interaction_base(g,f)
    normalizer=max(float(np.diag(K0).mean()),1e-8)
    nonlinear=[]
    for interaction in [.2,.6]:
        K=K0+interaction*K0*K0/normalizer
        Q=Q0+interaction*Q0*Q0/normalizer
        for scale in ['raw','log']:
            nonlinear.append(regression(K,Q,y,.4,scale))
    return .8*np.mean(chemical,axis=0)+.2*np.mean(nonlinear,axis=0)


def propose(case,assignment):
    """Fit only supplied supports, score every current query, request no assays."""
    query=case.get('query',[])
    if not query:return {'scores':{},'selected_ids':[]}
    pred=np.nan_to_num(predict_raw(case),nan=0.,posinf=1e6,neginf=-1e6)
    # Midranks preserve ties and bound finite scores strictly inside [-1, 1].
    scores=2*(rankdata(pred,method='average')-.5)/len(pred)-1
    return {'scores':{str(q['id']):float(s) for q,s in zip(query,scores)},'selected_ids':[]}
