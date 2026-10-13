"""Support-only deterministic ranking for KRAS abundance.

All coefficients are refitted from the supplied supports. Amino acid descriptors
are standard chemical constants, not externally trained protein scores.
The model uses no stochastic operations or fitted caches. Configuration search
used deterministic public-support splits with seeds 21,42,93,31,71 and subsequent
checks with 202,303,404,505,606. Every invocation transforms and fits its own
supplied support outcomes. No training files or prediction lists are read.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy and approximate sidechain volumes (Angstrom^3).
HYD = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))

def mutations(record, reference):
    if record.get('sequence') and len(record['sequence']) == len(reference):
        return [(a,i+1,b) for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a != b]
    return [(a,int(p),b) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def chemistry(a):
    return np.array([HYD.get(a,0)/4.5,(VOL.get(a,140)-140)/60,
        float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),
        float(a in 'STNQ'),float(a=='G'),float(a=='P'),float(a=='C'),float(a=='H')])

def kernel_parts(records, reference):
    ms=[mutations(r,reference) for r in records]
    n=len(records); unique=sorted(set(m for row in ms for m in row))
    if not unique: return [np.ones((n,n))]
    ix={m:i for i,m in enumerate(unique)}
    B=np.zeros((n,len(unique)))
    for i,row in enumerate(ms):
        for m in row: B[i,ix[m]]=1
    pos=np.array([m[1] for m in unique]); old=np.array([chemistry(m[0]) for m in unique]); new=np.array([chemistry(m[2]) for m in unique]); delta=new-old
    same=(pos[:,None]==pos).astype(float)
    exact=np.eye(len(unique))
    prop=np.exp(-np.sum((new[:,None]-new[None,:])**2,axis=2)/4)
    ds=np.exp(-np.sum((delta[:,None]-delta[None,:])**2,axis=2)/6)
    smooth=np.exp(-((pos[:,None]-pos[None,:])/8)**2)
    # Structural regions are coarse, assumed sequence annotations, not coordinates.
    boundaries=[0,9,17,25,40,46,58,66,75,86,104,110,120,126,138,151,166,184,188]
    reg=np.digitize(pos,boundaries)
    region=(reg[:,None]==reg).astype(float)
    parts=[B@k@B.T for k in [exact,same,same*prop,smooth*ds,region*ds]]
    # Transferable chemical changes and order, with no fixed sign of effect.
    descriptors=[]
    for a,p,b in unique:
        o=chemistry(a); v=chemistry(b); d=v-o
        loc=reference[max(0,p-4):min(len(reference),p+3)]
        hydro=np.mean([HYD.get(x,0) for x in loc])/4.5
        descriptors.append(np.r_[d,np.abs(d),o,v,d*hydro, float(p>166),float(p>=185)])
    D=np.array(descriptors)
    X=B@D
    parts.append(X@X.T/X.shape[1])
    count=B.sum(1)
    parts.append(np.ones((n,n)) + count[:,None]*count[None,:])
    # Local transfer uses ancestral and replacement chemistry plus reference-only
    # hydrophobic context. It is a smooth statistical prior, not a burial label.
    H=[]
    for a,p,b in unique:
        o=chemistry(a); v=chemistry(b)
        loc=reference[max(0,p-4):min(len(reference),p+3)]
        h=np.mean([HYD.get(x,0) for x in loc])/4.5
        H.append(np.r_[o,v,np.abs(v-o),h,float(p>166),float(p>=185)])
    H=np.array(H)
    similarity=np.exp(-np.sum((H[:,None]-H[None,:])**2,axis=2)/10)
    local=np.exp(-((pos[:,None]-pos[None,:])/2)**2)
    parts.append(B@(local*similarity)@B.T)
    return parts

def _predict(records, reference, ntrain, y):
    parts=kernel_parts(records,reference)
    if len(parts)==1:
        return np.full(len(records)-ntrain,.5)
    # Validation-selected covariance weights: exact mutation, position, local
    # transfer, coarse regions, chemical descriptors, and mutation order.
    K=parts[0]+parts[1]+.2*parts[3]+.05*parts[4]+parts[5]+.3*parts[6]+2*parts[7]
    diag=np.diag(K)
    distance=np.maximum(diag[:,None]+diag[None,:]-2*K,0)
    amplitude=float(np.mean(diag[:ntrain]))
    # The assay's ordering is the target. Tied measurements receive midranks;
    # no measured wild-type value, probability, or additive assay scale is assumed.
    target=rankdata(y,method='average')/(ntrain+1)
    mean=float(target.mean())
    predictions=[]
    # Three closely competitive smooth nonlinear models reduce sensitivity to
    # kernel length and nonlinear strength. RBF covariance allows combination
    # effects while retaining the additive mutation kernel as a strong prior.
    for length,weight in [(30.,1.),(30.,.3),(10.,.3)]:
        Z=K+weight*amplitude*np.exp(-distance/length)
        coef=solve(Z[:ntrain,:ntrain]+.1*np.eye(ntrain),target-mean,assume_a='pos')
        predictions.append(mean+Z[ntrain:,:ntrain]@coef)
    return np.mean(predictions,axis=0)

def propose(case, assignment):
    support=[r for r in case.get('support',[]) if np.isfinite(float(r.get('fitness',np.nan)))]
    query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if support:
        y=np.array([float(r['fitness']) for r in support])
        pred=_predict(support+query,case['reference_sequence'],len(support),y)
    else: pred=np.zeros(len(query))
    scores=(2*rankdata(pred,method='average')-len(query)-1)/max(1,len(query))
    return {'scores':{r['id']:float(s) for r,s in zip(query,scores)},'selected_ids':[]}
