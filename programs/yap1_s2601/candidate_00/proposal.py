#!/usr/bin/env python3
"""Support-only ensemble for the YAP1 binding assay.

Scientific constants below are amino-acid chemistry, not predictor scores.
Each invocation builds its feature vocabulary from the supplied records and
fits fresh coefficients solely to that invocation's support outcomes.
Regularization configurations are chosen by public-support cross-validation;
no fitted coefficients, fitness values, IDs, or prediction lists are stored.
The deployed fit uses no random operations; exploratory split seeds were fixed.
"""
import re
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata, norm

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy and approximate residue volumes (Angstrom^3).
HYD = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]

def mutations(record, reference):
    found = re.findall(r'([A-Z])(\d+)([A-Z])', record.get('mutation', ''))
    if found:
        return [(int(p),a,b) for a,p,b in found if a != b]
    return [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a != b]

def chemistry():
    return {a:np.array([HYD[i]/3.,(VOL[i]-140.)/60.,float(a in 'KR')-float(a in 'DE'),float(a=='H'),float(a in 'FWY'),float(a in 'NQST'),float(a=='P'),float(a=='G'),float(a in 'ILV'),float(a=='C')]) for i,a in enumerate(AA)}

def _distance(x, z):
    return np.maximum(0.,np.sum(x*x,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2.*x@z.T)

def kernel_blocks(records, reference):
    parsed=[mutations(r,reference) for r in records]
    vocabulary=sorted({m for ms in parsed for m in ms})
    if not vocabulary:
        return [np.zeros((len(records),len(records))) for _ in range(7)]
    index={m:i for i,m in enumerate(vocabulary)}
    incidence=np.zeros((len(records),len(vocabulary)))
    for row,ms in enumerate(parsed):
        for mutation in ms: incidence[row,index[mutation]]+=1.
    pc=chemistry()
    original=np.array([pc[a] for _,a,_ in vocabulary])
    destination=np.array([pc[b] for _,_,b in vocabulary])
    delta=destination-original
    positions=np.array([p for p,_,_ in vocabulary])
    same_site=(positions[:,None]==positions[None,:]).astype(float)
    source=np.array([a for _,a,_ in vocabulary])
    same_source=(source[:,None]==source[None,:]).astype(float)
    # A centered Gaussian residue kernel: changing WT to WT contributes zero.
    smooth=(np.exp(-_distance(destination,destination)/16.)
            -np.exp(-_distance(destination,original)/16.)
            -np.exp(-_distance(original,destination)/16.)
            +np.exp(-_distance(original,original)/16.))
    def aggregate(k): return incidence@k@incidence.T
    count=incidence.sum(axis=1)
    return [aggregate(same_site),incidence@incidence.T,
            aggregate(delta@delta.T),aggregate(same_source*(delta@delta.T)),
            aggregate(same_site*smooth),aggregate(same_source*smooth),
            count[:,None]*count[None,:]]

def target_values(values, mode='mix'):
    values=np.asarray(values,dtype=float)
    if np.all(values>0.):
        transformed=np.log(values)
    else:
        # Robust fallback for another scale or shifted public support data.
        scale=max(float(np.std(values)),1e-8)
        transformed=np.arcsinh((values-np.median(values))/scale)
    transformed=(transformed-np.mean(transformed))/max(float(np.std(transformed)),1e-8)
    ordinal=norm.ppf((rankdata(values,method='average')-.5)/len(values))
    if mode=='rank': return ordinal
    if mode=='log': return transformed
    return .5*(transformed+ordinal)

# (site, exact allele, global chemistry, same-source chemistry, ridge noise,
#  nonlinear interaction strength, mutation-count prior). Coefficients are
#  covariance amplitudes, not mutation effect estimates.
CONFIGURATIONS = (
    (.02,.02,.3,0.,.03,.2,.3),
    (.3,.3,1.,0.,.03,.2,.3),
    (.3,.1,1.,.03,.3,.2,.3),
)
ENSEMBLE_WEIGHTS = (.6,.2,.2)

def model_kernel(blocks, cfg):
    loc,exact,global_chem,source_chem,noise,interaction,count=cfg
    k=(loc*blocks[0]+exact*blocks[1]+global_chem*blocks[2]
       +source_chem*blocks[3]+.1*blocks[4]+.1*blocks[5]+count*blocks[6])
    # Squared PSD kernel borrows interaction effects across chemically similar
    # combinations. It does not assert a specific contact or binding mechanism.
    return k+interaction*k*k/4.+10.

def ensemble_predict(support, query, reference, configs=None):
    if not support: return np.zeros(len(query))
    n=len(support)
    blocks=kernel_blocks(list(support)+list(query),reference)
    values=np.array([float(r['fitness']) for r in support])
    predictions=[]
    configurations = CONFIGURATIONS if configs is None else tuple(configs)
    for cfg in configurations:
        kernel=model_kernel(blocks,cfg)
        factor=cho_factor(kernel[:n,:n]+cfg[4]*np.eye(n),check_finite=False)
        # Rank is the primary endpoint. A small log-scale component preserves
        # quantitative evidence while limiting the influence of extreme values.
        target=.25*target_values(values,'mix')+.75*target_values(values,'rank')
        coefficients=cho_solve(factor,target,check_finite=False)
        predictions.append(kernel[n:,:n]@coefficients)
    weights=ENSEMBLE_WEIGHTS if configs is None else np.ones(len(predictions))/len(predictions)
    return np.average(predictions,axis=0,weights=weights)

def propose(case, assignment):
    query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    pred=ensemble_predict(case.get('support',[]),query,case['reference_sequence'])
    pred=np.nan_to_num(pred,nan=0.,posinf=1e100,neginf=-1e100)
    # Bounded scores encode ordering only; ties stay tied. No new measurements.
    score=2.*(rankdata(pred,method='average')-.5)/len(pred)-1.
    return {'scores':{r['id']:float(v) for r,v in zip(query,score)},'selected_ids':[]}
