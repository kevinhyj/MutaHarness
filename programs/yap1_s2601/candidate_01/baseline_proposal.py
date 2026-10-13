"""Support-only deterministic ordinal predictor for substitution variants.

All coefficients are estimated afresh from the supplied support. Amino-acid
properties below are scientific constants, not predictor scores or fitted data.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy, approximate sidechain volume, net charge at pH 7,
# aromatic, proline, glycine, H-bond donor/acceptor sidechain flags.
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]

def _properties():
    raw = np.array([[HYDRO[i], VOL[i], {'D':-1,'E':-1,'K':1,'R':1,'H':.1}.get(a,0), float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQH'),float(a in 'DE'),float(a in 'KR'),float(a=='C')] for i,a in enumerate(AA)])
    return (raw-raw.mean(0))/np.maximum(raw.std(0),.1)

PROPS = _properties()

def _mutations(record, reference):
    sequence = record.get('sequence')
    if sequence and len(sequence)==len(reference):
        return [(i,reference[i],a) for i,a in enumerate(sequence) if a != reference[i]]
    return [(int(p)-1,w,a) for w,p,a in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def _features(records, reference):
    mutations = [_mutations(r,reference) for r in records]
    positions = sorted({p for ms in mutations for p,w,a in ms})
    ix = {p:i for i,p in enumerate(positions)}
    n,l=len(records),len(positions)
    pos=np.zeros((n,l)); exact=np.zeros((n,l*20)); prop=np.zeros((n,l*10)); glob=np.zeros((n,30))
    for i,ms in enumerate(mutations):
        for p,w,a in ms:
            if w not in AA or a not in AA: continue
            j=ix[p]; ai=AA.index(a); wi=AA.index(w)
            pos[i,j]=1; exact[i,j*20+ai]=1
            d=PROPS[ai]-PROPS[wi]
            prop[i,j*10:(j+1)*10]=d/np.sqrt(10)
            glob[i,:10]+=d/np.sqrt(10)
            glob[i,10:30]+=np.eye(20)[ai]-np.eye(20)[wi]
    return pos,exact,prop,glob

def propose(case, assignment):
    support=case.get('support',[]); query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if not support: return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    n=len(support)
    pos,exact,prop,glob=_features(support+query,case['reference_sequence'])
    k=pos@pos.T + .5*(exact@exact.T) + .15*(prop@prop.T) + .1*(glob@glob.T)
    y=np.array([float(r['fitness']) for r in support])
    y=np.log(np.maximum(y,1e-8)) if np.all(y>0) else np.arcsinh(y)
    mean=y.mean()
    alpha=solve(k[:n,:n]+np.eye(n)*.3,y-mean,assume_a='pos')
    pred=k[n:,:n]@alpha+mean
    scores=2*(rankdata(pred)-.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,scores)},'selected_ids':[]}
