"""Deterministic support-only, regularized mutational ranking model.
All coefficients are re-estimated on every call. No reference fitness is assumed.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy and approximate side-chain volumes: scientific constants.
HYD = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]

def mutations(record, reference):
    text = record.get('mutation', '')
    found = re.findall(r'([A-Z])(\d+)([A-Z])', text)
    if found:
        return [(int(p),a,b) for a,p,b in found if a != b]
    return [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a != b]

def chemistry():
    vals=[]
    for i,a in enumerate(AA):
        vals.append([HYD[i]/3., (VOL[i]-140)/60., float(a in 'KR')-float(a in 'DE'), float(a=='H'), float(a in 'FWY'), float(a in 'NQST'), float(a=='P'), float(a=='G'), float(a in 'ILV'), float(a=='C')])
    return {a:np.array(v) for a,v in zip(AA,vals)}

def design(records, reference, positions, alleles):
    pc=chemistry(); pi={p:i for i,p in enumerate(positions)}; ai={a:i for i,a in enumerate(alleles)}
    n=len(records); p=len(positions); d=len(pc['A'])
    loc=np.zeros((n,p)); exact=np.zeros((n,len(alleles))); prop=np.zeros((n,p*d)); glob=np.zeros((n,d)); count=np.zeros((n,1))
    for row,r in enumerate(records):
        ms=mutations(r,reference); count[row,0]=len(ms)
        for pos,a,b in ms:
            if pos not in pi or a not in pc or b not in pc: continue
            i=pi[pos]; loc[row,i]+=1
            delta=pc[b]-pc[a]; prop[row,i*d:(i+1)*d]+=delta; glob[row]+=delta
            if (pos,a,b) in ai: exact[row,ai[(pos,a,b)]]+=1
    return loc,exact,prop,glob,count

def predict(support, query, reference, config=None):
    config=config or (1., 1., .15, .05, .4)
    allrecords=list(support)+list(query)
    ms=[m for r in allrecords for m in mutations(r,reference)]
    pos=sorted({m[0] for m in ms}); alleles=sorted(set(ms))
    blocks=design(allrecords,reference,pos,alleles)
    X=np.concatenate([np.sqrt(w)*b for w,b in zip(config[:4],blocks[:4])],axis=1)
    n=len(support); T=X[:n]; Q=X[n:]; y=np.array([float(r['fitness']) for r in support]); y=np.log(np.maximum(y,1e-6))
    # Broad intercept prior, inferred entirely from the observed records.
    K=T@T.T+10.; k=Q@T.T+10.
    coef=solve(K+config[4]*np.eye(n),y,assume_a='pos')
    return k@coef

def propose(case, assignment):
    support=case.get('support',[]); query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if support: pred=predict(support,query,case['reference_sequence'])
    else: pred=np.zeros(len(query))
    # Empirical ranks are bounded ordinal values, never probabilities.
    score=2*(rankdata(pred,method='average')-.5)/len(pred)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,score)},'selected_ids':[]}
