"""Support-fitted, deterministic ordinal predictor; no stored outcomes or fitted state."""
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Standard physical descriptors: Kyte-Doolittle hydropathy, side-chain volume,
# formal charge, aromaticity, glycine, proline, and hydrogen-bond polarity.
_HYD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
_VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
_P = np.array([[ _HYD[i]/3, _VOL[i]/60, float(a in 'KR')-float(a in 'DE')+.15*(a=='H'),float(a in 'FWY'),float(a=='G'),float(a=='P'),float(a in 'NQSTDEKRH')] for i,a in enumerate(AA)])
_P = (_P-_P.mean(0))/np.maximum(_P.std(0),.1)
_AI = {a:i for i,a in enumerate(AA)}

def mutations(record, reference):
    seq = record.get('sequence')
    if seq:
        return [(i,_AI[b]) for i,(a,b) in enumerate(zip(reference,seq)) if a!=b and b in _AI]
    import re
    return [(int(p)-1,_AI[b]) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def features(records,reference):
    n=len(records); l=len(reference)
    pos=np.zeros((n,l)); exact=np.zeros((n,l*20)); chem=np.zeros((n,l*len(_P[0])))
    glob=np.zeros((n,23))
    for row,r in enumerate(records):
        for p,a in mutations(r,reference):
            if p>=l or reference[p] not in _AI: continue
            w=_AI[reference[p]]; d=_P[a]-_P[w]
            pos[row,p]=1; exact[row,p*20+a]=1; chem[row,p*7:(p+1)*7]=d/np.sqrt(7)
            glob[row,:7]+=d/np.sqrt(7); glob[row,7:14]+=np.abs(d)/np.sqrt(7)
            glob[row,14:21]+=_P[a]/np.sqrt(7); glob[row,21]+=1; glob[row,22]+=np.linalg.norm(d)/3
    return pos,exact,chem,glob

def propose(case,assignment):
    support=[r for r in case['support'] if r.get('fitness') is not None and np.isfinite(float(r['fitness']))]
    query=case['query']; n=len(support)
    if not n: return {'scores':{str(r['id']):0. for r in query},'selected_ids':[]}
    p,e,c,g=features(support+query,case['reference_sequence'])
    k=p@p.T+ .4*(e@e.T)+.2*(c@c.T)+.05*(g@g.T)+1.
    y=np.array([float(r['fitness']) for r in support]); mu=y.mean()
    coef=solve(k[:n,:n]+1.5*np.eye(n),y-mu,assume_a='pos')
    pred=mu+k[n:,:n]@coef
    scores=2*(rankdata(pred)-.5)/max(1,len(pred))-1
    return {'scores':{str(r['id']):float(s) for r,s in zip(query,scores)},'selected_ids':[]}
