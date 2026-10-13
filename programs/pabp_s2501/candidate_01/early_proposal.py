"""Deterministic support-only hierarchical sequence regression.
No training records, learned coefficients or query identities are stored here.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Kyte-Doolittle hydropathy, residue volume, formal charge; ordinary chemical constants.
HYD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]
CHG = [0,0,-1,-1,0,0,.1,0,1,0,0,0,0,0,1,0,0,0,0,0]
PROP = np.array([[HYD[i]/4.5,(VOL[i]-140)/60,CHG[i],float(a in 'FWY'),float(a in 'STNQ'),float(a=='P'),float(a=='G')] for i,a in enumerate(AA)])

def mutations(record, reference):
    seq = record.get('sequence')
    if seq and len(seq)==len(reference):
        return [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,seq)) if a!=b]
    out=[]
    for t in record.get('mutation','').split(':'):
        m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',t)
        if m: out.append((int(m[2]),m[1],m[3]))
    return out

def region(p):
    # Hypothesized RRM secondary structure from the supplied sequence's RNP motifs.
    # These boundaries are modeling assumptions, not observed structure.
    return int(np.searchsorted([134,141,151,157,166,174,181,191,195,200,201],p,side='right'))

def features(records, reference):
    length=len(reference)
    blocks=[]
    pos=np.zeros((len(records),length))
    exact=np.zeros((len(records),length*20))
    chem=np.zeros((len(records),length*21))
    reg=np.zeros((len(records),12*21))
    glob=np.zeros((len(records),21+400+20))
    for i,r in enumerate(records):
        for p,a,b in mutations(r,reference):
            if a not in AA or b not in AA or not 1<=p<=length: continue
            ai,bi=AA.index(a),AA.index(b)
            d=PROP[bi]-PROP[ai]
            v=np.r_[1.,d,np.abs(d),PROP[bi][:6]] # 21 dimensions
            pos[i,p-1]+=1
            exact[i,(p-1)*20+bi]+=1
            chem[i,(p-1)*21:p*21]+=v
            rr=region(p)
            reg[i,rr*21:(rr+1)*21]+=v
            glob[i,:21]+=v
            glob[i,21+ai*20+bi]+=1
            glob[i,421+bi]+=1
    # Return unnormalized components so regularization strengths have meaning.
    return pos,exact,chem,reg,glob

def kernels(records, reference):
    pos,exact,chem,reg,glob=features(records,reference)
    # Position pooling plus position-specific chemistry and exact substitutions.
    k=pos@pos.T + .5*(exact@exact.T) + .10*(chem@chem.T) + .05*(reg@reg.T)+.02*(glob@glob.T)
    return k

def propose(case, assignment):
    support=[r for r in case.get('support',[]) if np.isfinite(float(r['fitness']))]
    query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if len(support)<2: return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    k=kernels(support+query,case['reference_sequence'])
    n=len(support)
    y=np.log(np.maximum(np.array([float(r['fitness']) for r in support]),0)+.05)
    mean=float(y.mean())
    coef=solve(k[:n,:n]+1.*np.eye(n),y-mean,assume_a='pos')
    pred=k[n:,:n]@coef+mean
    scores=2*(rankdata(pred,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(s) for r,s in zip(query,scores)},'selected_ids':[]}
