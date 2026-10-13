"""Deterministic support-only sequence regression for GCN4 ranking.
All fitted quantities are estimated inside propose; no training data are stored.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Sequence-only hypotheses: the assayed 44-residue activation-domain segment,
# the central WxxLF motif, and coarse chemistry. These are not measured structures.
GROUPS = ['DE','KR','FWY','WFYLIMV','LIMV','STNQ','PG','P','G','A','H','C','NQ','ST','W','F','Y']
KD = dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
HELIX = dict(zip(AA,[1.45,.77,.98,1.53,1.12,.53,1.24,1.,1.07,1.34,1.20,.73,.59,1.17,.79,.79,.82,1.14,1.14,.61]))

def sequence(record, ref):
    if record.get('sequence'): return record['sequence']
    s=list(ref)
    for m in record.get('mutation','').split(':'):
        if m: s[int(m[1:-1])-1]=m[-1]
    return ''.join(s)

def features(records, ref):
    segments=[(0,44),(0,19),(19,24),(24,44),(0,12),(12,24),(24,36),(36,44)]
    comp=[]; phys=[]; positional=[]; onehot=[]; basic=[]; interactions=[]
    wt=ref[100:144]
    for r in records:
        s=sequence(r,ref)[100:144]
        n=sum(a!=b for a,b in zip(s,wt))
        basic.append([n/44, np.sqrt(n)/np.sqrt(44),float(n>15),float(n<=1)])
        comp.append([s[a:b].count(aa) for a,b in segments for aa in AA])
        row=[]
        for a,b in segments:
            seg=s[a:b]
            row.extend(sum(c in g for c in seg) for g in GROUPS)
            row.extend([sum(KD.get(c,0) for c in seg)/4, sum(HELIX.get(c,1) for c in seg)])
        phys.append(row)
        p=np.array([[float(c in g) for c in s] for g in GROUPS[:7]]+[[KD.get(c,0)/4 for c in s],[HELIX.get(c,1) for c in s]])
        positional.append(p.ravel())
        onehot.append([float(c==aa)-float(w==aa) for c,w in zip(s,wt) for aa in AA])
        acid=np.array([c in 'DE' for c in s],float); hydro=np.array([c in 'WFYLIMV' for c in s],float)
        arom=np.array([c in 'FWY' for c in s],float); base=np.array([c in 'KR' for c in s],float)
        ints=[]
        for radius in [2,4,8]:
            near=np.convolve(acid,np.ones(2*radius+1),mode='same')
            ints.extend([np.dot(near,hydro), np.dot(near,arom),np.max(np.convolve(hydro,np.ones(2*radius+1),mode='valid')),np.max(np.convolve(acid,np.ones(2*radius+1),mode='valid'))])
        ints.extend([acid.sum()*hydro.sum()/44,acid.sum()*arom.sum()/44,base.sum()*arom.sum()/44,
                     float(s[19]=='W'),float(s[22]=='L'),float(s[23]=='F'),float(s[19]=='W' and s[22]=='L' and s[23]=='F'),
                     sum(s[i]==wt[i] for i in [6,7,9,12]), sum(s[i] in 'FWYLIMV' for i in [6,7,9,12]),
                     sum(s[i] in 'P' for i in range(18,25)),sum(s[i] in 'DE' for i in range(0,19))])
        interactions.append(ints)
    return {k:np.array(v,dtype=float) for k,v in [('comp',comp),('phys',phys),('pos',positional),('onehot',onehot),('basic',basic),('inter',interactions)]}

def propose(case, assignment):
    support=case.get('support',[]); query=case.get('query',[])
    if not query: return {'scores':{},'selected_ids':[]}
    if len(support)<2: return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    f=features(support+query,case['reference_sequence']); n=len(support)
    x=np.hstack([f['comp'],f['basic']]); mu=x[:n].mean(0); sd=np.maximum(x[:n].std(0),1.)
    x=(x-mu)/sd
    y=np.array([r['fitness'] for r in support],float)
    model=Ridge(alpha=30).fit(x[:n],y)
    z=model.predict(x[n:]); z=2*(rankdata(z)-.5)/len(z)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,z)},'selected_ids':[]}
