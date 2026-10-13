"""Support-only deterministic ranking, using regularized additive mutation kernels.
All fitted quantities are recomputed from case['support']; no data files are read.
"""
import re
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Hydropathy (Kyte-Doolittle), volume, charge, polarity, and sidechain flags.
_H = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
_V = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
_Q = [0,0,-1,-1,0,0,.25,0,1,0,0,0,0,0,1,0,0,0,0,0]
_P = [8.1,5.5,13,12.3,5.2,9,10.4,5.2,11.3,4.9,5.7,11.6,8,10.5,10.5,9.2,8.6,5.9,5.4,6.2]
PROP = np.array([_H,_V,_Q,_P,[a in 'FYW' for a in AA],[a=='P' for a in AA],[a=='G' for a in AA],[a in 'STNQ' for a in AA]]).T.astype(float)
PROP = (PROP-PROP.mean(0))/PROP.std(0)
INDEX = {a:i for i,a in enumerate(AA)}

def mutations(record, reference):
    text = record.get('mutation','')
    out = [(int(p),a,b) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',text)]
    if not out and record.get('sequence'):
        out = [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a!=b]
    return out

def features(records,reference):
    n=len(reference)
    pos=np.zeros((len(records),n))
    exact=np.zeros((len(records),n*20))
    prop=np.zeros((len(records),n*8))
    global_prop=np.zeros((len(records),17))
    for i,r in enumerate(records):
        ms=mutations(r,reference)
        for p,a,b in ms:
            if b not in INDEX or a not in INDEX or not 1<=p<=n: continue
            j=p-1; delta=PROP[INDEX[b]]-PROP[INDEX[a]]
            pos[i,j]+=1; exact[i,j*20+INDEX[b]]+=1
            prop[i,j*8:(j+1)*8]+=delta/np.sqrt(8)
            global_prop[i,:8]+=delta/np.sqrt(8)
            global_prop[i,8:16]+=np.abs(delta)/np.sqrt(8)
        global_prop[i,16]=len(ms)
    return pos,exact,prop,global_prop

def kernel(f,g,config):
    p,e,d,h=f; q,z,v,j=g
    k=p@q.T + config.get('exact',.5)*(e@z.T) + config.get('chem',.25)*(d@v.T)
    if config.get('global',0): k += config['global']*(h@j.T)
    return k

def fit_predict(support,query,reference,config):
    f=features(support,reference); g=features(query,reference)
    k=kernel(f,f,config); kt=kernel(g,f,config)
    y=np.array([float(r['fitness']) for r in support])
    mode=config.get('target','raw')
    if mode=='log': y=np.log(np.maximum(y,0)+config.get('offset',.1))
    elif mode=='rank': y=(rankdata(y)-.5)/len(y)
    elif mode=='sqrt': y=np.sqrt(np.maximum(y,0))
    mean=y.mean(); alpha=config.get('alpha',1.)
    weights=cho_solve(cho_factor(k+alpha*np.eye(len(y)),check_finite=False),y-mean,check_finite=False)
    return mean+kt@weights

def propose(case,assignment):
    support=case.get('support',[]); query=case.get('query',[])
    if not support: return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    pred=fit_predict(support,query,case['reference_sequence'],{'alpha':1.,'chem':.25,'exact':.5,'global':.05,'target':'raw'})
    scores=2*(rankdata(pred)-.5)/max(1,len(pred))-1
    return {'scores':{r['id']:float(s) for r,s in zip(query,scores)},'selected_ids':[]}
