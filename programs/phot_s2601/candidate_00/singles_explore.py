import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,numpy as np
from scipy.stats import spearmanr
from proposal import mutations
c=json.load(open('case.json'));rs=[r for r in c['support'] if r['mutation_order']==1]
ms=[next(iter(mutations(r,c['reference_sequence']))) for r in rs]
y=np.array([r['fitness'] for r in rs]);aa='ACDEFGHIKLMNPQRSTVWY'
# Standard Kyte-Doolittle hydropathy. Binary chemistry descriptors are constants, not assay data.
hydro=dict(zip(aa,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
def chem(a):return np.array([hydro[a]/4.5,float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQ'),float(a=='C')])
def features(ms):
 return np.array([np.r_[chem(m[0]),chem(m[-1]),chem(m[-1])-chem(m[0])] for m in ms])
z=features(ms);pos=np.array([int(m[1:-1]) for m in ms]);mut=np.array([m[-1] for m in ms]);wt=np.array([m[0] for m in ms]);n=len(ms)
results=[]
for site in [0,.1,1,3]:
 for ch in [.01,.1,.3,1]:
  for length in [0,3,10]:
   for reg in [.1,.3,1,3]:
    k=ch*z@z.T+site*(pos[:,None]==pos[None,:])
    k+=.1*(mut[:,None]==mut[None,:])+.1*(wt[:,None]==wt[None,:])
    if length:k+=.1*np.exp(-abs(pos[:,None]-pos[None,:])/length)
    # exact intercept-aware leave-one-out predictions
    k+=10
    inv=np.linalg.inv(k+reg*np.eye(n));p=y-(inv@y)/np.diag(inv)
    results.append((spearmanr(y,p).statistic,np.sqrt(np.mean((y-p)**2)),site,ch,length,reg))
for r in sorted(results,reverse=True)[:20]:print(r)
