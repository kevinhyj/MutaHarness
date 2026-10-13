import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.special import expit,logit
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from proposal import *

# A priori approximate secondary-structure segments and hydrophobic face.
# These are hypotheses, not structure coordinates.
STRANDS=[(11,22,0),(25,37,1),(40,50,0),(80,89,0),(94,104,0),(117,124,1),(129,142,0),(145,153,0),(165,172,1),(181,190,0),(198,207,1),(219,229,1)]
def mfeat(ms,ref):
    out=[]
    for p,a,b in ms:
        h=HYD[a];dh=HYD[b]-h;v=(VOL[b]-VOL[a])/100;q=CH[b]-CH[a]
        strand=next(((l,u,f) for l,u,f in STRANDS if l<=p<=u),None)
        core=float(strand is not None and p%2==strand[2]);helix=float(58<=p<=75);critical=float(p in (65,66,67));outer=float(strand is not None and not core)
        x=[1,abs(dh)/5,abs(v),abs(q),max(0,-dh)/5,max(0,dh)/5,max(0,v),max(0,-v),float(b=='P'),float(a=='P'),float(b=='G'),float(a=='G'),float(a in 'FWY' and b not in 'FWY'),float(a in 'ILVMFWY' and b in 'DEKRQN'),float(a in 'DEKR' and b in 'DEKR' and q!=0),float(a in 'ST' and b in 'ST'),float(a in 'ILVM' and b in 'ILVM'),float(a in 'FWY' and b in 'FWY'),critical,helix,float(p<=9 or p>=230)]
        x += [core,outer,float(strand is not None),core*abs(dh)/5,core*max(0,-dh)/5,core*abs(v),core*abs(q),core*(b=='P'),outer*(b=='P'),helix*(b=='P'),float(strand is None)*(b=='P'),core*(a=='G'),float(p in (65,66,67,96,148,203,205,222)),float(b in 'DEKR'),float(a in 'DEKR'),float(b in 'QN'),float(b in 'FWY')]
        # Wildtype and target residue pooling gives generic tolerances.
        x += [float(a==aa) for aa in AA]+[float(b==aa) for aa in AA]
        out.append(x)
    return np.array(out) if out else np.zeros((1,77))

def extra_features(records,ref):
    rows=[]
    for r in records:
        ms=mutations(r,ref); m=mfeat(ms,ref)
        rows.append(np.r_[m.sum(0),m.max(0),len(ms)**2/10,np.log1p(len(ms))])
    return np.array(rows)

def tokens(records,ref):
    mts=[mutations(r,ref) for r in records]
    cats=sorted(set((p,a,b) for ms in mts for p,a,b in ms));pos=sorted(set(p for ms in mts for p,a,b in ms));trs=sorted(set((a,b) for ms in mts for p,a,b in ms))
    def design(rows):
        mt=[mutations(r,ref) for r in rows]
        e=np.array([[float(t in m) for t in cats] for m in mt]);p=np.array([[float(any(t[0]==pp for t in m)) for pp in pos] for m in mt]);t=np.array([[sum((a,b)==tr for pp,a,b in m) for tr in trs] for m in mt])
        return e,p,t
    return design

