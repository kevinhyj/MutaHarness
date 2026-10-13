"""Sequence-only features. All annotations are hypotheses, not measured structure."""
import numpy as np
from proposal import _mutations
AA='ACDEFGHIKLMNPQRSTVWY'
# Standard sidechain hydropathy, volume, charge; fixed chemical descriptors.
HYD=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
def chem(a):
 return np.array([HYD[a]/3, VOL[a]/80, float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),float(a=='G'),float(a=='P'),float(a in 'STNQ'),float(a=='C')])
def blocks(rows,ref):
 mm=[_mutations(r,ref) for r in rows]
 tokens=sorted({(p,b) for m in mm for p,a,b in m}); ti={v:i for i,v in enumerate(tokens)}
 n=len(rows);L=len(ref)
 out={k:np.zeros((n,d)) for k,d in [('exact',len(tokens)),('pos',L),('subst',400),('aa',40),('chem',24),('local',L*8),('region',12*24),('count',2),('struct',10)]}
 # Coarse KRAS fold boundaries drawn from sequence/domain knowledge, never coordinates.
 edges=[0,9,17,25,40,58,76,86,104,120,138,166,L]
 for i,m in enumerate(mm):
  out['count'][i]=[len(m),len(m)**2]
  for p,a,b in m:
   out['exact'][i,ti[(p,b)]]=1;out['pos'][i,p-1]+=1
   out['subst'][i,20*AA.index(a)+AA.index(b)]+=1
   out['aa'][i,AA.index(a)]+=1;out['aa'][i,20+AA.index(b)]+=1
   ca,cb=chem(a),chem(b); desc=np.r_[cb-ca,np.abs(cb-ca),cb*ca]
   out['chem'][i]+=desc
   reg=np.searchsorted(edges,p,side='left')-1; reg=max(0,min(11,reg))
   out['region'][i,24*reg:24*(reg+1)]+=desc
   weights=np.exp(-((np.arange(L)+1-p)/7)**2/2)
   out['local'][i]+=(weights[:,None]*(cb-ca)[None,:]).ravel()/np.sqrt(12)
   h=ref[max(0,p-5):min(L,p+4)]
   hp=np.mean([HYD[x] for x in h])/3
   out['struct'][i]+=np.array([hp,hp*(HYD[b]-HYD[a])/3,hp*(b in 'DEKR'),b=='P',b=='G',a=='G',p>166,p<25,26<=p<=40,59<=p<=76])
 return out
