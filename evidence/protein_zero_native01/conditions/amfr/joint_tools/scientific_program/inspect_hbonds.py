import numpy as np,json
from collections import defaultdict
atoms=defaultdict(dict)
for line in open('raw_evidence/AMFR_HUMAN.pdb'):
 if line.startswith('ATOM'):
  p=int(line[22:26]);a=line[12:16].strip();element=line[76:78].strip();atoms[p][a]=(element,np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]))
ref=json.load(open('case.json'))['reference_sequence']
donors={'R':['NE','NH1','NH2'],'N':['ND2'],'Q':['NE2'],'K':['NZ'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}
acceptors={'D':['OD1','OD2'],'E':['OE1','OE2'],'N':['OD1'],'Q':['OE1'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}
bonds=[]
for p,w in enumerate(ref,1):
 for dn in donors.get(w,[])+(['N'] if w!='P' else []):
  if dn not in atoms[p]:continue
  d=atoms[p][dn][1];hs=[x for e,x in atoms[p].values() if e=='H' and np.linalg.norm(x-d)<1.3]
  for q,a in enumerate(ref,1):
   if p==q:continue
   for an in acceptors.get(a,[])+['O','OXT']:
    if an not in atoms[q] or dn=='N' and an in ('O','OXT'):continue
    ac=atoms[q][an][1];dist=np.linalg.norm(d-ac)
    if dist>3.8:continue
    angle=max([float(np.degrees(np.arccos(np.clip(np.dot(d-h,ac-h)/np.linalg.norm(d-h)/np.linalg.norm(ac-h),-1,1)))) for h in hs] or [0])
    if angle>110:
     bonds.append((p,dn,q,an,round(float(dist),3),round(angle,1)))
print(bonds)
