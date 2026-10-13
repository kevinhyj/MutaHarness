import json,numpy as np
from pathlib import Path
from structure_tools import _read_pdb
r=_read_pdb(Path('raw_evidence/UBE4B_HUMAN.pdb'))
atoms={i+1:{k:np.array(v['coord']) for k,v in res['atoms'].items()} for i,res in enumerate(r)}
types={i+1:res['WT'] for i,res in enumerate(r)}
side_polar={'D':['OD1','OD2'],'E':['OE1','OE2'],'R':['NE','NH1','NH2'],'K':['NZ'],'N':['OD1','ND2'],'Q':['OE1','NE2'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}
for pair in [(8,28),(13,28),(17,29),(17,61),(29,61),(20,55),(23,42),(23,45),(39,41)]:
 i,j=pair; print('PAIR',i,types[i],j,types[j]); contacts=[]
 for a,x in atoms[i].items():
  for b,y in atoms[j].items():
   if a.startswith('H') or b.startswith('H'):continue
   d=np.linalg.norm(x-y)
   if d<4.0 and (a in side_polar.get(types[i],[]) or b in side_polar.get(types[j],[])):contacts.append((round(float(d),3),a,b))
 print(sorted(contacts))
# Nearby contacts for buried acids and polar network centers.
for i in [8,17,28,29,42,61]:
 print('POLAR SITE',i,types[i]);contacts=[]
 for j,other in atoms.items():
  if i==j: continue
  for a in side_polar.get(types[i],[]):
   for b in ['N','O']+side_polar.get(types[j],[]):
    if b not in other:continue
    d=np.linalg.norm(atoms[i][a]-other[b])
    if d<3.7: contacts.append((j,types[j],round(float(d),3),a,b))
 print(contacts)
