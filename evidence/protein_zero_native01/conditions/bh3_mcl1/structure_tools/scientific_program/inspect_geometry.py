import json
from pathlib import Path
import numpy as np
import structure_tools as st
c=json.load(open('case.json')); state=st._prepare(c)
res,mapping=st._map_raw_residues(state,st._read_pdb(Path('raw_evidence/3KZ0_hm.pdb'),include_zero_occupancy=True),None)
lookup={(r['chain'],r['position']):r for r in res if r['protein']}
charges=[]
for r in res:
 if r.get('chain')!='A': continue
 names={'D':['OD1','OD2'],'E':['OE1','OE2'],'R':['NE','NH1','NH2'],'K':['NZ'],'H':['ND1','NE2']}.get(r['WT'],[])
 if names: charges.append((r['position'],r['WT'],np.mean([r['atoms'][n]['coord'] for n in names],axis=0)))
for p in [6,7,9,10,11,13,14,15,16,17]:
 r=lookup['C',p]; aa=r['WT']; atoms=r['atoms']; cb=atoms.get('CB',atoms['CA'])['coord']
 nearest=sorted([(float(np.linalg.norm(np.array(cb)-v)),a,n) for n,a,v in charges])[:5]
 print(p,aa,'nearest Mcl1 charge-centers to CB',[(round(d,2),a,n) for d,a,n in nearest])
 if p==11:
  for n in ['CB','CG','CD','NE','CZ','NH1','NH2']:
   v=np.array(atoms[n]['coord']); nearby=sorted([(float(np.linalg.norm(v-x)),a,po) for po,a,x in charges])[:3]
   print(n,[(round(d,2),a,po) for d,a,po in nearby])
all_atoms=[a for r in res for a in r['atoms'].values()]
C_atoms=[a for r in res if r.get('chain')=='C' for a in r['atoms'].values()]
comp=st._sasa_arrays(np,all_atoms,100,1.4); iso=st._sasa_arrays(np,C_atoms,100,1.4)
ci={id(a):v for a,v in zip(all_atoms,comp)}; ii={id(a):v for a,v in zip(C_atoms,iso)}
print('peptide sidechain-heavy isolated / buried SASA')
for p in [6,7,9,10,11,13,14,15,16,17]:
 r=lookup['C',p]
 side=[a for n,a in r['atoms'].items() if n not in ('N','CA','C','O','OXT') and a['element']!='H']
 print(p,r['WT'],round(sum(ii[id(a)] for a in side),1),round(sum(ii[id(a)]-ci[id(a)] for a in side),1))
