"""WT-only backbone contact priors; no DSSP, energies or mutant coordinates."""
from pathlib import Path
import numpy as np
from structure_tools import _read_pdb, _map_raw_residues, _prepare

def build(case):
 rs,mapping=_map_raw_residues(_prepare(case),_read_pdb(Path('raw_evidence/PABP_YEAST.pdb')),None)
 r={x['position']:x for x in rs if x.get('chain') is None and 'position' in x}
 assert ''.join(r[p]['WT'] for p in sorted(r))==case['reference_sequence']
 out={}
 for p,x in r.items():
  hits=[];amide=x['atoms'].get('N');oxy=x['atoms'].get('O')
  for q,y in r.items():
   if abs(q-p)<=2 or not amide or 'O' not in y['atoms']:continue
   d=float(np.linalg.norm(np.array(amide['coord'])-np.array(y['atoms']['O']['coord'])))
   if 2.5<=d<=3.4:hits.append(dict(position=q,distance_A=d))
  helix_contact=False
  # An observed i-to-i+4 contact supplies a propensity prior; it is not a
  # secondary-structure assignment or a quantified hydrogen-bond energy.
  for op,np_ in [(p,p+4),(p-4,p)]:
   if op in r and np_ in r and 'O' in r[op]['atoms'] and 'N' in r[np_]['atoms']:
    dd=float(np.linalg.norm(np.array(r[op]['atoms']['O']['coord'])-np.array(r[np_]['atoms']['N']['coord'])))
    if 2.5<=dd<=3.4:helix_contact=True
  out[str(p)]=dict(nonlocal_backbone_N_O_distance_contacts=hits,has_i_i_plus4_backbone_contact=helix_contact)
 return dict(format='WT_backbone_distance_contact_prior_v1',mapping=mapping,limitations=['Static WT distance contacts are not confirmed hydrogen bonds.','No mutant structures, energy, affinity or epistasis.'],sites=out)
