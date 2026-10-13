"""WT polar-contact priors; never generates mutant atoms or validated energies."""
from pathlib import Path
import numpy as np
from structure_tools import _read_pdb

DONORS={'R':['NE','NH1','NH2'],'K':['NZ'],'N':['ND2'],'Q':['NE2'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2'],'W':['NE1']}
ACCEPTORS={'D':['OD1','OD2'],'E':['OE1','OE2'],'N':['OD1'],'Q':['OE1'],'S':['OG'],'T':['OG1'],'Y':['OH'],'H':['ND1','NE2']}

def polar_contacts(path):
    residues=_read_pdb(Path(path)); assert len(residues)==69
    atoms={i+1:r['atoms'] for i,r in enumerate(residues)}
    coords={i:{a:np.array(v['coord']) for a,v in row.items()} for i,row in atoms.items()}
    bonds=[]
    for i,res in enumerate(residues,1):
        wt=res['WT']
        for donor in (['N'] if wt!='P' else [])+DONORS.get(wt,[]):
            if donor not in atoms[i]: continue
            x=coords[i][donor]
            hs=[(a,y) for a,y in coords[i].items() if atoms[i][a]['element']=='H' and np.linalg.norm(y-x)<1.25]
            # An unprotonated histidine nitrogen is not assigned as donor.
            if not hs: continue
            for j,other in enumerate(residues,1):
                if i==j: continue
                for acc in ['O']+ACCEPTORS.get(other['WT'],[]):
                    if acc not in atoms[j]: continue
                    y=coords[j][acc]; d=float(np.linalg.norm(x-y))
                    if d>3.5 or d<2.2: continue
                    if donor=='N' and acc=='O':continue # Backbone-only contacts do not respond to sidechain chemistry.
                    angles=[]
                    for _,h in hs:
                        v1=x-h;v2=y-h
                        ang=float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1))))
                        if ang>=110 and np.linalg.norm(h-y)<2.6:angles.append(ang)
                    if not angles:continue
                    bonds.append({'donor_pos':i,'donor_WT':wt,'donor_atom':donor,'donor_sidechain':donor!='N',
                                  'acceptor_pos':j,'acceptor_WT':other['WT'],'acceptor_atom':acc,'acceptor_sidechain':acc!='O',
                                  'distance':d,'angle':max(angles),
                                  'salt':wt in 'KR' and donor!='N' and other['WT'] in 'DE' and acc!='O'})
    # Avoid treating two directions of a hydroxyl pair as independent bonds.
    unique={}
    for b in bonds:
        key=tuple(sorted([(b['donor_pos'],b['donor_atom']),(b['acceptor_pos'],b['acceptor_atom'])]))
        if key not in unique or b['angle']>unique[key]['angle']:unique[key]=b
    return list(unique.values()),residues

if __name__=='__main__':
    import json
    b,_=polar_contacts('raw_evidence/UBE4B_HUMAN.pdb')
    print(json.dumps(b,indent=2))
