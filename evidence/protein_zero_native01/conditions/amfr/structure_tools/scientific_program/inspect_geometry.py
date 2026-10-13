"""Verify supplied WT geometry and derive an explicitly heuristic contact prior."""
import json
from pathlib import Path
import numpy as np
from structure_tools import _read_pdb, _sasa_arrays

ROOT = Path(__file__).resolve().parent
ACCEPTORS = {'D':{'OD1','OD2'},'E':{'OE1','OE2'},'N':{'OD1'},
             'Q':{'OE1'},'S':{'OG'},'T':{'OG1'},'Y':{'OH'},'H':{'ND1','NE2'}}
BACKBONE = {'N','CA','C','O','OXT'}


def geometry():
    case = json.loads((ROOT/'case.json').read_text())
    residues = _read_pdb(ROOT/'raw_evidence/AMFR_HUMAN.pdb')
    assert ''.join(r['WT'] for r in residues) == case['reference_sequence']
    assert len({r['pdb_chain'] for r in residues}) == 1
    atoms = [a for r in residues for a in r['atoms'].values()]
    asas = _sasa_arrays(np, atoms, points=300, probe=1.4)
    cursor = 0
    for pos,r in enumerate(residues,1):
        r['position'] = pos
        r['atom_asa'] = {name:float(asas[cursor+j]) for j,name in enumerate(r['atoms'])}
        cursor += len(r['atoms'])
    donors, acceptors = [], []
    for r in residues:
        at = r['atoms']
        for name,a in at.items():
            if a['element'] not in {'N','O'}: continue
            hs = [h for h in at.values() if h['element']=='H' and np.linalg.norm(np.array(h['coord'])-a['coord']) < 1.25]
            if hs: donors.append((r,name,a,hs))
            if name == 'O' or name in ACCEPTORS.get(r['WT'],set()):
                if r['WT']=='H' and hs:continue
                acceptors.append((r,name,a))
    hb = {}
    bb = []
    for dr,dn,d,hs in donors:
        for ar,an,a in acceptors:
            if dr['position'] == ar['position']: continue
            dist = np.linalg.norm(np.array(d['coord'])-a['coord'])
            if dist > 3.6:continue
            best = None
            for h in hs:
                vh = np.array(h['coord']); v1=np.array(d['coord'])-vh; v2=np.array(a['coord'])-vh
                angle = float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1))))
                ha = float(np.linalg.norm(v2))
                if ha < 2.6 and angle > 110 and (best is None or ha < best[0]):best=(ha,angle)
            if best is None:continue
            edge={'donor_position':dr['position'],'donor_WT':dr['WT'],'donor_atom':dn,
                  'acceptor_position':ar['position'],'acceptor_WT':ar['WT'],'acceptor_atom':an,
                  'donor_sidechain':dn not in BACKBONE,'acceptor_sidechain':an not in BACKBONE,
                  'distance_A':float(dist),'H_acceptor_distance_A':best[0],'DHA_angle_deg':best[1],
                  'donor_atom_SASA':dr['atom_asa'][dn],'acceptor_atom_SASA':ar['atom_asa'][an]}
            if dn in BACKBONE and an in BACKBONE:
                bb.append(edge)
            else:
                # Merge bifurcated contacts from a single donor to one residue;
                # counting both carboxylate oxygens would overstate its energy.
                key=(dr['position'],dn,ar['position'])
                if key not in hb or best[0] < hb[key]['H_acceptor_distance_A']:hb[key]=edge
    helices={}
    summaries={s['position']:s.get('wt_structure',{}) for s in case['evidence']['sites']}
    alpha_edges=[e for e in bb if e['donor_position']-e['acceptor_position']==4]
    for r in residues:
        p=r['position']; summary=summaries[p]
        phi,psi=summary.get('phi_deg'),summary.get('psi_deg')
        count=sum(e['acceptor_position']<=p<=e['donor_position'] for e in alpha_edges)
        if r['WT']=='P' or (phi is not None and psi is not None and not (-105<phi<-30 and -80<psi<10)):
            helices[p]=0.
        elif count>=2:helices[p]=1.
        elif count==1:helices[p]=0.65
        else:helices[p]=0.
    sidechain_pairs=[]
    for i,r in enumerate(residues):
        for s in residues[i+1:]:
            xs=[a['coord'] for n,a in r['atoms'].items() if a['element']!='H' and n not in BACKBONE]
            ys=[a['coord'] for n,a in s['atoms'].items() if a['element']!='H' and n not in BACKBONE]
            if not xs or not ys:continue
            ds=np.linalg.norm(np.array(xs)[:,None,:]-np.array(ys)[None,:,:],axis=-1)
            if ds.min()<5.0:sidechain_pairs.append({'left':r['position'],'right':s['position'],'distance_A':float(ds.min())})
    out={'backbone_hydrogen_bonds':bb,'sidechain_hydrogen_bonds':list(hb.values()),
         'helix_compatibility':helices,'sidechain_contact_pairs':sidechain_pairs,
         'limitations':['Contacts describe static WT geometry, not mutant energies or validated epistasis.',
                        'Explicit source hydrogens are modeled; protonation and H-bond angles are uncertain.',
                        'Atom SASA is recomputed only for WT, using 300 sphere points.']}
    return out


if __name__ == '__main__':
    out=geometry()
    (ROOT/'geometry_features.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Helix compatibility:',out['helix_compatibility'])
    print('WT sidechain-involving hydrogen bonds:')
    for e in out['sidechain_hydrogen_bonds']:
        print(e['donor_position'],e['donor_WT'],e['donor_atom'],'->',e['acceptor_position'],e['acceptor_WT'],e['acceptor_atom'],round(e['distance_A'],2),round(e['DHA_angle_deg']))
