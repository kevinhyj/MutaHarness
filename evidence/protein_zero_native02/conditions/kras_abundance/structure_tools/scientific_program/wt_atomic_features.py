"""Deterministic WT-only descriptors; no mutant coordinates or energies."""
import json
from pathlib import Path
import numpy as np
import structure_tools as st

ROOT = Path(__file__).resolve().parent
BACKBONE = {'N','CA','C','O','OXT'}
DONORS = {'K':{'NZ'},'R':{'NE','NH1','NH2'},'N':{'ND2'},'Q':{'NE2'},
          'S':{'OG'},'T':{'OG1'},'Y':{'OH'},'W':{'NE1'},'H':{'ND1','NE2'}}
ACCEPTORS = {'D':{'OD1','OD2'},'E':{'OE1','OE2'},'N':{'OD1'},'Q':{'OE1'},
             'S':{'OG'},'T':{'OG1'},'Y':{'OH'},'H':{'ND1','NE2'}}

def glycine_cb_open_fraction(r,atoms):
    """WT solvent-space probe in the canonical CB direction, not a mutant model.

    Glycine's backbone ASA cannot stand in for the burial of an added side chain.
    A carbon-sized probe measures the openness of the initial CB direction only.
    """
    n,ca,c = [np.array(r['atoms'][name]['coord']) for name in ['N','CA','C']]
    b=ca-n;d=c-ca
    center=ca-.58273431*np.cross(b,d)+.56802827*b-.54067466*d
    points=200;z=1-(np.arange(points)+.5)*2/points
    angle=np.arange(points)*(np.pi*(3-np.sqrt(5)))
    radial=np.sqrt(1-z*z)
    surface=center+3.1*np.column_stack([np.cos(angle)*radial,np.sin(angle)*radial,z])
    def exposed(obstacles):
        mask=np.ones(points,dtype=bool)
        for atom in obstacles:
            if atom['element']=='H':continue
            radius=st._RADII_A[atom['element']]+1.4
            if np.linalg.norm(np.array(atom['coord'])-center)<radius+3.1:
                mask &= np.sum((surface-atom['coord'])**2,axis=1)>radius**2
        return int(mask.sum())
    denominator=exposed(r['atoms'].values())
    return exposed(atoms)/denominator if denominator else None

def generate():
    residues = st._read_pdb(ROOT/'raw_evidence/RASK_HUMAN.pdb')
    atoms, owners = [], []
    for ri,r in enumerate(residues):
        for a in r['atoms'].values():
            if a['element'] != 'H':
                atoms.append(a);owners.append(ri)
    # Use matching heavy-atom SASA algorithms for numerator and denominator,
    # avoiding the supplied all-atom ASA/theoretical max-area incompatibility.
    full_asa = st._sasa_arrays(np,atoms,200,1.4)
    assert full_asa is not None
    features = {}
    for ri,r in enumerate(residues):
        indices = [i for i,owner in enumerate(owners) if owner == ri]
        own = [atoms[i] for i in indices]
        isolated = st._sasa_arrays(np,own,200,1.4)
        side = [k for k,a in enumerate(own) if a['name'] not in BACKBONE]
        if not side:
            exposure = None
        else:
            exposure = sum(full_asa[indices[k]] for k in side)/sum(isolated[k] for k in side)
        contact_neighbors = []
        polar_contacts = []
        for rj,s in enumerate(residues):
            if rj==ri:continue
            for ai in side:
                a=own[ai]
                for b in s['atoms'].values():
                    if b['element']=='H':continue
                    dist = float(np.linalg.norm(np.array(a['coord'])-b['coord']))
                    if abs(rj-ri)>2 and dist<4.5:
                        contact_neighbors.append(rj+1)
                    a_donor=a['name'] in DONORS.get(r['WT'],set())
                    a_acceptor=a['name'] in ACCEPTORS.get(r['WT'],set())
                    b_donor=b['name'] in DONORS.get(s['WT'],set()) or (b['name']=='N' and s['WT']!='P')
                    b_acceptor=b['name'] in ACCEPTORS.get(s['WT'],set()) or b['name'] in {'O','OXT'}
                    if abs(rj-ri)>1 and dist<3.5 and ((a_donor and b_acceptor) or (a_acceptor and b_donor)):
                        polar_contacts.append({'position':rj+1,'WT':s['WT'],'atoms':[a['name'],b['name']],'distance_A':dist})
        # Existing backbone NH...O contacts support secondary structure penalties.
        nh_hbonds=[]
        if 'N' in r['atoms'] and 'H' in r['atoms']:
            n=np.array(r['atoms']['N']['coord']);h=np.array(r['atoms']['H']['coord'])
            for rj,s in enumerate(residues):
                if abs(rj-ri)<=1 or 'O' not in s['atoms']:continue
                o=np.array(s['atoms']['O']['coord'])
                no=float(np.linalg.norm(n-o));ho=float(np.linalg.norm(h-o))
                if no<3.5 and ho<2.6:
                    cosine=float(np.dot(n-h,o-h)/(np.linalg.norm(n-h)*np.linalg.norm(o-h)))
                    if cosine<-.5:
                        nh_hbonds.append({'acceptor_position':rj+1,'N_O_distance_A':no})
        features[ri+1]={'WT':r['WT'],'sidechain_exposure_matched_heavy_SASA':exposure,
                        'glycine_CB_direction_open_fraction':glycine_cb_open_fraction(r,atoms) if r['WT']=='G' else None,
                        'nonlocal_sidechain_contact_residues':sorted(set(contact_neighbors)),
                        'sidechain_polar_contacts':polar_contacts,'backbone_NH_Hbonds':nh_hbonds}
    (ROOT/'wt_atomic_features.json').write_text(json.dumps({
        'pdb_sha256':__import__('hashlib').sha256((ROOT/'raw_evidence/RASK_HUMAN.pdb').read_bytes()).hexdigest(),
        'coordinate_mapping':'Exact whole-chain sequence; 188 residues, chain A.',
        'SASA':'Heavy atoms, Shrake-Rupley 200 points, 1.4 A probe; full vs isolated WT residue.',
        'limitations':'WT contacts/exposure only, not mutant repacking or binding energies.',
        'sites':features},indent=2)+'\n')
    return features

if __name__=='__main__':
    f=generate()
    for p in [2,6,12,14,17,19,24,35,40,63,66,71,72,78,82,101,112,116,117,118,133,137,146,149,159]:
        s=f[p];e=s['sidechain_exposure_matched_heavy_SASA']
        print(p,s['WT'],None if e is None else round(e,3),len(s['nonlocal_sidechain_contact_residues']),len(s['sidechain_polar_contacts']),s['backbone_NH_Hbonds'])
