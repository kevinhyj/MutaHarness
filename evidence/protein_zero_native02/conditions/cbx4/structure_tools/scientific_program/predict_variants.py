"""Unsupervised stability prior; no assay labels or external data are used."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import structure_tools as st

ROOT = Path(__file__).resolve().parent
HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                 [1.0,1.5,-0.7,-0.7,2.6,0.0,0.4,2.4,-0.7,2.4,2.0,-0.4,1.0,-0.4,-0.7,0.0,0.35,2.0,3.0,2.3]))
VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                  [67,86,91,109,135,48,118,124,135,124,124,96,90,114,148,73,93,105,163,141]))
CHARGE = {a: (1 if a in 'KR' else -1 if a in 'DE' else 0) for a in HYDRO}
BACKBONE = {'N','CA','C','O','OXT'}
HELIX_COST = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                      [0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.2,.39,.21,.50,.66,.61,.53,.53]))
BETA_COST = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                     [.4,.2,.8,.5,.1,1,.3,0,.3,.3,.4,.6,3.2,.4,.3,.4,.1,0,.2,.1]))


def atomic_features(sites):
    """WT-only surface and directional polar contacts, without mutant coordinates.

    Sidechain exposure is computed relative to the same isolated WT residue,
    with heavy atoms in both calculations. This avoids mixing explicit-H atom
    areas with the heavy-atom theoretical maximum-ASA normalization.
    """
    residues = st._read_pdb(ROOT/'raw_evidence/CBX4_HUMAN.pdb')
    assert {r['pdb_chain'] for r in residues} == {'A'}
    assert [r['pdb_residue_number'] for r in residues] == list(range(1,51))
    all_atoms, heavy_atoms = [], []
    for r in residues:
        assert r['WT'] == sites[r['pdb_residue_number']]['WT']
        for a in r['atoms'].values():
            a['position'] = r['pdb_residue_number']
            a['WT'] = r['WT']
            a['xyz'] = np.array(a['coord'])
            all_atoms.append(a)
            if a['element'] != 'H':
                heavy_atoms.append(a)
    areas = st._sasa_arrays(np,heavy_atoms,300,1.4)
    complex_area = {id(a):v for a,v in zip(heavy_atoms,areas)}
    features, exposure = {}, {}
    for r in residues:
        heavy = [a for a in r['atoms'].values() if a['element'] != 'H']
        areas = st._sasa_arrays(np,heavy,300,1.4)
        iso_area = {id(a):v for a,v in zip(heavy,areas)}
        for a in heavy:
            exposure[(a['position'],a['name'])] = min(1.0,complex_area[id(a)]/max(1e-6,iso_area[id(a)]))
        sc = [a for a in heavy if a['name'] not in BACKBONE]
        fraction = sum(complex_area[id(a)] for a in sc)/sum(iso_area[id(a)] for a in sc) if sc else None
        p = r['pdb_residue_number']
        rsa = sites[p]['isolated_chain_wt_RSA']
        b_rsa = (1-min(1,max(0,rsa)))**1.4
        b_sc = (1-min(1,fraction))**1.4 if fraction is not None else b_rsa
        features[p] = {'sidechain_exposure':fraction,'buried':0.5*b_rsa+0.5*b_sc}
    donors = [a for a in heavy_atoms if a['element']=='N' or a['name'] in {'OG','OG1','OH'}]
    acceptors = [a for a in heavy_atoms if a['element']=='O']
    hbonds = []
    for donor in donors:
        hs = [h for h in all_atoms if h['position']==donor['position'] and h['element']=='H'
              and np.linalg.norm(h['xyz']-donor['xyz'])<1.25]
        for acc in acceptors:
            if donor['position']==acc['position']:
                continue
            distance = np.linalg.norm(donor['xyz']-acc['xyz'])
            if distance>3.6:
                continue
            angle = max([np.degrees(np.arccos(np.clip(np.dot(donor['xyz']-h['xyz'],acc['xyz']-h['xyz'])/
                         np.linalg.norm(donor['xyz']-h['xyz'])/np.linalg.norm(acc['xyz']-h['xyz']),-1,1)))
                         for h in hs], default=0)
            if angle<120:
                continue
            if donor['name'] in BACKBONE and acc['name'] in BACKBONE:
                # Backbone H bonds only change in this prior upon Pro introduction.
                strength = 0.7
            else:
                buried = 1-0.5*(exposure[(donor['position'],donor['name'])]+exposure[(acc['position'],acc['name'])])
                strength = (0.35+0.95*buried)*min(1.0,max(0.0,(3.8-distance)/0.8))
            hbonds.append({'donor':(donor['position'],donor['WT'],donor['name']),
                           'acceptor':(acc['position'],acc['WT'],acc['name']),
                           'distance':float(distance),'angle':float(angle),'strength':float(strength)})
    salts = {}
    for bond in hbonds:
        p,w,a = bond['donor']; q,v,b = bond['acceptor']
        if a not in BACKBONE and b not in BACKBONE and CHARGE[w]*CHARGE[v]==-1:
            pair = tuple(sorted((p,q)))
            buried = 1-0.5*(exposure[(p,a)]+exposure[(q,b)])
            salts[pair] = max(salts.get(pair,0),1.4+1.8*buried)
    # Merge multiple atom-level H bonds forming the same ionic residue pair.
    hbonds = [b for b in hbonds if not (b['donor'][2] not in BACKBONE and b['acceptor'][2] not in BACKBONE
               and tuple(sorted((b['donor'][0],b['acceptor'][0]))) in salts)]
    return features,hbonds,salts


def polar_retention(wt,alt,atom,role):
    if wt==alt:
        return 1.0
    if atom in BACKBONE:
        return 0.0 if role=='donor' and alt=='P' else 1.0
    if role=='donor':
        if wt in 'KR':
            return 0.85 if alt in 'KR' else 0.55 if alt in 'NQ' else 0.4 if alt in 'HSTY' else 0.0
        if wt=='W':
            return 0.65 if alt=='Y' else 0.4 if alt in 'HNQST' else 0.0
        if wt in 'ST':
            return 0.9 if alt in 'ST' else 0.5 if alt in 'NQ' else 0.3 if alt in 'YHKR' else 0.0
        return 0.85 if alt in 'NQ' else 0.55 if alt in 'STYHKR' else 0.0
    if wt in 'DE':
        return 0.8 if alt in 'DE' else 0.7 if alt in 'NQ' else 0.35 if alt in 'STYH' else 0.0
    if wt in 'ST':
        return 0.9 if alt in 'ST' else 0.6 if alt in 'NQDE' else 0.35 if alt in 'YH' else 0.0
    return 0.85 if alt in 'NQ' else 0.7 if alt in 'DE' else 0.4 if alt in 'STYH' else 0.0


def polar_penalty(mutations,hbonds,salts,sites):
    alternatives = {p:alt for p,wt,alt in mutations}
    total = 0.0
    for bond in hbonds:
        p,w,a = bond['donor']; q,v,b = bond['acceptor']
        if p not in alternatives and q not in alternatives:
            continue
        retain = polar_retention(w,alternatives.get(p,w),a,'donor') * polar_retention(v,alternatives.get(q,v),b,'acceptor')
        total += bond['strength']*(1-retain)
    for (p,q),strength in salts.items():
        if p not in alternatives and q not in alternatives:
            continue
        w,v = sites[p]['WT'],sites[q]['WT']
        a,b = alternatives.get(p,w),alternatives.get(q,v)
        if CHARGE[a]*CHARGE[b]==-1:
            # Reversed ionic pairs can partly retain attraction, but geometry is unknown.
            retain = 0.65 if CHARGE[a]!=CHARGE[w] else 0.85
            length_match = math.exp(-abs((VOLUME[a]+VOLUME[b])-(VOLUME[w]+VOLUME[v]))/110)
            retain *= length_match
        elif CHARGE[a]*CHARGE[b]==1:
            retain = -0.5
        else:
            donor_w,donor_a,acceptor_w,acceptor_a = (w,a,v,b) if CHARGE[w]>0 else (v,b,w,a)
            retain = 0.45*polar_retention(donor_w,donor_a,'SC','donor')*polar_retention(acceptor_w,acceptor_a,'SC','acceptor')
        total += strength*(1-retain)
    return float(total)


def load_and_validate():
    case = json.loads((ROOT/'case.json').read_text())
    manifest = json.loads((ROOT/'evidence_manifest.json').read_text())
    evidence = json.loads((ROOT/'structure_evidence.json').read_text())
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better']
    ref = case['reference_sequence']
    assert hashlib.sha256(ref.encode()).hexdigest() == evidence['reference_sequence_sha256']
    for path, info in manifest['files'].items():
        if isinstance(info, dict) and info.get('sha256'):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == info['sha256'], path
    # This case supplies a single isolated chain and no fixed binding partners.
    assert not case.get('reference_chains') and not case['context'].get('fixed_partners')
    raw = evidence['raw_coordinate_evidence']
    assert raw['raw_source']['matches_declared_source_sha256']
    assert raw['mapping'] == [{'chain': None, 'pdb_chain': 'A', 'method': 'exact_whole_chain_sequence',
                              'reference_length': 50, 'mapped_observed_residues': 50, 'unobserved_positions': []}]
    sites = {s['position']: s for s in raw['sites']}
    summary = {s['position']: s['wt_structure'] for s in case['evidence']['sites']}
    assert len(sites) == len(ref)
    for p, s in sites.items():
        assert s['WT'] == ref[p-1] and s['coordinate_present']
        assert s['backbone_complete'] and s['sidechain_complete']
    changes = {}
    for q in case['query']:
        seq = list(ref)
        muts = []
        for token in q['mutation'].split(':'):
            match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', token)
            assert match, token
            wt, p, alt = match.group(1), int(match.group(2)), match.group(3)
            assert ref[p-1] == wt and alt in HYDRO
            seq[p-1] = alt
            muts.append((p, wt, alt))
        assert len({p for p, _, _ in muts}) == len(muts) == q['mutation_order']
        assert ''.join(seq) == q['sequence']
        changes[q['id']] = muts
    assert len(changes) == len(case['query']) == 250
    assert set(changes) == {f'M{i:03d}' for i in range(1,251)}
    return case, sites, summary, changes


def site_penalty(p, wt, alt, sites, summary, features):
    if wt == alt:
        return 0.0
    rsa = sites[p]['isolated_chain_wt_RSA']
    assert rsa is not None
    buried = features[p]['buried']
    exposed = features[p]['sidechain_exposure']
    exposed = min(1,rsa) if exposed is None else exposed
    w = summary[p]
    phi, psi = w.get('phi_deg'), w.get('psi_deg')
    # Native polar-to-hydrophobic substitutions cannot be assumed to achieve
    # the full favorable transfer free energy without demonstrated repacking.
    delta_hydro = HYDRO[wt]-HYDRO[alt]
    penalty = 0.15+0.25*buried+buried*(max(delta_hydro,0)-0.25*min(max(-delta_hydro,0),2))
    penalty += 0.12*exposed*max(-delta_hydro,0)
    delta_volume = VOLUME[alt]-VOLUME[wt]
    penalty += 0.020*buried*max(0,-delta_volume)
    penalty += 0.012*buried*max(0,delta_volume)+0.0006*buried**2*max(0,delta_volume)**2
    if CHARGE[alt] and not CHARGE[wt]:
        penalty += 1.7*buried
    # Weak native chemical compatibility prior, additional to explicit contacts.
    if (wt in 'NQST') != (alt in 'NQST'):
        penalty += 0.25*buried
    if phi is not None and psi is not None:
        strand = phi < -70 and psi > 65
        helix = -95 < phi < -35 and -75 < psi < -25
        if alt == 'P' and wt != 'P':
            penalty += 3.0 if strand else 2.8 if helix else 0.9
        if alt == 'G' and wt != 'G':
            penalty += 0.9 if strand or helix else 0.35
        if wt == 'G' and phi > 0:
            penalty += 1.7 if alt in 'NDS' else 3.0 if alt in 'TVI' else 2.7
        if phi > 0 and wt != 'G':
            penalty += -0.3 if alt in 'GNSD' else -0.1 if alt=='C' else 0.1
        if wt == 'P' and alt != 'P':
            penalty += 0.85
        if alt not in 'PG' and wt not in 'PG':
            if helix:
                penalty += 0.6*(HELIX_COST[alt]-HELIX_COST[wt])
            elif strand:
                penalty += 0.45*(BETA_COST[alt]-BETA_COST[wt])
    if wt in 'FWY' and alt not in 'FWY':
        penalty += 0.35*buried
    return float(penalty)


def packing_pair_penalty(mutations,sites,features):
    if len(mutations)!=2:
        return 0.0
    (p,w,a),(q,v,b) = mutations
    contacts = [c for c in sites[p]['heavy_atom_contacts'] if c['neighbor'].get('position')==q]
    if not contacts or contacts[0]['minimum_heavy_atom_distance_A']>4.0:
        return 0.0
    dv1,dv2 = VOLUME[a]-VOLUME[w],VOLUME[b]-VOLUME[v]
    buried = min(features[p]['buried'],features[q]['buried'])
    # Conservative partial volume compensation; no optimized mutant coordinates.
    if dv1*dv2<0:
        return -0.4*0.020*buried*min(abs(dv1),abs(dv2))
    if dv1>0 and dv2>0:
        return 0.00025*buried**2*dv1*dv2
    return 0.0


def run():
    case, sites, summary, changes = load_and_validate()
    features,hbonds,salts = atomic_features(sites)
    diagnostics = []
    predictions = []
    for q in case['query']:
        mutations = changes[q['id']]
        effects = [site_penalty(p,wt,alt,sites,summary,features) for p,wt,alt in mutations]
        polar = polar_penalty(mutations,hbonds,salts,sites)
        pair = packing_pair_penalty(mutations,sites,features)
        penalty = sum(effects)+polar+pair
        score = float(np.tanh((0.3-penalty)/3.5))
        assert math.isfinite(score) and -1 <= score <= 1
        predictions.append({'id':q['id'], 'score':round(score,7)})
        diagnostics.append({'id':q['id'], 'mutation':q['mutation'], 'site_penalties':effects,
                            'polar_contact_penalty':polar,'pair_packing_penalty':pair,'penalty':penalty})
    result = {'predictions':predictions, 'summary':
        'Zero-shot stability ranking for cDNA-display proteolysis using burial, packing, backbone preferences, '
        'directional WT hydrogen bonds and salt bridges; interacting double mutants share contact-loss penalties '
        'and receive conservative volume-compensation terms. Source hashes and exact 50/50-residue single-chain '
        'mapping verified; all mutation sequences checked; no fixed partners, MSA or measured outcomes. '
        'One global tanh scale, higher is better. Scores are uncalibrated priors; static WT geometry, mutant '
        'repacking, proteolysis-specific effects and epistasis leave substantial ranking uncertainty.'}
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (ROOT/'wt_model_features.json').write_text(json.dumps({'sites':features,'hydrogen_bonds':hbonds,
         'salt_bridges':[{'positions':list(p),'strength':s} for p,s in salts.items()]},indent=2)+'\n')
    print(json.dumps({'n':len(predictions),'min':min(x['score'] for x in predictions),
                      'max':max(x['score'] for x in predictions),'complete_ids':True}))


if __name__ == '__main__':
    run()
