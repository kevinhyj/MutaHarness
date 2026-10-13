"""Label-free KRAS abundance prior; all coefficients are fixed, not fitted.

No assay outcomes are available. This program makes relative predictions only.
"""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import structure_tools

ROOT = Path(__file__).resolve().parent
# Fauchere-Pliska side-chain transfer hydrophobicity: unlike Kyte-Doolittle,
# aromatic side chains retain their hydrophobic character in a folding prior.
HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY', [.31,1.54,-.77,-.64,1.79,0.0,.13,1.80,-.99,1.70,1.23,-.60,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
VOL = dict(zip('ACDEFGHIKLMNPQRSTVWY', [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHARGE = {a: (1 if a in 'KR' else -1 if a in 'DE' else 0) for a in HYDRO}
AA_GROUPS = ['AVLIM', 'FWY', 'KR', 'DE', 'NQ', 'ST']
# These sites lie in the nucleotide pocket or commonly encode signaling
# specificity. Selection there is especially confounded with protein function.
FUNCTION_SITES = {12,13,16,17,35,37,38,40,41,57,58,59,60,61,116,117,119,146}
POCKET_SITES = {12,13,16,17,35,57,58,59,60,61,116,117,119,146}

def read(name):
    return json.loads((ROOT / name).read_text())

def load_inputs():
    case, manifest, msa, structure = [read(f) for f in ['case.json','evidence_manifest.json','msa_evidence.json','structure_evidence.json']]
    wt = case['reference_sequence']
    wt_hash = hashlib.sha256(wt.encode()).hexdigest()
    assert wt_hash == msa['reference_sequences_sha256']['__single__'] == structure['reference_sequence_sha256']
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    checks = {}
    for name, meta in manifest['files'].items():
        if isinstance(meta, dict) and 'sha256' in meta:
            path = ROOT / name
            checks[name] = 'absent' if not path.exists() else hashlib.sha256(path.read_bytes()).hexdigest()
            assert checks[name] in ['absent', meta['sha256']], name
    qids = [q['id'] for q in case['query']]
    assert len(qids) == 250 and len(set(qids)) == 250
    assert set(qids) == {f'M{i:03d}' for i in range(1,251)}
    assert set(qids) == {q['id'] for q in msa['candidates']} == {q['id'] for q in structure['candidates']}
    assert structure['raw_coordinate_evidence']['reference_sequence_sha256'] == wt_hash
    assert structure['raw_coordinate_evidence']['raw_source']['sha256'] == manifest['files']['raw_evidence/RASK_HUMAN.pdb']['sha256']
    profile = read('raw_evidence/msa_profile.json')
    assert profile['msa_sha256'] == manifest['files']['raw_evidence/RASK_HUMAN_2023-08-07_b03.a2m']['sha256']
    for row in case['evidence']['sites']:
        p = row['position']; upstream = profile['positions'][str(p)]
        assert row['WT'] == wt[p-1] == upstream['wt']
        assert row['msa']['frequencies'] == upstream['frequency']
        assert row['msa']['non_gap_depth'] == upstream['non_gap_effective_depth']
    # Preserve lowercase insertion residues when checking the WT alignment row.
    with (ROOT/'raw_evidence/RASK_HUMAN_2023-08-07_b03.a2m').open() as handle:
        assert handle.readline().startswith('>')
        aligned = []
        for line in handle:
            if line.startswith('>'): break
            aligned.append(line.strip())
    assert ''.join(aligned).replace('-','').replace('.','').upper() == wt
    for q in case['query']:
        expected = list(wt)
        changes = []
        for term in q['mutation'].split(':'):
            match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', term)
            assert match, term
            w, p, a = match.groups(); p = int(p)
            assert wt[p-1] == w and a in HYDRO
            expected[p-1] = a
            changes.append((p,w,a))
        assert len(changes) == q['mutation_order']
        assert len(set(p for p,w,a in changes)) == len(changes)
        assert ''.join(expected) == q['sequence'], q['id']
        q['_changes'] = changes
    return case, manifest, msa, structure, checks

def structural_context(case, structure):
    """Observed WT contact flags only; no mutant coordinates or energies."""
    residues = structure_tools._read_pdb(ROOT/'raw_evidence/RASK_HUMAN.pdb')
    protein = [r for r in residues if r['protein']]
    assert {r['pdb_chain'] for r in protein} == {'A'}
    assert ''.join(r['WT'] for r in protein) == case['reference_sequence']
    assert len(residues) == len(protein), 'Unexpected context: retain partners before proceeding'
    raw_sites = structure['raw_coordinate_evidence']['sites']
    bypos = {}
    for site, residue in zip(raw_sites,protein):
        assert site['position'] == residue['pdb_residue_number']
        assert site['WT'] == residue['WT']
        assert site['pdb_chain'] == residue['pdb_chain']
        assert site['backbone_complete'] and site['sidechain_complete']
        assert np.allclose(site['CA_coordinate_A'],residue['atoms']['CA']['coord'])
        bypos[site['position']] = residue
    charged_atoms = {'K':['NZ'],'R':['NE','NH1','NH2'],'D':['OD1','OD2'],'E':['OE1','OE2']}
    result = {}
    for p, residue in bypos.items():
        atoms = residue['atoms']; w = residue['WT']
        donor, salt = [], []
        for j, other in bypos.items():
            if abs(j-p) <= 2: continue
            distance = float(np.linalg.norm(np.array(atoms['N']['coord'])-np.array(other['atoms']['O']['coord'])))
            if distance <= 3.5:
                donor.append({'position':j,'distance_A':distance})
            b = other['WT']
            if w not in charged_atoms or b not in charged_atoms or CHARGE[w] == CHARGE[b]: continue
            distance = min(float(np.linalg.norm(np.array(atoms[u]['coord'])-np.array(other['atoms'][v]['coord'])))
                           for u in charged_atoms[w] for v in charged_atoms[b])
            if distance <= 4.0:
                salt.append({'position':j,'WT':b,'distance_A':distance})
        result[p] = {'backbone_donor_distance_contacts':donor,'opposite_charge_distance_contacts':salt}
    return result

def site_penalty(p, w, a, raw, obs, term, context):
    if w == a:
        return 0.0, {}
    rsa = raw.get('isolated_chain_wt_RSA')
    conf = raw.get('source_asserted_AF2_pLDDT')
    if rsa is None:
        burial, reliability = 0.3, 0.4
    else:
        burial = (1.0-min(1.0,max(0.0,rsa))) ** 1.5
        reliability = min(1.0,max(0.0,((conf or 70.0)-45.0)/40.0))
    hyd = max(0.0,HYDRO[w]-HYDRO[a])
    conservative = any(w in group and a in group for group in AA_GROUPS)
    hyd_factor = 0.45 if w in 'FWY' and a in 'FWY' else 0.65 if {w,a} <= set('CST') else 1.0
    # Both cavities and overpacking can destabilize; these are volume priors,
    # never claims of observed mutant clashes or repacking.
    volume_change = (VOL[a]-VOL[w])/100.0
    physical = burial*(0.9*hyd_factor*hyd + (1.2 if volume_change < 0 else 1.4)*abs(volume_change))
    if CHARGE[a] and not CHARGE[w]:
        physical += 1.0*burial**1.5
    if CHARGE[w] and not CHARGE[a]:
        physical += 0.4*burial*(1.0+0.6*bool(context['opposite_charge_distance_contacts']))
    if CHARGE[w]*CHARGE[a] == -1:
        physical += burial*(0.6+0.8*bool(context['opposite_charge_distance_contacts']))
    if w in 'NQ' and a in 'AVLIMFWY':
        physical += 0.35*burial
    # Surface hydrophobicization is a weak solubility/aggregation prior.
    if rsa is not None:
        physical += 0.20*min(1.0,rsa)*max(0.0,HYDRO[a]-HYDRO[w])
    if p in POCKET_SITES:
        # An apo AF2 snapshot can close over a nucleotide pocket. Ligand-site
        # RSA is less direct evidence of protein-core burial.
        physical *= 0.65
    if p == 40:
        physical *= 0.70
    phi, psi = obs.get('phi_deg'), obs.get('psi_deg')
    alpha = phi is not None and psi is not None and -100 < phi < -35 and -75 < psi < -5
    beta = phi is not None and psi is not None and -170 < phi < -70 and psi > 65
    backbone = 0.0
    if a == 'P' and w != 'P':
        donor = bool(context['backbone_donor_distance_contacts'])
        backbone += 0.10 + (1.15 if donor else 0.0)
        if donor and alpha: backbone += 0.80
        if donor and beta: backbone += 0.60
        if phi is not None:
            if phi > 0: backbone += 2.0
            elif phi < -90: backbone += min(1.0,(-90-phi)/50.0)
    if a == 'G' and w not in 'GP':
        backbone += (0.60 if alpha else 0.15)*(0.4+0.6*burial)
    if w == 'G' and phi is not None and phi > 35 and a != 'G':
        backbone += 0.40 if a in 'DNS' else 1.0
    elif phi is not None and phi > 35 and w != 'G' and a in 'IVTFWY':
        backbone += 0.30
    physical += backbone
    # Nucleotide occupancy stabilizes the fold. Loss of direct coordination is
    # only a moderate prior; affinity and occupancy are not measured here.
    ligand = 0.0
    if p == 16 and a != 'K': ligand = 0.25 if a == 'R' else 0.75
    if p == 17 and a != 'S': ligand = 0.30 if a == 'T' else 0.40
    if p == 57 and a != 'D': ligand = 0.45
    if p == 116 and a != 'N': ligand = 0.45
    if p == 119 and a != 'D': ligand = 0.45
    physical += ligand
    physical *= reliability
    # Natural selection combines folding and function; surface conservation is weak
    # abundance evidence. Zero homolog depth is explicitly absent evidence.
    lr = term.get('sum_log_frequency_ratio')
    depth = term.get('effective_non_gap_depth')
    evo = 0.0
    if lr is not None and depth is not None and depth > 0:
        support = depth/(depth+20.0)
        evo = max(-1.5,min(6.0,-lr))*(0.10+0.22*burial)*support
        if conservative: evo *= 0.55
        if p in FUNCTION_SITES: evo *= 0.35
    return physical+evo, {'rsa':rsa,'burial':burial,'physical':physical,'backbone_prior':backbone,'ligand_prior':ligand,'evolutionary':evo,'msa_available':lr is not None and depth is not None and depth > 0}

def main():
    case, manifest, msa, structure, checks = load_inputs()
    context = structural_context(case,structure)
    raw = {x['position']:x for x in structure['raw_coordinate_evidence']['sites']}
    st = {x['id']:x for x in structure['candidates']}
    ms = {x['id']:x for x in msa['candidates']}
    results, details = [], []
    for q in case['query']:
        sites = {x['position']:x for x in st[q['id']]['sites']}
        terms = {x['position']:x for x in ms[q['id']]['substitution_terms']}
        penalties, features = [], []
        for p,w,a in q['_changes']:
            assert sites[p]['WT'] == terms[p]['WT'] == w
            assert sites[p]['ALT'] == terms[p]['ALT'] == a
            penalty, feature = site_penalty(p,w,a,raw[p],sites[p]['wt_observations'],terms[p],context[p])
            penalties.append(penalty)
            features.append(dict(position=p,WT=w,ALT=a,penalty=penalty,**feature))
        total = sum(penalties)
        # A single monotonic global mapping; no blockwise normalization or WT label.
        score = math.tanh((0.80-total)/3.20)
        assert math.isfinite(score) and -1 <= score <= 1
        results.append({'id':q['id'],'score':score})
        details.append({'id':q['id'],'mutation':q['mutation'],'total_penalty':total,'sites':features})
    summary = 'Label-free KRAS abundance prior combining WT burial, side-chain chemistry/volume, backbone constraints and a weak homolog-frequency prior, attenuated at functional sites; additive mutation effects on one global scale. All identities, source hashes and residue mappings verified; only chain A is supplied, with no fixed partners. Zero-depth MSA sites are absent evidence; 117 candidates have incomplete evolutionary coverage. Scores are uncalibrated; apo WT geometry, nucleotide occupancy and double-mutant epistasis remain uncertain.'
    output = {'predictions':results,'summary':summary}
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
    (ROOT/'prediction_details.json').write_text(json.dumps({'source_checks':checks,'candidates':details},indent=2)+'\n')
    (ROOT/'structural_context.json').write_text(json.dumps({'reference_sequence_sha256':structure['reference_sequence_sha256'],'pdb_sha256':structure['raw_coordinate_evidence']['raw_source']['sha256'],'limitations':'WT distance contacts only; donor and salt contact flags lack hydrogen-bond angle/energy validation. No mutant structure.','sites':context},indent=2)+'\n')
    print('Saved 250 complete finite predictions; range',min(x['score'] for x in results),max(x['score'] for x in results))

if __name__ == '__main__':
    main()
