"""Unsupervised YAP WW-domain binding ranking, using public WT evidence only.

Effects are weak physicochemical priors in arbitrary log-activity units, not
estimated kcal/mol. Folding and peptide recognition are modeled separately.
There is no outcome fitting, MSA inference, mutant repacking, or pairwise
epistasis inference. Source/reference hashes and every substitution are checked.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import structure_tools as st
from predict_baseline import CASE, verified_changes, save_predictions

# Standard BLOSUM62 is a general substitution prior, not a case-specific MSA.
AA = 'ARNDCQEGHILKMFPSTWYV'
B62 = np.array([
 [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
 [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
 [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
 [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
 [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
 [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
 [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
 [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
 [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
 [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
 [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
 [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
 [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
 [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
 [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
 [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
 [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
 [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
 [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
 [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
], dtype=float)
VOLUME = dict(zip(AA, [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,
                     166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
# Coarse nonpolar sidechain character; aromatic packing gets separate terms.
NONPOLAR = {a:0.0 for a in AA}
NONPOLAR.update({a:1.0 for a in 'IVLFWY'})
NONPOLAR.update({'M':0.9,'A':0.5,'C':0.55,'P':0.45,'T':0.15,'H':0.25})

def torsion(a,b,c,d):
    b0 = -(b-a)
    b1 = c-b
    b2 = d-c
    b1 = b1/np.linalg.norm(b1)
    v = b0-np.dot(b0,b1)*b1
    w = b2-np.dot(b2,b1)*b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))

def load_and_verify():
    manifest = json.loads(Path('evidence_manifest.json').read_text())
    evidence = json.loads(Path('structure_evidence.json').read_text())
    raw = evidence['raw_coordinate_evidence']
    sequence_hash = hashlib.sha256(CASE['reference_sequence'].encode()).hexdigest()
    assert sequence_hash == evidence['reference_sequence_sha256'] == raw['reference_sequence_sha256']
    source = Path('raw_evidence/1JMQ_hm.pdb')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == manifest['files'][str(source)]['sha256'] == raw['raw_source']['sha256']
    assert manifest['support_count'] == 0 and manifest['measurements_present'] is False
    assert manifest['allowed_modalities']['msa'] is False
    assert CASE['context']['higher_is_better'] is True
    assert np.allclose(B62, B62.T)
    # File-wide zero occupancy is a declared modeled-coordinate placeholder.
    residues = st._read_pdb(source, include_zero_occupancy=True)
    residues, mapping = st._map_raw_residues(st._prepare(CASE),residues,None)
    assert mapping == raw['mapping']
    atoms = {(r['chain'],r['position']):r for r in residues}
    sites = {(s['chain'],s['position']):s for s in raw['sites']}
    for key,site in sites.items():
        assert site['WT'] == CASE['reference_chains'][key[0]][key[1]-1]
        assert site['coordinate_present'] and site['backbone_complete'] and site['sidechain_complete']
        assert not site['missing_standard_heavy_atoms']
        assert np.allclose(atoms[key]['atoms']['CA']['coord'],site['CA_coordinate_A'])
    ev_candidates = {q['id']:q for q in evidence['candidates']}
    raw_candidates = {q['id']:q for q in raw['candidates']}
    for q in CASE['query']:
        assert ev_candidates[q['id']]['mutation'] == q['mutation']
        observed = [(m['chain'],m['position'],m['WT'],m['ALT'])
                    for m in raw_candidates[q['id']]['mutations']]
        assert observed == verified_changes(q)
    return raw,sites,atoms,source_hash,sequence_hash

def site_features(sites, atoms):
    features = {}
    for p in range(1,len(CASE['reference_sequence'])+1):
        r = atoms['A',p]
        s = sites['A',p]
        rsa = s['isolated_chain_wt_RSA']
        assert rsa is not None and math.isfinite(rsa)
        # RSA is observed WT exposure, never imputed mutant exposure.
        burial = max(0.0,1.0-min(1.0,rsa))
        coords = lambda pos,name: np.array(atoms['A',pos]['atoms'][name]['coord'],float)
        phi = None if p == 1 else torsion(coords(p-1,'C'),coords(p,'N'),coords(p,'CA'),coords(p,'C'))
        psi = None if p == len(CASE['reference_sequence']) else torsion(coords(p,'N'),coords(p,'CA'),coords(p,'C'),coords(p+1,'N'))
        donor_contact = False
        for (chain,pos),other in atoms.items():
            if chain == 'A' and abs(pos-p) <= 2:
                continue
            for name,atom in other['atoms'].items():
                if atom['element'] == 'O' and np.linalg.norm(coords(p,'N')-atom['coord']) < 3.3:
                    donor_contact = True
        features[p] = {'WT':s['WT'],'RSA':rsa,'burial':burial,'phi':phi,'psi':psi,
                       'backbone_N_close_to_nonlocal_O':donor_contact,
                       'partner_buried_SASA_A2':s['observed_atom_SASA_buried_by_context_A2']}
    return features

def effects(p,w,a,f):
    """Return folding-margin and recognition effects; higher is favorable."""
    b = f['burial']
    i,j = AA.index(w),AA.index(a)
    substitution_cost = (B62[i,i]+B62[j,j])/2-B62[i,j]
    fold = -0.055*substitution_cost*(0.35+0.65*b)
    fold -= 1.0*b*max(0.0,NONPOLAR[w]-NONPOLAR[a])
    fold -= 0.30*b*abs(VOLUME[a]-VOLUME[w])/100
    if a in 'DEKR' and w not in 'DEKR' and b > 0.6:
        fold -= 0.9*(b-0.6)/0.4
    if a == 'G' and w != 'G':
        fold -= 0.15+0.25*b  # unfolded-state entropy and lost packing
    if w == 'G':
        if f['phi'] is not None and f['phi'] > 0:
            fold -= 0.7 if a in 'ASC' else 1.5
        else:
            fold -= 0.2
    if a == 'P':
        phi = f['phi']
        fold -= 2.0 if phi is not None and phi > 0 else (
            1.25 if phi is not None and phi < -110 else (
            0.75 if phi is not None and phi < -85 else 0.2))
        if f['backbone_N_close_to_nonlocal_O']:
            fold -= 0.5
    if w == 'P':
        fold -= 0.25+0.5*b
    # WW-domain aromatic scaffold: first Trp and opposing Tyr/Phe.
    if p == 13:
        fold -= 0.55 if a in 'FY' else 2.8
    if p == 24:
        fold -= 0.7 if a == 'H' else (0.25 if a in 'FW' else 1.6)
    if p == 25:
        fold -= 0.05 if a == 'Y' else (0.9 if a in 'ILMV' else 1.8)
    if p == 27:
        # Buried Asn makes a characteristic WW backbone-polar network.
        fold -= {'D':0.7,'S':0.8,'H':0.75,'K':1.3}.get(a,0.8)
    if p == 35:
        fold -= 0.5 if a not in 'FWY' else 0.1
    if p == 37:
        # Asp sidechain is a modeled backbone acceptor, rather than a
        # generic exposed acidic residue. Downweighted tail reliability.
        fold -= {'E':0.15,'N':0.0,'H':0.35}.get(a,0.55)
    if p == 38:
        fold -= 0.45  # conserved C-terminal WW proline cap
    if p == 39 and a not in 'KRH':
        fold -= 0.35  # observed guanidinium/backbone polar contacts
    # Modest WW-specific surface/turn priors. No measured values are used.
    # Negatively charged turn residues can limit stability in this small fold.
    if p == 6:
        fold += {'E':0.0,'N':0.25,'Q':0.2,'A':0.3,'G':0.25,
                 'H':0.35,'R':0.55,'K':0.7,'V':-0.05,'Y':-0.05}.get(a,0.0)
    if p == 14:
        fold += {'D':0.0,'Q':0.45,'A':0.3,'G':0.35,'K':0.25}.get(a,0.0)
    if p == 30:
        fold += {'N':1.1,'Q':0.45,'A':0.15,'H':0.4,'K':0.25,
                 'R':0.4,'E':-0.1,'G':-0.1,'V':-0.1,'L':-0.2,'Y':-0.1}.get(a,0.0)
    if p == 31 and a in 'HKR':
        fold += 0.1  # loop adaptation is allowed; no rigid mutant geometry

    binding = 0.0
    # Only named peptide contacts/WW recognition chemistry affect binding.
    if p == 18:
        binding += {'S':-0.03,'I':0.02,'A':-0.06,'P':-0.15}.get(a,-0.1)
    if p == 19 and a == 'R':
        binding += 0.24  # weak exposed-loop electrostatic/recognition prior
    if p == 22:
        binding += {'H':0.0,'R':0.03,'L':-0.12,'P':-0.15}.get(a,0.0)
    if p == 24:
        binding -= 0.35 if a in 'HFW' else 0.8
    if p == 26:
        binding += {'M':0.13,'V':-0.15,'Q':-0.75,'R':-0.9}.get(a,-0.2)
    if p == 28:
        # His and Gln31 contact the fixed peptide Tyr hydroxyl.
        binding += {'Q':-0.25,'Y':-0.25,'R':-0.3,'L':-0.9,'P':-1.2}.get(a,-0.6)
    if p == 31:
        binding += {'R':0.65,'K':0.5,'H':0.28,'E':-0.15,'L':-0.65,'P':-0.75}.get(a,0.0)
    if p == 32:
        binding += {'S':-0.04,'A':-0.1,'V':-0.08,'I':-0.12,'N':-0.12,'P':-0.3}.get(a,-0.1)
    if p == 33:
        binding += {'S':-0.1,'N':-0.35,'P':-0.9}.get(a,-0.4)
    if p == 34:
        binding += {'S':-0.02,'A':-0.03,'R':-0.1,'P':-0.15}.get(a,-0.05)
    if p == 35:
        # Loss of the large proline-binding aromatic groove is severe.
        binding -= 0.55 if a in 'FY' else 2.4
    if p == 36:
        binding += {'H':0.18,'R':0.10,'K':0.08}.get(a,0.0)
    return float(fold),float(binding)

def log_folded(margin):
    return -float(np.logaddexp(0.0,-margin))

def main():
    raw,sites,atoms,source_hash,sequence_hash = load_and_verify()
    features = site_features(sites,atoms)
    predictions,details = [],[]
    # Weakly stable WW-domain prior. Folding gains saturate; loss of folding
    # is nonlinear. This is a mechanistic link, not measured epistasis.
    wt_margin = 0.25
    for q in CASE['query']:
        changes = verified_changes(q)
        singles = []
        for chain,p,w,a in changes:
            assert chain == 'A' and features[p]['WT'] == w
            fold,binding = effects(p,w,a,features[p])
            singles.append({'mutation':f'{chain}:{w}{p}{a}','fold_margin_effect':fold,'recognition_effect':binding})
        fold_effect = sum(s['fold_margin_effect'] for s in singles)
        binding_effect = sum(s['recognition_effect'] for s in singles)
        raw_score = log_folded(wt_margin+fold_effect)-log_folded(wt_margin)+binding_effect
        # Same absolute, monotone transform for all candidates; no blocks.
        score = round(math.tanh(raw_score/2.5),6)
        predictions.append({'id':q['id'],'score':score})
        details.append({'id':q['id'],'mutation':q['mutation'],'single_effects':singles,
                        'fold_effect':fold_effect,'binding_effect':binding_effect,'raw_score':raw_score})
    summary = ('Unsupervised YAP WW-domain binding prior combines substitution chemistry, WT burial/backbone constraints, aromatic scaffold preservation, and contacts with fixed peptide P; folding gains saturate separately from recognition effects. WT coordinate/reference hashes and exact A/P mapping verified; all residues and standard heavy atoms covered. No MSA or assay outcomes; missing summary fields remain absent. One global monotone scale, higher is better. Static modeled WT geometry cannot establish mutant relaxation or epistasis; rankings are uncertain.')
    save_predictions(predictions,'final_predictions.json',summary)
    Path('prediction_details.json').write_text(json.dumps({'site_features':features,'candidates':details},indent=2)+'\n')
    Path('model_rationale.json').write_text(json.dumps({
        'source_sha256':source_hash,'reference_sha256':sequence_hash,
        'revisions':[{'from':'initial checkpoint','to':'final mechanistic prior',
                      'reason':'Verified complete raw WT complex adds exposure, positive-phi Gly12 constraint, buried Asn27 and C-terminal polar networks, and atom-specific peptide contacts. Ile29 is exposed rather than a buried-core position. Separate saturating folding availability from fixed-partner recognition; retain only modest unsupervised WW surface/turn priors.'}],
        'no_fitted_coefficients':True,'no_measured_outcomes':True,'no_pairwise_epistasis_terms':True,
        'summary_fields_with_missing_RSA':250,'raw_coordinate_residues_covered':56,
        'standard_heavy_atoms_missing':0,'MSA_available':False,
        'limitations':['heuristic constants are not assay calibrated','modeled WT complex lacks mutant relaxation',
                       'source-supplied biological assembly and affinities are unverified',
                       'WT exposure/contact chemistry and general evolutionary substitutions are imperfect assay priors']},indent=2)+'\n')
    ordered = sorted(zip(predictions,details),key=lambda x:x[0]['score'],reverse=True)
    print('Verified predictions:',len(predictions),'range:',min(q['score'] for q in predictions),max(q['score'] for q in predictions))
    print('Highest / lowest predicted candidates (heuristic only):')
    for prediction,detail in ordered[:12]+ordered[-8:]:
        print(prediction['id'],prediction['score'],detail['mutation'],round(detail['fold_effect'],2),round(detail['binding_effect'],2))

if __name__ == '__main__':
    main()
