"""Unsupervised KRAS abundance prior; uses only this case's permitted WT evidence.

Scores are rankings, not calibrated DMS outcomes. All coefficients are fixed
biophysical heuristics; no labels, outside files, or parameter fitting are used.
"""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY', [.45,.65,-.2,-.2,1.,0.,.15,1.,-.25,1.,.9,-.1,.35,-.1,-.25,-.05,.15,1.,1.1,.8]))
VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY', [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
CHARGE = {'D':-1.,'E':-1.,'K':1.,'R':1.,'H':.15}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_inputs():
    case = json.loads((ROOT/'case.json').read_text())
    manifest = json.loads((ROOT/'evidence_manifest.json').read_text())
    evidence = json.loads((ROOT/'structure_evidence.json').read_text())
    assert not manifest['measurements_present'] and manifest['support_count'] == 0
    assert case['context']['higher_is_better'] is True
    refhash = hashlib.sha256(case['reference_sequence'].encode()).hexdigest()
    assert refhash == evidence['reference_sequence_sha256']
    for filename, spec in manifest['files'].items():
        if isinstance(spec, dict) and 'sha256' in spec:
            assert sha(ROOT/filename) == spec['sha256'], filename
    raw = evidence['raw_coordinate_evidence']
    assert raw['reference_sequence_sha256'] == refhash
    assert sha(ROOT/'raw_evidence/RASK_HUMAN.pdb') == raw['raw_source']['sha256']
    assert raw['mapping'][0]['method'] == 'exact_whole_chain_sequence'
    sites = {x['position']: x for x in raw['sites']}
    summaries = {x['position']:x['wt_structure'] for x in case['evidence']['sites']}
    for p, site in sites.items():
        assert site['WT'] == case['reference_sequence'][p-1]
    return case, manifest, evidence, sites, summaries

def changes(case, q):
    seq = list(case['reference_sequence'])
    muts = []
    for token in q['mutation'].split(':'):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', token)
        assert match, token
        w,p,a = match.groups(); p=int(p)
        assert seq[p-1] == w
        seq[p-1] = a
        muts.append((w,p,a))
    assert ''.join(seq) == q['sequence']
    assert len(muts) == q['mutation_order']
    assert len({m[1] for m in muts}) == len(muts)
    return muts

def site_penalty(w,p,a,sites,summaries,atomic=None):
    if w == a:
        return 0.
    s, t = sites.get(p,{}), summaries.get(p,{})
    features = (atomic or {}).get(p,{})
    rsa = features.get('sidechain_exposure_matched_heavy_SASA')
    if rsa is None and w == 'G':
        rsa = features.get('glycine_CB_direction_open_fraction')
    if rsa is None:
        rsa = s.get('isolated_chain_wt_RSA')
    # Absent observations are neutral missing priors, never observed zero.
    rsa = .5 if rsa is None else min(1.,max(0.,rsa))
    conf = s.get('source_asserted_AF2_pLDDT')
    conf = .7 if conf is None else min(1.,max(0.,(conf-40)/45))
    packing_count = len(features.get('nonlocal_sidechain_contact_residues',[]))
    packing = min(1., .45 + packing_count/12.) if features else 1.
    if w == 'G':
        # An absent WT sidechain has no sidechain-contact count. Use the
        # directional probe as a weak starting-point prior, not a clash energy.
        packing = .65
    b = (1-rsa)**1.5 * conf * packing
    hdiff = HYDRO[a]-HYDRO[w]
    dv = (VOLUME[a]-VOLUME[w])/100.
    ca,cw = CHARGE.get(a,0.),CHARGE.get(w,0.)
    penalty = .07 + 1.45*b*max(-hdiff,0.)
    penalty += 1.05*b*max(abs(ca)-abs(cw),0.)
    penalty += 1.1*b*max(-dv,0.) + 1.4*b*max(dv-.12,0.)
    penalty += .40*rsa*max(hdiff,0.)
    penalty += .15*b*abs(ca-cw)
    phi,psi = t.get('phi_deg'),t.get('psi_deg')
    alpha = phi is not None and psi is not None and -105<phi<-30 and -80<psi<20
    nh_bond = bool(features.get('backbone_NH_Hbonds'))
    if a == 'P' and w != 'P':
        # A proline removes NH donation. Alpha-like torsions alone do not
        # demonstrate an established helix (e.g. exposed switch-II positions).
        backbone_cost = 1.65 if alpha and nh_bond else 1.1 if nh_bond else .22
        if phi is not None and (phi>0 or phi<-105):
            backbone_cost += .8
        penalty += conf*backbone_cost
    if a == 'G' and w != 'G':
        penalty += conf*(.45 if alpha else .12)
    if w == 'G' and a != 'G':
        penalty += conf*(1.0 if phi is not None and phi>0 else .18)
    if w == 'P' and a != 'P':
        penalty += .35*conf
    # WT polar networks supply a folding prior beyond hydrophobic packing.
    # These are distance-compatible donor/acceptor contacts, not bond energies.
    partners = {x['position'] for x in features.get('sidechain_polar_contacts',[])}
    polar_w = w in 'DENQKRHSTYW'
    polar_a = a in 'DENQKRHSTYW'
    lost_network = 1. if polar_w and not polar_a else .65 if cw*ca<0 else .15 if w!=a and polar_w else 0.
    if (w,a) in [('K','R'),('R','K'),('D','E'),('E','D'),('S','T'),('T','S'),('N','Q'),('Q','N')]:
        lost_network = .03
    penalty += min(.7,.22*len(partners))*lost_network*conf
    # The canonical switch regions are conformationally mobile. A single WT
    # snapshot can overstate the energetic impact of local packing changes.
    if 30<=p<=40 or 60<=p<=74:
        penalty *= .78
    if p>166:
        penalty *= .18
    return penalty

def main():
    case, manifest, evidence, sites, summaries = load_inputs()
    atomic_path = ROOT/'wt_atomic_features.json'
    if not atomic_path.exists():
        from wt_atomic_features import generate
        generate()
    atomic_packet = json.loads(atomic_path.read_text())
    assert atomic_packet['pdb_sha256'] == sha(ROOT/'raw_evidence/RASK_HUMAN.pdb')
    atomic = {int(p):v for p,v in atomic_packet['sites'].items()}
    assert len(atomic) == len(case['reference_sequence'])
    predictions, details = [], []
    for q in case['query']:
        muts = changes(case,q)
        penalties = [site_penalty(w,p,a,sites,summaries,atomic) for w,p,a in muts]
        total = sum(penalties)
        score = round(math.tanh((.80-total)/1.65),7)
        predictions.append({'id':q['id'],'score':score})
        details.append({'id':q['id'],'mutation':q['mutation'],'penalties':penalties,'total_penalty':total,'score':score})
    assert len(predictions) == 250
    assert {x['id'] for x in predictions} == {f'M{i:03d}' for i in range(1,251)}
    assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in predictions)
    summary = ('Unsupervised KRAS abundance prior: additive hydrophobic/packing, charge, polar-contact and '
               'backbone penalties from hash-verified WT coordinates, matched side-chain SASA and confidence; '
               'flexible switches/tail downweighted. All 188 residues map exactly; no fixed partners, MSA or '
               'measured outcomes supplied. One global monotonic scale; higher predicts greater abundance. '
               'Uncalibrated: mutant relaxation, nucleotide occupancy and epistasis remain uncertain.')
    result = {'predictions':predictions,'summary':summary}
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'prediction_details.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps({'count':len(predictions),'score_range':[min(x['score'] for x in predictions),max(x['score'] for x in predictions)],'source_hashes_verified':True}))

if __name__ == '__main__':
    main()
