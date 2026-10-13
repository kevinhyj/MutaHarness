"""Unsupervised, WT-structure-informed ranking for the supplied stability assay.

All coefficients are fixed scientific priors, never fitted to assay outcomes.
The checkpoint uses only generic exposure/backbone/packing terms. Refinements
must be motivated by verified WT observations and recorded in model_notes.md.
"""
import hashlib
import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Sidechain volumes (A^3), hydrophobic transfer scale, and helix penalties.
AA = 'ACDEFGHIKLMNPQRSTVWY'
VOL = dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
HYD = dict(zip(AA, [0.31,1.54,-0.77,-0.64,1.79,0.,0.13,1.8,-0.99,1.7,1.23,-0.6,0.72,-0.22,-1.01,-0.04,0.26,1.22,2.25,0.96]))
HEL = dict(zip(AA, [0.,0.68,0.69,0.40,0.54,1.0,0.61,0.41,0.26,0.21,0.24,0.65,3.2,0.39,0.21,0.50,0.66,0.61,0.49,0.53]))
POLAR = set('DENQHSTYKR')
CHARGE = {'D':-1, 'E':-1, 'K':1, 'R':1}
DONORS = dict(zip(AA,[0,0,0,0,0,0,1,0,1,0,0,1,0,1,3,1,1,0,1,1]))
ACCEPTOR_COUNTS = dict(zip(AA,[0,0,2,2,0,0,1,0,0,0,0,1,0,1,0,1,1,0,0,1]))
# Approximate sidechain polar group reach in bonds; long charged sidechains can
# expose their termini. This is a prior, not mutant accessibility computation.
REACH = dict(zip(AA,[1,2,3,4,4,0,3,3,5,3,4,3,2,4,6,2,2,2,4,6]))


def load_inputs():
    case = json.loads((ROOT/'case.json').read_text())
    manifest = json.loads((ROOT/'evidence_manifest.json').read_text())
    evidence = json.loads((ROOT/'structure_evidence.json').read_text())
    raw = evidence['raw_coordinate_evidence']
    seq_hash = hashlib.sha256(case['reference_sequence'].encode()).hexdigest()
    source_hash = hashlib.sha256((ROOT/'raw_evidence/AMFR_HUMAN.pdb').read_bytes()).hexdigest()
    assert seq_hash == evidence['reference_sequence_sha256'] == raw['reference_sequence_sha256']
    assert source_hash == raw['raw_source']['sha256'] == manifest['files']['raw_evidence/AMFR_HUMAN.pdb']['sha256']
    assert not manifest['measurements_present'] and manifest['support_count'] == 0
    assert case['context']['higher_is_better']
    assert not case.get('partners') and not case['context'].get('partners')
    assert len(raw['mapping']) == 1 and raw['mapping'][0]['mapped_observed_residues'] == len(case['reference_sequence'])
    sites = {s['position']:s for s in raw['sites']}
    summary = {s['position']:s.get('wt_structure',{}) for s in case['evidence']['sites']}
    return case, sites, summary


def mutations(query, reference):
    changes = []
    rebuilt = list(reference)
    for token in query['mutation'].split(':'):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', token)
        assert match, token
        wt, pos, alt = match.group(1), int(match.group(2)), match.group(3)
        assert reference[pos-1] == wt and alt in AA and wt != alt
        assert pos not in [p for _,p,_ in changes]
        rebuilt[pos-1] = alt
        changes.append((wt,pos,alt))
    assert ''.join(rebuilt) == query['sequence']
    assert len(changes) == query['mutation_order']
    return changes


def generic_terms(wt, pos, alt, sites, summary):
    s = sites[pos]
    rsa = s['isolated_chain_wt_RSA']
    assert rsa is not None and s['backbone_complete'] and s['sidechain_complete']
    burial = max(0.,1.-min(1.,rsa))
    phi, psi = summary[pos].get('phi_deg'), summary[pos].get('psi_deg')
    # Null torsions are missing observations, not coil. Known contiguous helices
    # supply a modest prior only; positive-phi turns are treated separately.
    helix = float(phi is not None and psi is not None and -105 < phi < -30 and -80 < psi < 10)
    if phi is None and (2 <= pos <= 16 or 22 <= pos <= 32 or 35 <= pos <= 43):
        helix = 0.7
    dv = VOL[alt]-VOL[wt]
    terms = {'change_prior': -0.15-0.25*burial,
             'solvation': 0.65*burial*(HYD[alt]-HYD[wt]),
             'packing': -burial*(0.010*max(-dv,0)+0.00013*max(dv,0)**2),
             'helix': helix*(HEL[wt]-HEL[alt]),
             'buried_charge': -1.0*burial*(abs(CHARGE.get(alt,0))-abs(CHARGE.get(wt,0))),
             'polar_network': -0.8*burial if wt in POLAR and alt not in POLAR else 0.}
    if alt == 'P' and helix:
        terms['proline'] = -1.2*helix
    if phi is not None and phi > 0 and wt == 'G' and alt != 'G':
        terms['positive_phi'] = -2.0 if alt not in 'DN' else -1.1
    if wt == 'P' and alt != 'P' and not helix:
        terms['turn_proline'] = -0.7
    return terms


def refined_terms(wt, pos, alt, sites, summary, geom):
    terms = generic_terms(wt,pos,alt,sites,summary)
    burial = max(0.,1.-min(1.,sites[pos]['isolated_chain_wt_RSA']))
    helix = geom['helix_compatibility'][str(pos)]
    dv = VOL[alt]-VOL[wt]
    # Cavity costs are approximately proportional to lost volume. For enlarged
    # sidechains use a weak quadratic prior because sidechain repacking is absent.
    terms['packing'] = -burial*(0.018*max(-dv,0)+0.00010*max(dv,0)**2)
    terms['helix'] = helix*(HEL[wt]-HEL[alt]) if wt!='P' else 0.
    terms.pop('polar_network',None)
    terms.pop('turn_proline',None)
    terms.pop('proline',None)
    # Native charges may already be stabilized. Removal is therefore rewarded
    # less than addition of a new buried charge is penalized.
    escaped_burial = burial*min(1.,(max(2,REACH[wt])/max(2,REACH[alt]))**0.8)
    terms['buried_charge'] = (0.4*burial*max(abs(CHARGE.get(wt,0))-abs(CHARGE.get(alt,0)),0)
                              -escaped_burial*max(abs(CHARGE.get(alt,0))-abs(CHARGE.get(wt,0)),0))
    terms['new_polar_groups'] = -burial*(0.50*max(DONORS[alt]-DONORS[wt],0)
                                        +0.30*max(ACCEPTOR_COUNTS[alt]-ACCEPTOR_COUNTS[wt],0))
    if wt=='P':
        terms['proline_loop_entropy']=-0.75
    if alt=='P':
        terms['proline']= -0.8*helix
        if helix==0:
            # A turn can contain an important backbone NH without being helical.
            n_hb=sum(e['donor_position']==pos for e in geom['backbone_hydrogen_bonds'])
            terms['turn_NH_loss']=-0.55*min(1,n_hb)
            phi=summary[pos].get('phi_deg')
            if phi is not None and not (-95<phi<-40):terms['turn_proline_geometry']=-1.2
    phi=summary[pos].get('phi_deg')
    if phi is not None and phi>0 and wt!='G':
        terms['positive_phi'] = -1.0 if alt in 'IVTFWY' else (0.15 if alt in 'CDNSGA' else -0.15)
    return terms


def retained_role(wt, alt, role):
    """Broad chemistry/size compatibility; no claimed mutant hydrogen bond."""
    if wt==alt:return 1.
    count = DONORS if role=='donor' else ACCEPTOR_COUNTS
    if count[alt]==0:return 0.
    if {wt,alt} <= set('ST'):return 0.90
    if {wt,alt} <= set('NQ'):return 0.65
    if {wt,alt} <= set('DE'):return 0.75
    # Isosteric amide/carboxylate changes can preserve the compatible role.
    if {wt,alt} <= set('DN') or {wt,alt} <= set('EQ'):return 0.85
    delta=abs(REACH[alt]-REACH[wt])
    retain=0.72*math.exp(-0.40*delta)
    if alt in 'WY':retain*=0.65
    if alt in 'KR':retain*=0.70
    return retain


def interaction_terms(changes, sites, geom):
    variants={p:(w,a) for w,p,a in changes}
    terms=[]
    for e in geom['sidechain_hydrogen_bonds']:
        dp,ap=e['donor_position'],e['acceptor_position']
        donor_changed=dp in variants and e['donor_sidechain']
        acceptor_changed=ap in variants and e['acceptor_sidechain']
        if not donor_changed and not acceptor_changed:continue
        rd=retained_role(*variants[dp],'donor') if donor_changed else 1.
        ra=retained_role(*variants[ap],'acceptor') if acceptor_changed else 1.
        # Polar group burial matters more than whole-residue exposure for caps.
        b1=max(0.,1.-e['donor_atom_SASA']/35.)
        b2=max(0.,1.-e['acceptor_atom_SASA']/35.)
        energy=0.55+0.50*(b1+b2)/2.
        terms.append({'type':'WT_contact_loss','positions':[dp,ap],
                      'value':-energy*(1.-rd*ra)})
    # Salt bridges need not pass an explicit-hydrogen angular criterion. Look
    # only for opposite-charge WT residues with a direct sidechain contact.
    for e in geom['sidechain_contact_pairs']:
        i,j=e['left'],e['right']
        wi,wj=sites[i]['WT'],sites[j]['WT']
        ai=variants.get(i,(wi,wi))[1]; aj=variants.get(j,(wj,wj))[1]
        if i not in variants and j not in variants:continue
        bi=max(0.,1.-min(1.,sites[i]['isolated_chain_wt_RSA']))
        bj=max(0.,1.-min(1.,sites[j]['isolated_chain_wt_RSA']))
        if CHARGE.get(wi,0)*CHARGE.get(wj,0)<0 and e['distance_A']<3.5:
            preserved=CHARGE.get(ai,0)*CHARGE.get(aj,0)<0
            if not preserved:
                loss=-(0.35+0.50*min(bi,bj))
                terms.append({'type':'WT_salt_loss','positions':[i,j],'value':loss})
        if CHARGE.get(ai,0)*CHARGE.get(aj,0)>0 and CHARGE.get(wi,0)*CHARGE.get(wj,0)<=0:
            terms.append({'type':'nearby_like_charge_prior','positions':[i,j],
                          'value':-0.7*min(bi,bj)*max(0.,(5.-e['distance_A'])/2.)})
        if i in variants and j in variants:
            # Weak contact-volume correction: simultaneous expansion is risky;
            # expansion/shrinkage can compensate. These are uncertain priors.
            dvi=VOL[ai]-VOL[wi]; dvj=VOL[aj]-VOL[wj]
            proximity=max(0.,min(1.,(5.-e['distance_A'])/2.))
            value=-0.00008*min(bi,bj)*proximity*dvi*dvj
            terms.append({'type':'coupled_volume_prior','positions':[i,j],
                          'value':max(-0.45,min(0.45,value))})
    return terms


def predict(checkpoint=True):
    case, sites, summary = load_inputs()
    geom=None
    if not checkpoint:
        from inspect_geometry import geometry
        geom=geometry()
        (ROOT/'geometry_features.json').write_text(json.dumps(geom,indent=2)+'\n')
        # Stable key type in memory and on disk.
        geom['helix_compatibility']={str(k):v for k,v in geom['helix_compatibility'].items()}
    predictions, diagnostics = [], []
    for query in case['query']:
        changes = mutations(query, case['reference_sequence'])
        details = [{'position':pos,'WT':wt,'ALT':alt,
                    'terms':generic_terms(wt,pos,alt,sites,summary) if checkpoint else refined_terms(wt,pos,alt,sites,summary,geom)} for wt,pos,alt in changes]
        interactions=[] if checkpoint else interaction_terms(changes,sites,geom)
        raw_score = sum(sum(s['terms'].values()) for s in details)+sum(e['value'] for e in interactions)
        score = float(math.tanh(raw_score/3.0))
        predictions.append({'id':query['id'],'score':score})
        diagnostics.append({'id':query['id'],'mutation':query['mutation'],'raw_score':raw_score,'sites':details,'interactions':interactions})
    result = {'predictions':predictions,'summary':'Unsupervised stability ranking using verified WT exposure, sidechain packing, solvation and backbone compatibility; additive mutation effects on one global tanh scale (higher is better). No measured supports or MSA; static WT geometry cannot establish mutant relaxation, epistasis or assay calibration, so uncertainty is substantial.'}
    if not checkpoint:
        result['summary']='Unsupervised stability prior combining WT burial, solvation, packing, backbone compatibility and conservative polar-contact loss/coupling terms. PDB/reference hashes and exact 47-residue mapping verified; all standard heavy atoms present, missing torsions handled explicitly, no fixed partners or MSA. One global tanh scale; higher predicts better stability. No outcomes, mutant relaxation or validated epistasis; scores are uncalibrated and uncertainty is substantial.'
    validate(result, case)
    out = 'checkpoint_predictions.json' if checkpoint else 'final_predictions.json'
    (ROOT/out).write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print('Saved',out,'with',len(predictions),'finite, identity-checked scores.')
    return result


def validate(result, case):
    ids = [p['id'] for p in result['predictions']]
    assert len(ids) == 250 and len(set(ids)) == 250
    assert set(ids) == {q['id'] for q in case['query']}
    assert all(math.isfinite(p['score']) and -1 <= p['score'] <= 1 for p in result['predictions'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    predict(checkpoint=not args.final)
