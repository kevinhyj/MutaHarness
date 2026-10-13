"""Zero-shot structure-informed stability ranking. No assay labels are used."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np
from structure_tools import MAXASA_A2, describe_case
from wt_interactions import polar_contacts

ROOT = Path(__file__).resolve().parent
# Physical-property priors: Kyte-Doolittle hydropathy, residue volumes (A^3),
# and relative helix free energies (Pace/Scholtz; proline handled separately).
HYD=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
VOL=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
HELIX=dict(zip('ACDEFGHIKLMNPQRSTVWY',[0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]))
CHARGE={a:(-1 if a in 'DE' else 1 if a in 'KR' else 0) for a in HYD}
# Sidechain water/octanol partition constants (Fauchere-Pliska); aromatic
# hydrophobicity is represented more appropriately than by Kyte-Doolittle.
TRANSFER=dict(zip('ACDEFGHIKLMNPQRSTVWY',[.31,1.54,-.77,-.64,1.79,0,.13,1.80,-.99,1.70,1.23,-.60,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
REACH=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1,2,2,3,3,0,3,2,4,3,3,2,2,3,4,1,1,2,4,3]))
POLAR=set('DEHKNQRSTY')

def load_and_check():
    case=json.load(open(ROOT/'case.json'))
    manifest=json.load(open(ROOT/'evidence_manifest.json'))
    evidence=json.load(open(ROOT/'structure_evidence.json'))
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    assert case['context']['higher_is_better'] is True
    for f,v in manifest['files'].items():
        if isinstance(v,dict):
            assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v['sha256'], f
    wt=case['reference_sequence']
    assert hashlib.sha256(wt.encode()).hexdigest()==evidence['reference_sequence_sha256']
    parsed=describe_case(case) # Supplied tool verifies notation, residue identity, sequences.
    assert {q['id'] for q in case['query']}=={f'M{i:03d}' for i in range(1,251)}
    assert len(case['query'])==250
    assert not case.get('reference_chains') and case['domain']=='protein' # No fixed binding partners supplied.
    raw=evidence['raw_coordinate_evidence']
    assert raw['raw_source']['sha256']==manifest['files']['raw_evidence/UBE4B_HUMAN.pdb']['sha256']
    assert raw['mapping'][0]['method']=='exact_whole_chain_sequence'
    assert raw['mapping'][0]['unobserved_positions']==[]
    for s in raw['sites']:
        assert wt[s['position']-1]==s['WT']
    return case,evidence,parsed

def site_cost(w,a,s):
    if w==a: return 0.0
    obs=s['wt_observations']; rsa=s['derived_descriptors']['wt_rsa']
    # Missing exposure cannot be treated as buried or exposed: use only weak generic prior.
    if rsa is None: return .4 + .2*(a in 'GP')
    burial=(max(0.,1-min(1.,rsa)))**1.8
    # Loss of buried hydrophobic packing, cavity formation and overpacking.
    hyd=.20*burial*max(0.,HYD[w]-HYD[a])
    cavity=.013*burial*max(0.,VOL[w]-VOL[a])
    over=.008*burial*max(0.,VOL[a]-VOL[w])
    charge=.6*burial*abs(CHARGE[w]-CHARGE[a])
    phi,psi=obs['phi_deg'],obs['psi_deg']
    helical=phi is not None and psi is not None and -100<phi<-30 and -80<psi<0
    beta=phi is not None and psi is not None and phi<-70 and (psi>70 or psi<-150)
    backbone=0.
    if helical:
        backbone+=.8*(HELIX[a]-HELIX[w])
        if a=='P': backbone+=.7
    elif a=='G': backbone+=.5
    if a=='P' and w!='P': backbone+=1.5 if beta else .5
    if w=='P': backbone+=.7
    if phi is not None and phi>0:
        backbone+=(-.5 if a=='G' else .35 if a not in 'DNST' else 0.)
    # Small untrained disruption prior, mild penalty for newly exposed hydrophobes.
    surface=.08*min(1.,rsa)*max(0.,HYD[a]-HYD[w])
    return .15+hyd+cavity+over+charge+backbone+surface

def predict_checkpoint(case,evidence,parsed):
    rows=[]; diagnostics=[]
    for q,c in zip(case['query'],parsed['candidates']):
        assert q['id']==c['id']
        costs=[site_cost(s['WT'],s['ALT'],s) for s in c['sites']]
        cost=sum(costs)
        rows.append({'id':q['id'],'score':round(float(-math.tanh(cost/4.)),8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'cost':cost,'site_costs':costs})
    return {'predictions':rows,'summary':'Unsupervised additive stability prior using source-verified WT exposure, residue packing/charge changes and backbone compatibility; higher scores predict higher cDNA-display proteolysis stability. No supports or MSA are available. All 69 residues map exactly to the verified single-chain structure; WT geometry does not predict mutant relaxation or validated epistasis. One global tanh scale; substantial uncalibrated uncertainty.'},diagnostics

def validate(output):
    rows=output['predictions']
    assert len(rows)==250 and len({r['id'] for r in rows})==250
    assert {r['id'] for r in rows}=={f'M{i:03d}' for i in range(1,251)}
    assert all(math.isfinite(r['score']) and -1<=r['score']<=1 for r in rows)

def role_retention(w,a,role,salt=False):
    """Weak chemistry/size prior, NOT a mutant H-bond or energy calculation."""
    if w==a: return 1.
    donors=set('HKNQRSTWY'); acceptors=set('DEHNQSTY')
    if a not in (donors if role=='donor' else acceptors): return 0.
    if w in 'ST' and a in 'ST': return .9
    if w in 'DE' and a in 'DE' or w in 'KR' and a in 'KR':
        return .85*math.exp(-.30*abs(VOL[a]-VOL[w])/30)
    geometry=math.exp(-.45*abs(REACH[a]-REACH[w]))
    chemical=.75
    if w in 'ST' and a not in 'STY': chemical=.5
    if a in 'HWY' and w not in 'HWY': chemical=.35
    if salt and CHARGE[a]==0: chemical*=.55
    return chemical*geometry

def backbone_classes(parsed):
    obs={s['position']:s['wt_observations'] for c in parsed['candidates'] for s in c['sites']}
    # Include all sites even if not mutated in this batch.
    case=json.load(open(ROOT/'case.json'))
    for s in case['evidence']['sites']:
        obs.setdefault(s['position'],s['wt_structure'])
    alpha={p for p,o in obs.items() if o['phi_deg'] is not None and o['psi_deg'] is not None
           and -100<o['phi_deg']<-30 and -80<o['psi_deg']<0}
    # A lone alpha-like torsion in a turn is not a regular helix.
    helix={p for p in alpha if any({k,k+1,k+2}<=alpha for k in (p-2,p-1,p))}
    return helix

def refined_site_cost(w,a,s,helix):
    if w==a:return {'packing':0.,'backbone':0.,'prior':0.,'surface':0.}
    rsa=s['derived_descriptors']['wt_rsa']; o=s['wt_observations']; p=s['position']
    if rsa is None:
        return {'packing':0.,'backbone':.25*(a in 'GP'),'prior':.25,'surface':0.}
    b=max(0.,1-min(1.,rsa))**1.7
    hyd_loss=.65*b*max(0.,TRANSFER[w]-TRANSFER[a])
    cavity=.012*b*max(0.,VOL[w]-VOL[a])
    overpacking=.010*b*max(0.,VOL[a]-VOL[w])
    new_polar=.4*b*(a in POLAR and w not in POLAR)
    new_charge=.9*b*(CHARGE[a]!=0 and CHARGE[w]==0)
    # Favorable hydrophobic burial is deliberately weak: WT coordinates do
    # not establish that the new sidechain can pack or retain the polar network.
    hyd_gain=-.10*b*max(0.,TRANSFER[a]-TRANSFER[w])
    packing=hyd_loss+cavity+overpacking+new_polar+new_charge+hyd_gain
    surface=.15*min(1.,rsa)*max(0.,TRANSFER[a]-TRANSFER[w])
    phi,psi=o['phi_deg'],o['psi_deg']
    beta=phi is not None and psi is not None and phi<-70 and (psi>70 or psi<-150)
    backbone=0.
    if a=='P':
        if p in helix:backbone=1.8 if p-1 not in helix else 3.0
        elif beta:backbone=2.0+(.5 if phi<-110 else 0.)
        else:backbone=.7+(.5 if phi is not None and phi<-100 else 0.)
    elif w=='P':backbone=.85+.25*b
    elif p in helix:
        backbone=.8*(HELIX[a]-HELIX[w])+.25*(a=='G')
    elif a=='G':backbone=.6 if beta else .35
    # Relative positive-phi compatibility: the WT non-glycine already has
    # its own conformational penalty; do not penalize all replacements equally.
    if phi is not None and phi>0:
        strain={x:(0 if x=='G' else .8 if x in 'DNST' else 1.3 if x=='A' else 4 if x=='P' else 2.3) for x in HYD}
        backbone+=.7*(strain[a]-strain[w])
    return {'packing':packing,'backbone':backbone,'prior':.12*(.4+.6*b),'surface':surface}

def predict_refined(case,evidence,parsed):
    from collections import Counter
    bonds,residues=polar_contacts(ROOT/'raw_evidence/UBE4B_HUMAN.pdb')
    raw={s['position']:s for s in evidence['raw_coordinate_evidence']['sites']}
    # Exposure is recomputed in the supplied raw-coordinate evidence, with
    # standard-heavy-atom completeness required for RSA.
    assert all(s['sidechain_complete'] and s['backbone_complete'] and s['isolated_chain_wt_RSA'] is not None for s in raw.values())
    mult=Counter()
    for b in bonds:
        for side in ['donor','acceptor']:
            if b[side+'_sidechain']:mult[(b[side+'_pos'],b[side+'_atom'])]+=1
    for b in bonds:
        exposure=[max(0.,1-min(1.,raw[b[x+'_pos']]['isolated_chain_wt_RSA']))**1.7 for x in ['donor','acceptor']]
        multiplicity=max(mult.get((b[x+'_pos'],b[x+'_atom']),1) for x in ['donor','acceptor'])
        geometry=min(1.,max(.5,(b['angle']-90)/70))*min(1.,max(.5,(3.7-b['distance'])/.8))
        b['prior_weight']=(1.15 if b['salt'] else .7)*(.45+.55*max(exposure))*geometry/math.sqrt(multiplicity)
    helix=backbone_classes(parsed); rows=[]; diagnostics=[]
    for q,c in zip(case['query'],parsed['candidates']):
        changes={s['position']:s['ALT'] for s in c['sites']}
        terms=[refined_site_cost(s['WT'],s['ALT'],s,helix) for s in c['sites']]
        network=0.; network_terms=[]
        for bond in bonds:
            if not (bond['donor_pos'] in changes or bond['acceptor_pos'] in changes):continue
            retain=[]
            for role in ['donor','acceptor']:
                w=bond[role+'_WT'];a=changes.get(bond[role+'_pos'],w)
                # Proline eliminates a backbone NH donor; other substitutions
                # retain this backbone atom in this explicitly static prior.
                rr=role_retention(w,a,role,bond['salt']) if bond[role+'_sidechain'] else (0. if role=='donor' and a=='P' else 1.)
                retain.append(rr)
            loss=bond['prior_weight']*(1-retain[0]*retain[1])
            if bond['salt']:
                a=changes.get(bond['donor_pos'],bond['donor_WT']);z=changes.get(bond['acceptor_pos'],bond['acceptor_WT'])
                if CHARGE[a]*CHARGE[z]>0:loss+=.7*bond['prior_weight']
            network+=loss
            if loss:network_terms.append({'positions':[bond['donor_pos'],bond['acceptor_pos']],'cost':loss})
        cost=sum(sum(t.values()) for t in terms)+network
        score=-math.tanh(cost/4.)
        rows.append({'id':q['id'],'score':round(float(score),8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'cost':cost,'site_terms':terms,'polar_network_cost':network,'affected_WT_contacts':network_terms})
    summary='Zero-shot stability ranking from WT burial, sidechain packing/polarity and backbone compatibility, plus conservative preservation of observed WT polar contacts; shared contacts counted once for double mutants. Source hashes verified; all 69 residues map exactly to chain A with complete heavy atoms. No partners, MSA or measured outcomes are supplied. One global tanh scale, higher is better. Scores are uncalibrated priors: static WT contacts cannot establish mutant relaxation, proteolysis effects or experimental epistasis; uncertainty is substantial.'
    return {'predictions':rows,'summary':summary},diagnostics,bonds

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',action='store_true');args=ap.parse_args()
    case,evidence,parsed=load_and_check()
    if args.checkpoint:
        output,diag=predict_checkpoint(case,evidence,parsed)
        filename='checkpoint_predictions.json'
    else:
        output,diag,bonds=predict_refined(case,evidence,parsed)
        (ROOT/'wt_polar_contact_priors.json').write_text(json.dumps(bonds,indent=2)+'\n')
        filename='final_predictions.json'
    validate(output)
    (ROOT/filename).write_text(json.dumps(output,indent=2)+'\n')
    (ROOT/filename.replace('predictions','diagnostics')).write_text(json.dumps(diag,indent=2)+'\n')
    print('Saved',filename,':',len(output['predictions']),'scores; range',min(r['score'] for r in output['predictions']),max(r['score'] for r in output['predictions']))
