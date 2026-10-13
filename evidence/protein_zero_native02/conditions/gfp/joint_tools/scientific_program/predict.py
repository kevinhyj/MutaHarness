"""Label-free GFP fluorescence ranking; all coefficients are scientific priors."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np
import structure_tools
import msa_tools

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT/name).read_text())

def inputs():
    case, manifest = read('case.json'), read('evidence_manifest.json')
    msa, structure = read('msa_evidence.json'), read('structure_evidence.json')
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better']
    reference = case['reference_sequence']
    digest = hashlib.sha256(reference.encode()).hexdigest()
    assert digest == msa['reference_sequences_sha256']['__single__']
    assert digest == structure['reference_sequence_sha256']
    for path, spec in manifest['files'].items():
        if isinstance(spec,dict) and 'sha256' in spec:
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == spec['sha256'], path
    assert len(case['query']) == 250
    assert {q['id'] for q in case['query']} == {f'M{i:03}' for i in range(1,251)}
    mi = {q['id']:q for q in msa['candidates']}
    si = {q['id']:q for q in structure['candidates']}
    for q in case['query']:
        seq = list(reference)
        mutations = [re.fullmatch(r'([A-Z])(\d+)([A-Z])',t).groups() for t in q['mutation'].split(':')]
        assert len(mutations) == q['mutation_order']
        assert len({int(p) for w,p,a in mutations}) == len(mutations)
        for w,p,a in mutations:
            p=int(p); assert reference[p-1] == w; seq[p-1] = a
        assert ''.join(seq) == q['sequence']
        assert q['mutation'] == mi[q['id']]['mutation'] == si[q['id']]['mutation']
    return case, msa, structure

def checkpoint():
    case, msa, structure = inputs()
    mi={q['id']:q for q in msa['candidates']}
    si={q['id']:q for q in structure['candidates']}
    predictions=[]
    for q in case['query']:
        cost=0.
        for t,s in zip(mi[q['id']]['substitution_terms'],si[q['id']]['sites']):
            assert (t['position'],t['WT'],t['ALT']) == (s['position'],s['WT'],s['ALT'])
            w,a=t['WT'],t['ALT']
            if w==a: continue
            lr=t['sum_log_frequency_ratio'] if t['non_gap_depth']>0 else None
            rsa=s['derived_descriptors']['wt_rsa']
            burial=1-min(1,rsa) if rsa is not None else 0.5
            chemical=0.45*burial*(a in 'DEKR' and w not in 'DEKR')
            backbone=1.8*(a=='P' and w!='P') + 0.5*(w=='G' and a!='G')
            cost += 0.35 + (0.30*max(-lr,-1.5) if lr is not None else 0) + chemical + backbone
        predictions.append({'id':q['id'],'score':round(math.tanh((4-cost)/8),8)})
    out={'predictions':predictions,'summary':'Initial label-free GFP fluorescence ranking: mutation burden plus observed natural-homolog log-frequency ratios and conservative WT burial/backbone penalties. Zero-depth sites use structural and generic priors, not neutral MSA evidence. No outcomes or validated epistasis; high uncertainty.'}
    validate(out)
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

def validate(out):
    assert len(out['predictions'])==250
    assert {q['id'] for q in out['predictions']} == {f'M{i:03}' for i in range(1,251)}
    assert all(math.isfinite(q['score']) and -1<=q['score']<=1 for q in out['predictions'])

VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',
    [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY',
    [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
CHARGE = {'D':-1.,'E':-1.,'K':1.,'R':1.,'H':0.2}
# General engineering knowledge, not outcomes for the supplied candidates.
# Modest transfer priors for established cycle3, superfolder and Venus changes.
FOLDING_PRIORS = {'Y39N','F46L','F99S','M153T','V163A','I171V','S175G','A206V'}

def geometry(case, structure):
    residues = structure_tools._read_pdb(ROOT/'raw_evidence/GFP_AEQVI.pdb')
    assert all(r['protein'] and r['pdb_chain']=='A' for r in residues)
    assert ''.join(r['WT'] for r in residues) == case['reference_sequence']
    assert len(residues)==238 and len({r['pdb_residue_number'] for r in residues})==238
    raw=structure['raw_coordinate_evidence']
    assert raw['reference_sequence_sha256']==structure['reference_sequence_sha256']
    assert raw['raw_source']['sha256'] == hashlib.sha256((ROOT/'raw_evidence/GFP_AEQVI.pdb').read_bytes()).hexdigest()
    assert raw['raw_source']['matches_declared_source_sha256']
    ri={s['position']:s for s in raw['sites']}
    result={}
    def atom(i,name): return np.array(residues[i]['atoms'][name]['coord'])
    ring=np.array([atom(65,n) for n in ['CG','CD1','CD2','CE1','CE2','CZ','OH']])
    for i,r in enumerate(residues):
        pos=i+1
        assert ri[pos]['WT']==r['WT'] and ri[pos]['chain'] is None
        assert np.allclose(atom(i,'CA'),ri[pos]['CA_coordinate_A'])
        bonds=[]
        if i>0 and r['WT']!='P':
            n=atom(i,'N'); direction=atom(i-1,'C')-atom(i-1,'O')
            h=n+direction/np.linalg.norm(direction)
            for j in range(len(residues)):
                if abs(i-j)<=1: continue
                c,o=atom(j,'C'),atom(j,'O')
                distance=lambda x,y:max(float(np.linalg.norm(x-y)),0.5)
                # DSSP-like electrostatic donor descriptor of WT only; not an energy change.
                e=27.888*(1/distance(o,n)+1/distance(c,h)-1/distance(o,h)-1/distance(c,n))
                if e<-.5 and distance(o,n)<3.6: bonds.append((j+1,e))
        cis=False
        if i>0:
            xyz=np.array([atom(i-1,'CA'),atom(i-1,'C'),atom(i,'N'),atom(i,'CA')])
            b=xyz[1:]-xyz[:-1];u=np.cross(b[0],b[1]);v=np.cross(b[1],b[2])
            omega=np.degrees(np.arctan2(np.dot(np.cross(u,v),b[1]/np.linalg.norm(b[1])),np.dot(u,v)))
            cis=abs(omega)<40
        side=np.array([a['coord'] for name,a in r['atoms'].items() if name not in {'N','CA','C','O','OXT'} and a['element']!='H'])
        ring_distance=float(np.linalg.norm(side[:,None,:]-ring[None,:,:],axis=2).min()) if len(side) else None
        result[pos]={'rsa':ri[pos]['isolated_chain_wt_RSA'],
            'donor_strength': min(1.,max([-e/2.5 for j,e in bonds],default=0.)),
            'cis_peptide':cis,'ring_distance_A':ring_distance,
            'coordinate_complete':bool(ri[pos]['backbone_complete'] and ri[pos]['sidechain_complete'])}
    return result

def mutation_prior(t, site, g):
    p,w,a=t['position'],t['WT'],t['ALT']
    name=f'{w}{p}{a}'
    if w==a: return {'mutation':name,'cost':0.,'functional_penalty':0.,'msa_observed':t['sum_log_frequency_ratio'] is not None}
    obs=site['wt_observations']; rsa=g['rsa']
    assert rsa is not None and g['coordinate_complete']
    # RSA >1 remains in provenance; only its use as an exposure feature is bounded.
    burial=math.exp(-max(0.,rsa)/0.25)
    conf=obs['pLDDT']
    reliability=min(1.,max(0.25,(conf-40)/35)) if conf is not None else 0.5
    qw,qa=CHARGE.get(w,0.),CHARGE.get(a,0.)
    dv=VOLUME[a]-VOLUME[w]
    hyd=abs(HYDRO[a]-HYDRO[w])/9.
    charge=abs(qa-qw)
    aromatic_change=float((w in 'FWY') != (a in 'FWY'))
    chemical=abs(dv)/100.+hyd+charge/2.+0.3*aromatic_change
    # Solvent-exposed chemical changes are mild; buried charge and packing changes stronger.
    packing=abs(dv)/100.*(1.15 if dv>0 else 0.9)
    charge_intro=3.*max(0.,abs(qa)-abs(qw))
    charge_loss=0.9*max(0.,abs(qw)-abs(qa))
    charge_flip=2.*float(qw*qa<0)
    hydrophobic_loss=1.1*max(0.,HYDRO[w]-HYDRO[a])/9.
    physical=0.3*chemical+burial*(packing+charge_intro+charge_loss+charge_flip+hydrophobic_loss)
    # Natural ALT observations moderate a physical penalty, without changing frequencies.
    # This does not assert neutral fitness: homologs have different genetic backgrounds.
    depth=t['non_gap_depth'] or 0.
    alt=t['frequency_ALT'] if depth>0 else None
    if alt is not None: physical *= 1.-0.4*min(1.,alt/0.2)
    phi,psi=obs['phi_deg'],obs['psi_deg']
    backbone=0.
    if a=='P':
        phi_mismatch=min(1.5,max(0.,abs(phi+65)-20)/50) if phi is not None else 0.5
        constrained=float(phi is not None and psi is not None and
                          ((phi<-85 and (psi>70 or psi<-150)) or (-110<phi<-30 and -80<psi<30)))
        backbone=0.35+1.8*phi_mismatch+1.8*g['donor_strength']+0.65*constrained*burial
    if w=='P': backbone+=3.8 if g['cis_peptide'] else 0.35+0.35*burial
    if w=='G' and phi is not None and phi>0:
        # Positive-phi geometry disfavors most non-Gly residues, less so Ser/Asn/Asp.
        backbone+=3.*math.exp(-0.5*((phi-75)/50)**2)*(0.65 if a in 'SND' else 1.)
    if a=='G': backbone+=0.2+0.35*burial
    photophysics=0.
    if g['ring_distance_A'] is not None and g['ring_distance_A']<5.:
        photophysics=0.3*burial*chemical*(5-g['ring_distance_A'])/3.
    evolutionary=0.
    lr=t['sum_log_frequency_ratio']
    if depth>0 and lr is not None:
        evolutionary=0.32*(depth/(depth+5.))*max(-3.,min(6.,-lr))
    cost=0.30+evolutionary+reliability*(physical+backbone+photophysics)
    if name in FOLDING_PRIORS:
        cost=0.30*cost-0.65
    # Green fluorescence requires chromophore formation and its conjugated Tyr ring.
    # These are mechanism priors, not measured effects or mutant-structure predictions.
    functional=0.
    if p==66 and a not in 'YFW': functional=8.
    if p==65 and a=='P': functional=5.
    if p==67 and a!='G': functional=5.
    if p==96 and a!='R': functional=2.5 if a=='K' else 4.
    if p==222 and a not in 'ED': functional=2.
    return {'mutation':name,'cost':float(cost),'functional_penalty':functional,
        'evolutionary_component':float(evolutionary),'physical_component':float(reliability*physical),
        'backbone_component':float(reliability*backbone),'rsa':rsa,
        'chromophore_ring_distance_A':g['ring_distance_A'],'msa_observed':depth>0 and lr is not None,
        'transferred_folding_prior':name in FOLDING_PRIORS}

def final_predictions():
    case,msa,structure=inputs()
    # Reproduce fixed scores as a computational/coverage check, not predictive validation.
    reproduced=msa_tools.score_case(case)
    for orig,check in zip(msa['candidates'],reproduced['candidates']):
        assert orig['id']==check['id']
        for name in ['sum_log_frequency_ratio','sum_true_log_odds_ratio']:
            x,y=orig['scores'][name],check['scores'][name]
            assert x['complete']==y['complete'] and x['observed_terms']==y['observed_terms']
            assert x['score']==y['score'] or (x['score'] is not None and y['score'] is not None and math.isclose(x['score'],y['score'],abs_tol=1e-10))
    geo=geometry(case,structure)
    mi={q['id']:q for q in msa['candidates']};si={q['id']:q for q in structure['candidates']}
    diagnostics=[];predictions=[]
    for q in case['query']:
        terms=[]
        for t,s in zip(mi[q['id']]['substitution_terms'],si[q['id']]['sites']):
            assert (t['chain'],t['position'],t['WT'],t['ALT'])==(s['chain'],s['position'],s['WT'],s['ALT'])
            terms.append(mutation_prior(t,s,geo[t['position']]))
        additive=sum(t['cost'] for t in terms)
        worst=max([t['cost'] for t in terms],default=0.)
        functional=sum(t['functional_penalty'] for t in terms)
        # A severe single insult receives modest extra weight; not fitted epistasis.
        risk=additive+0.4*max(0.,worst)+functional
        score=math.tanh((5.-risk)/7.)
        predictions.append({'id':q['id'],'score':round(score,8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'risk':risk,
            'additive_burden':additive,'worst_mutation_burden':worst,'functional_penalty':functional,
            'msa_complete':mi[q['id']]['scores']['sum_log_frequency_ratio']['complete'],
            'msa_observed_terms':sum(t['msa_observed'] for t in terms),'terms':terms})
    out={'predictions':predictions,'summary':
        'Label-free GFP fluorescence ranking combines mutation burden, observed MSA log-frequency ratios, WT burial/backbone constraints, chromophore chemistry and modest known GFP folding priors. Source hashes and all residue identities/mappings verified; one global score scale. MSA Neff is 13.9; 17 candidates have incomplete coverage, with missing terms handled by non-MSA priors. No assay calibration, relaxed mutants or validated epistasis; substantial uncertainty.'}
    validate(out)
    (ROOT/'final_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    return out

if __name__ == '__main__':
    import sys
    if '--checkpoint' in sys.argv: checkpoint()
    else: final_predictions()
