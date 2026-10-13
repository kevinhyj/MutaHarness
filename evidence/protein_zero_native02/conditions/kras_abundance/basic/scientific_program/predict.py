"""Outcome-free, knowledge-based KRAS abundance predictor.

No MSA, atomic coordinates, assay labels, learned variant effects, or external
files are used. Burial values below are approximate Ras topology priors, not
computed solvent-accessibility observations. Weights are fixed physical priors;
there is no outcome fitting or candidate-block normalization.
"""
import hashlib
import json
import math
import re
from pathlib import Path

VOLUME = dict(zip('ARNDCQEGHILKMFPSTWYV',
    [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
# Side-chain transfer hydrophobicity (a general physicochemical scale).
HYDRO = {'A':.31,'R':-1.01,'N':-.60,'D':-.77,'C':1.54,'Q':-.22,
         'E':-.64,'G':0.,'H':.13,'I':1.80,'L':1.70,'K':-.99,
         'M':1.23,'F':1.79,'P':.72,'S':-.04,'T':.26,'W':2.25,'Y':.96,'V':1.22}
HYDROPHOBIC = set('AVILMFWYC')
CHARGED = set('DEKR')
# Confidence-shrunk estimates of burial: topology/side-chain environment priors.
# Unlisted residues in the folded G-domain receive a low exposure prior (.10).
BURIAL = {
 2:.02,3:.10,4:.18,5:.30,6:.90,7:.35,8:.88,9:.35,10:.10,
 11:.12,12:.12,13:.10,14:.48,15:.10,16:.10,17:.15,18:.40,19:.85,
 20:.18,21:.60,22:.12,23:.75,24:.45,25:.10,26:.08,27:.18,28:.42,29:.30,
 30:.04,31:.04,32:.08,33:.04,34:.03,35:.07,36:.15,37:.04,38:.04,
 39:.08,40:.18,41:.08,42:.06,43:.22,44:.45,45:.42,46:.68,47:.08,
 48:.12,49:.10,50:.22,51:.48,52:.70,53:.62,54:.10,55:.90,56:.72,
 57:.15,58:.18,59:.18,60:.05,61:.06,62:.04,63:.10,64:.48,65:.20,
 66:.12,67:.55,68:.13,69:.08,70:.10,71:.72,72:.40,73:.10,74:.12,
 75:.10,76:.08,77:.40,78:.90,79:.78,80:.60,81:.86,82:.42,83:.48,
 84:.75,85:.25,86:.15,87:.10,88:.12,89:.15,90:.65,91:.15,92:.10,
 93:.70,94:.30,95:.30,96:.58,97:.22,98:.10,99:.28,100:.65,101:.07,
 102:.18,103:.72,104:.10,105:.05,106:.04,107:.08,108:.10,109:.14,
 110:.25,111:.70,112:.92,113:.65,114:.70,115:.12,116:.20,117:.12,
 118:.18,119:.18,120:.48,121:.08,122:.12,123:.08,124:.10,125:.48,
 126:.06,127:.16,128:.08,129:.18,130:.60,131:.10,132:.16,133:.78,
 134:.62,135:.16,136:.20,137:.60,138:.14,139:.10,140:.15,141:.48,
 142:.10,143:.15,144:.12,145:.40,146:.35,147:.16,148:.08,149:.08,
 150:.12,151:.15,152:.80,153:.12,154:.22,155:.52,156:.88,157:.28,
 158:.10,159:.92,160:.58,161:.12,162:.10,163:.80,164:.10,165:.10,166:.18}
HELIX = set(range(16,26)) | set(range(67,75)) | set(range(87,105)) | set(range(127,139)) | set(range(152,167))
HELIX_START = {16,67,87,127,152}
BETA = set(range(5,11)) | set(range(39,47)) | set(range(49,58)) | set(range(79,82)) | set(range(116,120)) | set(range(145,148))
GLY_TURN = {10:.30,12:.16,13:.24,15:.34,48:.32,60:.25,75:.20}


def mutation_terms(m):
    w,p,a=re.fullmatch(r'([A-Z])(\d+)([A-Z])',m).groups(); p=int(p)
    b=BURIAL.get(p,.10) if p <=166 else .01
    if p>=167:
        # Disordered hypervariable region. Prenylation/localization is not
        # automatically abundance; do not assign catalytic/affinity penalties.
        loss=.006+.008*abs(HYDRO[w]-HYDRO[a])+.00012*abs(VOLUME[w]-VOLUME[a])
        if a in 'FW':loss+=.012
        return {'tail':loss}
    terms={'base':.012}
    # Burial desolvation/packing. Positive costs only: no unsupported claims of
    # stability improvement from a more hydrophobic residue.
    terms['hydrophobic_loss']=.25*b*max(0.,HYDRO[w]-HYDRO[a])
    dv=VOLUME[a]-VOLUME[w]
    terms['cavity']=.40*b*max(0.,-dv)/100.
    terms['crowding']=(.65*b*b+.10*b)*max(0.,dv-8.)/100.
    if w in HYDROPHOBIC:
        unsatisfied={'N':.22,'Q':.20,'S':.07,'T':.05,'H':.12,
                     'D':.30,'E':.30,'K':.28,'R':.30}.get(a,0.)
        terms['buried_polar']=b*unsatisfied
    else:
        terms['buried_polar']=0.
    # Solvent-exposed hydrophobicity and loss of surface hydrogen bonds/charge.
    terms['surface_hydrophobicity']=.075*(1.-b)*max(0.,HYDRO[a]-HYDRO[w]) if w not in HYDROPHOBIC else 0.
    if w in CHARGED and a not in CHARGED:
        terms['charge_change']=.025*(1.-b)
    elif w in CHARGED and a in CHARGED and ((w in 'DE')!=(a in 'DE')):
        terms['charge_change']=.045+.08*b
    else:terms['charge_change']=0.
    if w in 'NQST' and a in HYDROPHOBIC:
        terms['hydrogen_bond_loss']=.025+.035*b
    # Proline is strongly disruptive within regular secondary structure,
    # but helix caps and loops have different backbone requirements.
    if a=='P':
        if p in HELIX: penalty=(.30 if p in HELIX_START else .62)+.20*b
        elif p in BETA:penalty=.55+.18*b
        else:penalty=.09
        terms['proline']=penalty
    if a=='G' and w!='G':
        terms['glycine_entropy']=.14 if p in HELIX else (.07 if p in BETA else .015)
    if w=='P' and a!='P':terms['turn_proline_loss']=.14
    if w=='G' and p in GLY_TURN:
        terms['glycine_turn']=GLY_TURN[p]*(.5+max(0.,VOLUME[a]-90.)/130.)
        if a=='P':terms['glycine_turn']+=.24
    # Recurrent oncogenic G12 C/D/V substitutions commonly retain a folded
    # nucleotide-bound protein; loss of catalytic activity is not penalized.
    if p==12 and a in 'CDVS':
        terms['glycine_turn']= {'C':.055,'D':.065,'V':.080,'S':.040}[a]
    # Nucleotide occupancy can stabilize Ras. Only direct binding disruption
    # risk is included here, rather than conservation of catalytic output.
    nucleotide=0.
    if p==16 and a not in 'KR':nucleotide=.23 if a in 'EG' else .17
    if p==17 and a not in 'ST':nucleotide=.17 if a in 'NG' else .23
    if p==35 and a not in 'ST':nucleotide=.10
    if p==57 and a not in 'DE':nucleotide=.15
    if p==116 and a not in 'NQ':nucleotide=.12 if a=='H' else (.35 if a=='W' else .25)
    if p==117 and a not in 'KR':nucleotide=.12 if a=='N' else .20
    if p==119 and a not in 'DE':nucleotide=.32
    if p==146 and a in 'FWY':nucleotide=.24
    terms['nucleotide']=nucleotide
    # The first two residues do not belong to the folded core.
    if p<=2:
        terms={k:.6*v for k,v in terms.items()}
    return terms


def predict(case):
    predictions=[]; features=[]
    for q in case['query']:
        details={m:mutation_terms(m) for m in q['mutation'].split(':')}
        loss=sum(sum(t.values()) for t in details.values())
        # Global monotonic compression of an additive destabilization burden.
        # Offset is an arbitrary ranking origin, not a measured WT value.
        score=math.tanh(.24-1.15*loss)
        predictions.append({'id':q['id'],'score':round(score,6)})
        features.append({'id':q['id'],'mutation':q['mutation'],'destabilization_prior':loss,'terms':details})
    summary=('Outcome-free KRAS abundance prior combining approximate Ras fold environments, '
             'side-chain hydrophobicity/volume, backbone disruption and nucleotide-binding risk. '
             'All mutation identities and 250 unique IDs were verified. No measured supports, '
             'MSA, WT coordinates or fixed partners were supplied; evolutionary/structural evidence '
             'and source mappings are therefore unavailable. One global monotonic scale ranks additive '
             'mutation costs; scores are not calibrated assay values. Burial estimates, nucleotide '
             'effects and double-mutant epistasis remain uncertain.')
    return {'predictions':predictions,'summary':summary},features


def validate(case,result):
    wt=case['reference_sequence']; ids=[q['id'] for q in case['query']]
    assert len(ids)==250 and len(set(ids))==250 and set(ids)=={f'M{i:03d}' for i in range(1,251)}
    assert len(result['predictions'])==250 and {p['id'] for p in result['predictions']}==set(ids)
    assert len({p['id'] for p in result['predictions']})==250
    for q in case['query']:
        actual={f'{w}{i+1}{a}' for i,(w,a) in enumerate(zip(wt,q['sequence'])) if w!=a}
        assert len(wt)==len(q['sequence']) and actual==set(q['mutation'].split(':'))
        assert q['mutation_order']==len(actual)
        for m in q['mutation'].split(':'):
            w,p,a=re.fullmatch(r'([A-Z])(\d+)([A-Z])',m).groups()
            assert wt[int(p)-1]==w and q['sequence'][int(p)-1]==a
    assert case['context']['higher_is_better'] is True
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in result['predictions'])


if __name__=='__main__':
    case=json.loads(Path('case.json').read_text())
    manifest=json.loads(Path('evidence_manifest.json').read_text())
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    assert manifest['allowed_modalities']=={'msa':False,'structure':False}
    result,features=predict(case);validate(case,result)
    for name in ['checkpoint_predictions.json','final_predictions.json']:
        Path(name).write_text(json.dumps(result,indent=2)+'\n')
    Path('prediction_features.json').write_text(json.dumps(features,indent=2)+'\n')
    audit={'input_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['case.json','evidence_manifest.json']},
           'reference_sequence_sha256':hashlib.sha256(case['reference_sequence'].encode()).hexdigest(),
           'reference_length':len(case['reference_sequence']),
           'msa_coverage':0,'coordinate_coverage':0,'observed_supports':0,
           'fixed_partners':'None supplied in case.json',
           'positions':'1-based identity checked against reference and mutant sequences',
           'model_revision_rationale':'Replaced binary core/loop penalties with graded approximate environments and side-chain physicochemistry to resolve conservative packing changes; distinguish helix caps from helix interiors. Added direct nucleotide-binding risk, without conflating catalytic conservation with abundance. No outcome-based model selection or validated epistasis.'}
    Path('model_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Validated 250 predictions; range',min(p['score'] for p in result['predictions']),max(p['score'] for p in result['predictions']))
    lookup={q['id']:q['mutation'] for q in case['query']}
    ordered=sorted(result['predictions'],key=lambda q:q['score'])
    print('Lowest:',[(q['id'],lookup[q['id']],q['score']) for q in ordered[:12]])
    print('Highest:',[(q['id'],lookup[q['id']],q['score']) for q in ordered[-12:]])
