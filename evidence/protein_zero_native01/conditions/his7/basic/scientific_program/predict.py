import json, math, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
case=json.loads((ROOT/'case.json').read_text())
manifest=json.loads((ROOT/'evidence_manifest.json').read_text())
ref=case['reference_sequence']
assert manifest['support_count']==0 and manifest['measurements_present'] is False
assert case['context']['higher_is_better'] is True
assert manifest['allowed_modalities']=={'msa':False,'structure':False}
AA='ARNDCQEGHILKMFPSTWYV'
# BLOSUM62: substitution compatibility prior, not a measured fitness model.
B=np.array([
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
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]])
assert np.array_equal(B,B.T)

def mutations(q):
    expected=list(ref); out=[]
    for token in q['mutation'].split(':'):
        m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',token)
        assert m, token
        a,p,b=m.group(1),int(m.group(2)),m.group(3)
        assert a in AA and b in AA and a!=b and 1<=p<=len(ref)
        assert ref[p-1]==a,(q['id'],token,ref[p-1])
        assert expected[p-1]==a, 'Repeated position'
        expected[p-1]=b;out.append((a,p,b))
    assert ''.join(expected)==q['sequence'],q['id']
    assert len(out)==q['mutation_order']
    return out

def generic_cost(a,p,b):
    # Conservative substitutions have small costs, with additional sensitivity
    # to hydrophobic-to-charged changes in hydrophobic sequence neighborhoods.
    s=float(B[AA.index(a),AA.index(b)])
    cost=.035+.08*max(0.,1.-s)
    hydrophobic=set('AILMFWV')
    window=ref[max(0,p-5):min(len(ref),p+4)]
    fraction=sum(x in hydrophobic for x in window)/len(window)
    if a in hydrophobic and b in 'DEKR': cost+=.55+fraction
    if a in hydrophobic and b=='G':cost+=.3+fraction*.4
    if a=='P' or b=='P': cost+=.25
    if a=='G' and b not in 'AS':cost+=.2
    if p>=201:cost*=.45  # polar, glycine/proline-rich terminal region
    return cost

def checkpoint_cost(q):
    ms=mutations(q)
    return sum(generic_cost(*m) for m in ms)

def output_predictions(cost_fn, output_name, summary):
    predictions=[]; diagnostics=[]
    for q in case['query']:
        cost=cost_fn(q)
        score=round(float(math.tanh((1.8-cost)/2.)),8)
        predictions.append({'id':q['id'],'score':score})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'cost':float(cost),'score':score})
    ids=[p['id'] for p in predictions]
    assert len(ids)==250 and len(set(ids))==250
    assert set(ids)=={f'M{i:03d}' for i in range(1,251)}
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    result={'predictions':predictions,'summary':summary}
    (ROOT/output_name).write_text(json.dumps(result,indent=2)+'\n')
    return result, diagnostics

if __name__=='__main__' and not (ROOT/'checkpoint_predictions.json').exists():
    result,_=output_predictions(checkpoint_cost,'checkpoint_predictions.json',
      'Initial unsupervised sequence-only ranking: BLOSUM62 compatibility, hydrophobic-to-charge disruption, backbone changes and lower penalties in the polar terminal region. All 250 mutations reconstructed and checked against candidate sequences. No measured supports, evolutionary alignment, structure or fixed partners are supplied; scores are relative priors, not calibrated growth predictions, and epistasis is uncertain. One shared monotonic scale; higher is better.')
    print('Saved validated checkpoint for',len(result['predictions']),'candidates.')

# Revised scientific model: a shared, low-dimensional damage prior rather than
# fitted candidate/block intercepts. All local assignments below are hypotheses
# from WT sequence and IGPD biochemistry, not observations of mutant geometry.
HYDRO={a:v for a,v in zip('ACDEFGHIKLMNPQRSTVWY',
[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,2.5,2.5])}
VOLUME={a:v for a,v in zip('ACDEFGHIKLMNPQRSTVWY',
[67,86,91,109,135,48,118,124,135,124,124,96,90,114,148,73,93,105,163,141])}
NONPOLAR=set('AILMVFWY')
CHARGED=set('DEKR')
POLAR=set('NQST')

def burial_prior(a,p):
    # Hydrophobic neighborhoods are weak proxies: no coordinate-derived SASA.
    segment=ref[max(0,p-5):min(len(ref),p+4)]
    fraction=sum(x in NONPOLAR or x=='C' for x in segment)/len(segment)
    burial=min(.65,max(.15,fraction-.05))
    if p in {17,19,21,23}:burial=.95 # alternating nonpolar beta-like segment
    if p in {18,20,22,24}:burial=.1 # alternating polar/small-residue face
    if p in {78,82,66}:burial=.9 # contiguous hydrophobic active-site region
    if p in {193,195,196,198,200}:burial=.85 # C-terminal nonpolar helix
    if p in {141,142,143,163,164}:burial=.65
    if p in {7,8}:burial=.35
    if p==11:burial=.12 # polar/charged neighborhood; do not treat all Ile alike
    if p>=201:burial=.15
    return burial

def revised_mutation_cost(a,p,b):
    bl=float(B[AA.index(a),AA.index(b)])
    # Base substitution compatibility; conventional matrices are weaker than
    # chemistry at a probable core/functional position.
    cost=.018+.065*max(0.,1.-bl)
    buried=burial_prior(a,p)
    if a in NONPOLAR or a=='C':
        if b in CHARGED:cost+=2.1*buried
        elif b in POLAR:
            if a=='C' and b in 'ST':
                cost+=.12*buried # similar small sidechains; no disulfide evidence
            else:
                cost+=(.14*max(0.,HYDRO[a]-HYDRO[b]) + (.2 if b in 'NQ' else 0.))*buried
        if b=='G':cost+=1.15*buried*(VOLUME[a]-VOLUME['G'])/(VOLUME['I']-VOLUME['G'])
        if a in NONPOLAR and b in NONPOLAR:
            cost+=.003*abs(VOLUME[a]-VOLUME[b])*buried
    if a in 'DE' and b in 'KR' or a in 'KR' and b in 'DE':
        cost+=.3 # charge reversal, not proof of a particular salt bridge
    if a in NONPOLAR and b=='P':cost+=.6*buried
    if a=='G' and b not in 'AS':cost+=.3
    if a=='P' and b!='P':cost+=.35
    # Sequence-specific constraints with explicit evidence limits.
    if p==11:
        cost*=.35 # candidate local sequence is rich in polar/charged residues
    if p==56:
        # Histidine in the conserved IGPD metal-site region. Acidic residues
        # retain coordination chemistry; an amide is a weaker replacement.
        cost+={'E':.55,'D':.8,'Q':1.1}.get(b,.9)
    if p==186:
        # Disruption of histidine in the catalytic DHH...E motif.
        cost+=2.5
    if p==60 and b=='A':cost+=.35
    if p==62 and b=='F':cost+=.35 # larger aromatic near the catalytic region
    if p in {78,82} and b=='G':cost+=.3
    if p==80 and b=='Y':cost+=.55 # altered acidic/polar active-site region
    if p==84 and b=='Y':cost+=.65
    if p==86:
        # Do not assume every histidine in this His-rich protein is a ligand.
        if b=='D':cost+=.45
        elif b in NONPOLAR:cost+=.06
    if p==96 and b=='R':cost+=.45
    if p==116 and b=='Q':cost+=.55 # basic substrate-facing motif, uncertain
    if p==119:
        if b=='P':cost+=1.2 # constrain the glycine-rich FGSGF loop
        elif b=='Y':cost+=.18
        elif b=='H':cost+=.04
    if p==120 and b not in 'AG':cost+=.15
    if p==139 and b!='P':cost+=.5 # loss of a loop/helix cap
    if p==141 and b=='C':cost+=.45 # expansion at a small-residue position
    if p==144 and b=='K':cost+=.35
    if p==161 and b!='P':cost+=.85 # backbone before catalytic-region H162
    if p==154 and b=='H':cost+=.25
    if p==157 and b=='R':cost+=.8
    if p==165:
        # Local geometry can matter even for conservative acid shortening.
        cost+={'D':.18,'Q':.18,'H':.2}.get(b,.15)
    if p==169 and b=='G':cost+=.35 # helix-propensity loss in FLESFAEA segment
    if p==180:
        # Proline at the edge of the polar RGKNDHHRSE loop is less alarming
        # than a proline insertion in an uninterrupted secondary structure.
        if b=='P':cost=.18
        elif b=='T':cost=.2
    if p==195 and b=='S':cost+=.35
    if p==200 and b=='T':cost+=.15
    if p>=201:
        # Sequence becomes rich in polar residues/proline/glycine. This is a
        # lower-sensitivity prior, not observed structural disorder.
        cost*=.4
        if p==208 and b=='E':cost+=.2
        if p==210 and b=='Y':cost+=.18
    return float(cost)

def refined_cost(q):
    ms=mutations(q)
    components=[revised_mutation_cost(*m) for m in ms]
    total=sum(components)
    # A glycine replacement while retaining the WT proline/Asn turn motif is
    # potentially less compatible than remodeling neighboring turn residues.
    # Keep this single local-context prior weak: no validated epistasis supplied.
    d={p:b for a,p,b in ms}
    turn_penalty=0.
    if d.get(208)=='E' and d.get(206,'P')=='P':
        turn_penalty=.18 if d.get(207,'N')=='T' else .3
    total+=turn_penalty
    # The most severe event is more informative than several weak, uncertain
    # substitutions; combine additive burden and a bottleneck component.
    return .85*total+.15*max(components)

if __name__=='__main__':
    result,diagnostics=output_predictions(refined_cost,'final_predictions.json',
      'Sequence-only IGPD functional ranking combines BLOSUM62, hydrophobic/charge disruption, catalytic-region and loop sensitivity, and lower terminal-region penalties; one weak turn-context interaction is hypothetical. All mutations reconstruct supplied sequences; all 250 IDs use one global scale, higher is better. Input hashes are recorded. No outcomes, MSA depth, evolutionary source mappings or structural coordinates are supplied; no fixed partners are specified. Scores are uncalibrated priors: beneficial substitutions, oligomer assembly and epistasis remain uncertain, especially among otherwise conservative variants.')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (ROOT/'model_rationale.json').write_text(json.dumps({
      'initial_checkpoint':'Generic BLOSUM62 and local hydrophobic-disruption model; saved before refinement.',
      'substantive_revision':'Corrected overly uniform substitution sensitivity using WT sequence context and IGPD biochemistry: alternating hydrophobic N-terminal positions, core hydrophobic substitutions, constrained loops, catalytic-region histidine, and polar terminal sequence. No outcome-based tuning or frequency-as-conservation inference.',
      'implementation_correction':'Scaled hydrophobic-to-polar penalties to chemical change; cysteine-to-serine/threonine gets a smaller penalty than large hydrophobe-to-polar changes. Glycine cavity penalty scales with sidechain volume, so Ala-to-Gly is not treated as Ile-to-Gly. Alternating polar positions 18/20/22/24 get lower burial priors.',
      'epistasis_limit':'One weak proline/glycine terminal turn-context penalty. No predicted relaxed coordinates, validated contacts or experimentally established epistasis.',
      'burden_model':'Shared additive penalties blended with the strongest individual penalty; no candidate block normalization or fitted block intercepts.',
      'input_checks':{'support_count':manifest['support_count'],'measurements_present':manifest['measurements_present'],'allowed_modalities':manifest['allowed_modalities'],'reference_length':len(ref),'candidate_count':len(case['query']),'fixed_partners_present':False},
      'source_integrity':'Local input SHA256 values are stored in input_provenance.json; no external evolutionary or coordinate source is provided.'
    },indent=2)+'\n')
    import hashlib
    (ROOT/'input_provenance.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['case.json','evidence_manifest.json']},indent=2)+'\n')
    print('Saved refined predictions and diagnostics for',len(result['predictions']),'candidates.')
