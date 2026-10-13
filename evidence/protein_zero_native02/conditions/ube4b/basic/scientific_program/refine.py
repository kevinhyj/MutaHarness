"""Fixed, unsupervised stability heuristic conditioned on the supplied WT sequence.

No positions, contacts, or coefficients are fitted to outcomes or query-group counts.
The helix and burial profiles are low-confidence primary-sequence hypotheses,
not observations of coordinates. Mutation contributions are additive.
"""
import json, math, hashlib
from pathlib import Path
import numpy as np
from predict import AA, BLOSUM, VOL, CHARGE, parse_and_check

# General Chou-Fasman propensities and coarse hydrophobic character.
ALPHA=dict(zip(AA,[1.42,.98,.67,1.01,.70,1.11,1.51,.57,1.,1.08,1.21,1.16,1.45,1.13,.57,.77,.83,1.08,.69,1.06]))
TURN=dict(zip(AA,[.66,.95,1.56,1.46,1.19,.98,.74,1.56,.95,.47,.59,1.01,.60,.60,1.52,1.43,.96,.96,1.14,.50]))
HYDRO=dict(zip(AA,[.35,-1.,-.60,-.90,.60,-.60,-.85,-.15,-.35,1.,.90,-.90,.70,1.,-.15,-.40,-.20,1.10,.70,.90]))


def sequence_profiles(wt):
    n=len(wt); helix=np.zeros(n); turn=np.zeros(n)
    # Proline/glycine-delimited stretches can support regular secondary structure.
    # Taper at boundaries; never interpret this as a known secondary structure.
    start=0
    for stop in list(i for i,a in enumerate(wt) if a in 'PG')+[n]:
        length=stop-start
        if length>=7:
            mean_alpha=np.mean([ALPHA[a] for a in wt[start:stop]])
            confidence=np.clip(.53+.55*(mean_alpha-1.0)+.015*(length-7),.35,.78)
            for j in range(start,stop):
                edge=min(j-start,stop-1-j)
                helix[j]=confidence*min(1.,.40+.24*edge)
        start=stop+1
    for i in range(n):
        for offset,weight in [(-2,.45),(-1,.35),(1,.60),(2,.20)]:
            j=i+offset
            if 0<=j<n and wt[j]=='P':turn[i]+=weight
        for offset in [-1,1]:
            j=i+offset
            if 0<=j<n and wt[j]=='G':turn[i]+=.40
    turn=np.minimum(turn,1.)
    burial=np.zeros(n)
    for i,a in enumerate(wt):
        lo=max(0,i-4);hi=min(n,i+5)
        neighbors=[wt[j] for j in range(lo,hi) if j!=i]
        density=np.mean([b in 'AVILMFWYC' for b in neighbors])
        burial[i]=.12+.43*density+.13*helix[i]
        if a in 'VILMFWYC':burial[i]+=.16
        elif a=='A':burial[i]+=.07
        # Chain ends have lower sequence-only evidence of packing.
        edge=min(i,n-1-i)
        burial[i]*=min(1.,.55+.15*edge)
    return helix,turn,np.clip(burial,.05,.80)


def network_loss(old,new,turn,burial,helix):
    # U-box folds commonly depend on polar networks rather than zinc coordination.
    # Without contacts, use only a small chemical prior, not assigned salt bridges.
    if old in 'DE':
        strength=.22+.38*turn+.28*burial
        retention=0. if new in 'DE' else .30 if new in 'NQ' else .65 if new in 'STH' else 1.25 if new in 'KR' else 1.
    elif old in 'KR':
        strength=.22+.25*burial+.10*helix
        retention=0. if new in 'KR' else .30 if new in 'NQH' else .70 if new in 'ST' else 1.25 if new in 'DE' else 1.
    else:return 0.
    return strength*retention


def single_cost(old,pos,new,wt,profiles):
    i=pos-1;hp,tp,bp=(float(v[i]) for v in profiles)
    a,b=AA.index(old),AA.index(new)
    general=.09*((BLOSUM[a,a]+BLOSUM[b,b])/2-BLOSUM[a,b])
    terms={'general_compatibility':general}
    dh=HYDRO[new]-HYDRO[old];dv=VOL[new]-VOL[old]
    if old in 'AVILMFWYC':
        terms['hydrophobic_core']=1.12*bp*max(0.,-dh)
        terms['packing']=bp*(.012*max(0.,-dv)+.009*max(0.,dv))
    else:
        # A buried polar site might gain hydrophobic packing, but loss of its
        # network is separately charged. Shrink this uncertain benefit strongly.
        terms['packing']=.004*bp*abs(dv)
        terms['hydrophobic_core']=-.20*bp*max(0.,dh) if new in 'VILMFWY' else 0.
        terms['surface_hydrophobe']=.10*(1-bp)*max(0.,dh) if new in 'VILMFWY' else 0.
    terms['polar_network']=network_loss(old,new,tp,bp,hp)
    if old not in 'GP' and new not in 'GP':
        terms['helix_propensity']=.40*hp*math.log(ALPHA[old]/ALPHA[new])
        terms['turn_propensity']=.70*tp*max(0.,math.log(TURN[old]/TURN[new]))
    if old in 'STND' and new not in 'GP':
        terms['turn_bulk']=.011*tp*max(0.,VOL[new]-max(120.,VOL[old]))
    if new=='G':
        terms['glycine_entropy']=.20*(1-tp)+.90*hp
    if new=='P':
        terms['proline_backbone']=2.8*hp+.20*(1-tp)
        adjacent=sum(0<=j<len(wt) and wt[j]=='P' for j in [i-1,i+1])
        terms['consecutive_proline']=.65*adjacent
    if old=='P':terms['native_proline']=.65+.45*tp
    if old=='G':
        # P-X-G is consistent with a tight beta turn, for which glycine can
        # accommodate positive phi. This remains an uncertain sequence prior.
        tight=i>=2 and wt[i-2]=='P'
        terms['native_glycine']=.80+1.50*int(tight)+.40*tp
    # Reduce predictions for poorly constrained terminal residues.
    terminal=min(i,len(wt)-1-i)
    factor=.40 if terminal==0 else .50 if terminal==1 else .65 if terminal==2 else .85 if terminal==3 else 1.
    terms={k:float(v*factor) for k,v in terms.items()}
    return float(sum(terms.values())),terms


def main():
    case=json.load(open('case.json'));manifest=json.load(open('evidence_manifest.json'))
    wt,candidates,parsed=parse_and_check(case)
    assert not manifest['measurements_present'] and manifest['support_count']==0
    assert manifest['allowed_modalities']=={'msa':False,'structure':False}
    profiles=sequence_profiles(wt)
    breakdown={};pred=[]
    for c in candidates:
        parts=[]
        for old,pos,new in parsed[c['id']]:
            cost,terms=single_cost(old,pos,new,wt,profiles)
            parts.append({'mutation':f'{old}{pos}{new}','cost':cost,'terms':terms})
        total=sum(p['cost'] for p in parts)
        score=float(np.tanh(-total/4.))
        pred.append({'id':c['id'],'score':round(score,8)})
        breakdown[c['id']]={'mutation':c['mutation'],'total_cost':total,'components':parts}
    summary=('Sequence-only stability prior combining general substitution compatibility, uncertain local packing/helix/turn preferences, and a modest U-box polar-network prior. Mutation costs are additive and mapped to one global [-1,1] scale; higher means predicted greater stability. All 250 mutation identities were verified and input hashes recorded. No measured supports, MSA, coordinates, or fixed partners were supplied; scores are uncalibrated, and burial, hydrogen bonds, epistasis, and proteolysis-specific effects remain uncertain.')
    result={'predictions':pred,'summary':summary}
    assert len(pred)==250 and len({p['id'] for p in pred})==250
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in pred)
    Path('final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('prediction_details.json').write_text(json.dumps({'profiles':[{ 'position':i+1,'wt':a,'helix_prior':float(profiles[0][i]),'turn_prior':float(profiles[1][i]),'burial_prior':float(profiles[2][i])} for i,a in enumerate(wt)],'candidates':breakdown},indent=2)+'\n')
    Path('model_rationale.md').write_text('The checkpoint used general substitution distances, local hydrophobic density, and uniform glycine/proline penalties. The refinement corrects context-insensitive backbone penalties and adds cavity/overpacking and polar-network terms appropriate to a stability assay of a U-box. All local context is inferred from the supplied primary sequence, with reduced confidence; no observed coordinates, conservation scores, contacts, or assay outcomes are claimed. The core and turn terms are modest and no candidate-pair interactions are inferred from the query design. Coefficients are fixed scientific heuristics, not fitted. No accuracy claim follows from the execution checks.\n')
    audit=json.load(open('input_audit.json'))
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in audit['sources'].items())
    print('Validated final:',len(pred),'predictions; range',min(p['score'] for p in pred),max(p['score'] for p in pred))
    for name,ls in [('Highest',sorted(pred,key=lambda v:-v['score'])[:15]),('Lowest',sorted(pred,key=lambda v:v['score'])[:15])]:
        print(name)
        for p in ls:print(p['id'],breakdown[p['id']]['mutation'],round(p['score'],3))
    print('Selected sequence-prior profiles')
    for pos in [8,13,17,20,23,24,28,29,31,39,41,42,44,45,55,61,66]:
        print(pos,wt[pos-1],*[round(float(v[pos-1]),2) for v in profiles])
if __name__=='__main__':main()
