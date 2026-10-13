"""Unsupervised, assay-directed BH3/Mcl-1 ranking. No fitted outcome data."""
import json, math, re, hashlib
from collections import Counter
from pathlib import Path

AA=set('ACDEFGHIKLMNPQRSTVWY')
# Approximate relative binding penalties (kcal/mol-like units), not measured ddG.
# Core h1/h2/h3/h4 = C6/C10/C13/C17; D15 makes the conserved BH3 salt bridge.
CORE={
6:dict(A=0.85,C=0.65,D=3.4,E=3.1,F=0.05,G=1.65,H=1.65,I=0,K=3.2,L=0.05,M=0.05,N=2.1,P=4.5,Q=2.0,R=3.5,S=1.8,T=1.25,V=0.35,W=0.85,Y=0.75),
10:dict(A=1.8,C=1.35,D=4.0,E=3.8,F=0.65,G=2.8,H=2.25,I=0.2,K=3.8,L=0,M=0.25,N=2.8,P=5.0,Q=2.7,R=4.0,S=2.5,T=1.8,V=0.75,W=1.6,Y=1.35),
13:dict(A=1.0,C=0.8,D=2.7,E=2.7,F=0.35,G=1.9,H=1.5,I=0,K=3.0,L=0.1,M=0.15,N=1.8,P=4.5,Q=1.7,R=3.2,S=1.65,T=1.1,V=0.3,W=1.1,Y=0.95),
17:dict(A=0.95,C=0.65,D=2.3,E=2.1,F=0,G=1.65,H=1.1,I=-0.1,K=2.5,L=0.05,M=0.1,N=1.15,P=4.2,Q=1.2,R=2.7,S=1.4,T=0.95,V=-0.15,W=0.5,Y=0.25),
}
# Lower values imply better helical propensity; these contribute modestly off-core.
HELIX=dict(A=0,L=.21,R=.21,M=.24,K=.26,Q=.39,E=.40,I=.41,W=.49,S=.50,Y=.53,F=.54,V=.61,H=.61,N=.65,T=.66,C=.68,D=.69,G=1.0,P=3.2)

def mutations(q,c):
    refs=c['reference_chains']; actual=dict(refs); parsed=[]
    for token in q['mutation'].split(';'):
        mt=re.fullmatch(r'([A-Za-z0-9]+):([A-Z])(\d+)([A-Z])',token)
        assert mt, token
        chain,wt,pos,mut=mt.groups(); pos=int(pos)
        assert wt in AA and mut in AA and wt!=mut
        assert refs[chain][pos-1]==wt,(q['id'],token)
        assert not any(ch==chain and p==pos for ch,_,p,_ in parsed)
        seq=actual[chain]; actual[chain]=seq[:pos-1]+mut+seq[pos:]
        parsed.append((chain,wt,pos,mut))
    assert len(parsed)==q['mutation_order']
    assert actual==q['mutated_chains']
    assert actual[c['context']['variant_chain']]==q['sequence']
    assert all(actual[ch]==seq for ch,seq in c['context']['fixed_partner_chains'].items())
    return parsed

def mutation_penalty(wt,p,a):
    if p in CORE: return CORE[p][a]
    if p==15:
        return {'E':1.5,'N':2.5,'Q':2.8,'H':3.3,'K':4.7,'R':4.7}.get(a,3.6)+(1.5 if a=='P' else 0)
    if a=='P': return 3.0 if 4<=p<=20 else .7
    penalty=.65*(HELIX[a]-HELIX[wt])
    if p==7: # small residue adjacent to h1 at the groove edge
        penalty+=dict(F=.65,W=1.05,Y=.75,L=.15,I=.2,V=.2,K=.3,N=.1).get(a,0)
    if p==14: # native glycine fits the groove between h3 and conserved Asp
        penalty+=dict(A=.55,S=.35,N=.55,D=.7,H=1.0,I=1.35,V=1.05,L=1.2,F=1.65,W=1.95).get(a,.4)
    if p==11: # mostly solvent-facing; charge reversal is less damaging than core loss
        penalty+=dict(D=.4,E=.2,K=0,N=.15,G=.25).get(a,.15)
    if p in (9,16):
        penalty+=.3 if a in 'ILVFWY' else .05
    return penalty

def run():
    case=json.load(open('case.json')); manifest=json.load(open('evidence_manifest.json'))
    assert manifest['support_count']==0 and not manifest['measurements_present']
    assert case['context']['higher_is_better']
    preds=[]; diagnostics=[]
    for q in case['query']:
        ms=mutations(q,case)
        assert all(ch=='C' for ch,_,_,_ in ms)
        terms=[mutation_penalty(wt,p,a) for _,wt,p,a in ms]
        penalty=sum(terms)
        score=math.tanh((.75-penalty)/3.5)
        preds.append(dict(id=q['id'],score=round(score,8)))
        diagnostics.append(dict(id=q['id'],mutation=q['mutation'],penalty=penalty,terms=terms))
    assert len(preds)==250 and len({p['id'] for p in preds})==250
    assert {p['id'] for p in preds}=={f'M{i:03d}' for i in range(1,251)}
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in preds)
    summary=('Unsupervised BH3/Mcl-1 binding prior: position-specific hydrophobic-anchor compatibility, conserved Asp contact and modest helix/steric penalties, with Mcl-1 held fixed. One global monotonic scale; higher is better. No measured supports, MSA or coordinates are supplied; effects and multi-mutation interactions are uncertain, and scores are not calibrated assay values.')
    output=dict(predictions=preds,summary=summary)
    Path('predictions.json').write_text(json.dumps(output,indent=2)+'\n')
    Path('prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    Path('input_audit.json').write_text(json.dumps(dict(input_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['case.json','evidence_manifest.json']},msa_available=False,structure_available=False,validated_candidates=len(preds),variant_chain='C',fixed_chains=list(case['context']['fixed_partner_chains']),model_version='final_native_glycine_prior'),indent=2)+'\n')
    print('Saved final predictions:',len(preds),'scores; range:',min(p['score'] for p in preds),max(p['score'] for p in preds))

if __name__=='__main__': run()
