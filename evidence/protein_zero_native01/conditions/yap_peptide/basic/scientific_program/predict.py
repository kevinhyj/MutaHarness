"""Sequence-only YAP WW / fixed PPxY partner ranking prior; no outcome fitting."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
case = json.loads((ROOT/'case.json').read_text())
manifest = json.loads((ROOT/'evidence_manifest.json').read_text())
wt = case['reference_sequence']
AA = 'ACDEFGHIKLMNPQRSTVWY'
# BLOSUM62 is a broad substitution prior, not a target-assay predictor.
B62 = '''
4 0 -2 -1 -2 0 -2 -1 -1 -1 -1 -2 -1 -1 -1 1 0 0 -3 -2
0 9 -3 -4 -2 -3 -3 -1 -3 -1 -1 -3 -3 -3 -3 -1 -1 -1 -2 -2
-2 -3 6 2 -3 -1 -1 -3 -1 -4 -3 1 -1 0 -2 0 -1 -3 -4 -3
-1 -4 2 5 -3 -2 0 -3 1 -3 -2 0 -1 2 0 0 -1 -2 -3 -2
-2 -2 -3 -3 6 -3 -1 0 -3 0 0 -3 -4 -3 -3 -2 -2 -1 1 3
0 -3 -1 -2 -3 6 -2 -4 -2 -4 -3 0 -2 -2 -2 0 -2 -3 -2 -3
-2 -3 -1 0 -1 -2 8 -3 -1 -3 -2 1 -2 0 0 -1 -2 -3 -2 2
-1 -1 -3 -3 0 -4 -3 4 -3 2 1 -3 -3 -3 -3 -2 -1 3 -3 -1
-1 -3 -1 1 -3 -2 -1 -3 5 -2 -1 0 -1 1 2 0 -1 -2 -3 -2
-1 -1 -4 -3 0 -4 -3 2 -2 4 2 -3 -3 -2 -2 -2 -1 1 -2 -1
-1 -1 -3 -2 0 -3 -2 1 -1 2 5 -2 -2 0 -1 -1 -1 1 -1 -1
-2 -3 1 0 -3 0 1 -3 0 -3 -2 6 -2 0 0 1 0 -3 -4 -2
-1 -3 -1 -1 -4 -2 -2 -3 -1 -3 -2 -2 7 -1 -2 -1 -1 -2 -4 -3
-1 -3 0 2 -3 -2 0 -3 1 -2 0 0 -1 5 1 0 -1 -2 -2 -1
-1 -3 -2 0 -3 -2 0 -3 2 -2 -1 0 -2 1 5 -1 -1 -3 -3 -2
1 -1 0 0 -2 0 -1 -2 0 -2 -1 1 -1 0 -1 4 1 -2 -3 -2
0 -1 -1 -1 -2 -2 -2 -1 -1 -1 -1 0 -1 -1 -1 1 5 0 -2 -2
0 -1 -3 -2 -1 -3 -3 3 -2 1 1 -3 -2 -2 -3 -2 0 4 -3 -1
-3 -2 -4 -3 1 -2 -2 -3 -3 -2 -1 -4 -4 -2 -3 -3 -2 -3 11 2
-2 -2 -3 -2 3 -3 2 -1 -2 -1 -1 -2 -3 -1 -2 -2 -2 -1 2 7
'''
blosum=np.array([list(map(float,l.split())) for l in B62.strip().splitlines()])
idx={a:i for i,a in enumerate(AA)}
# Approximate functional regions from canonical three-stranded type-I WW fold.
strands={13,14,15,16,17,22,23,24,25,26,27,32,33,34,35,36,37}
core={7:.20,9:.35,11:.30,13:1.0,15:.60,24:.75,25:.65,26:.45,29:.50,35:.35}
interface={24:.80,26:.35,27:.40,28:.65,32:.55,35:1.0}
hydro=set('AVILMFWY'); arom=set('FWY'); charged=set('DEKR')

def effect(p,a,b):
    # Evolutionary exchangeability is weak at exposed positions and stronger in core.
    distance=max(0.,(blosum[idx[a],idx[a]]-blosum[idx[a],idx[b]])/6.)
    fold=.12*distance
    bind=0.
    if p in core:
        w=core[p]
        if a in arom:
            fold += w*(.22 if b in arom else .65 if b=='H' else .80 if b in hydro else 1.65)
        elif b not in hydro:
            fold += w*(1.6 if b in charged else .85)
        elif a in 'ILVM' and b in 'ILVM':
            fold += .03*w
        else:
            fold += .16*w
    if b=='P' and p in strands: fold+=1.15
    if b=='G' and p in strands: fold+=.40
    if p in {12,21}:
        fold+= .18 if b in 'AS' else .48 if b in 'DN' else .95
    if a=='P':
        fold+= (1.25 if p==38 else .12 if p in {8,10} else .2)
    if p in interface:
        w=interface[p]
        if a in arom: bind += w*(.15 if b in arom else .95 if b=='H' else 1.85)
        elif p==28: bind += w*(.25 if b in 'YFQ' else .65 if b in 'NRK' else 1.0)
        elif p==32: bind += w*(.05 if b=='S' else .40 if b=='N' else .70)
        elif p==27: bind += w*(.2 if b in 'QS' else .4 if b=='D' else .9)
        elif p==26 and b not in hydro: bind+=w*.9
    # Small mechanistic priors for turns and electrostatic ligand recognition.
    if p==30:
        fold += {'N':-.85,'G':.30,'A':.05,'E':.15,'H':-.15,'Q':-.12,'K':-.10}.get(b,.55)
    if p==6:
        fold += {'R':-.90,'K':-.90,'N':0.,'H':-.10,'Q':-.08,'E':.05,'V':.50,'G':.15,'Y':.15}.get(b,.05)
    if p==19 and b in 'RK': bind-=.45
    if p==23:
        bind += .05 if b=='H' else .15 if b in 'FL' else .30
    if p==27:
        # The conserved Asn side-chain supports the second hairpin H-bond network.
        fold += .15 if b in 'QS' else .35 if b in 'DH' else .65
    if p==31 and b=='P': fold+=.60
    if p==36:
        # Peripheral recognition-site charge/packing: weaker than central contacts.
        bind += -.15 if b in 'RK' else -.12 if b=='H' else .15 if b=='E' else .10 if b=='L' else 0.
    if p==11 and b=='P': fold+=.40
    if p==31:
        bind += {'R':-.55,'K':-.35,'H':-.15,'E':.20,'P':.65,'L':.20}.get(b,0.)
    if p==14 and b in 'AQ': fold-=.15
    if p==11 and b=='V': fold-=.15
    return fold,bind

def validate(q):
    assert q['id'] and q['sequence']
    sequence=list(wt); mutations=[]
    for token in q['mutation'].split(';'):
        m=re.fullmatch(r'([A-Za-z0-9]+):([A-Z])(\d+)([A-Z])',token)
        assert m, token
        ch,a,pos,b=m.groups(); p=int(pos)
        assert ch==case['context']['variant_chain']=='A'
        assert wt[p-1]==a and a!=b and b in AA
        assert p not in {x[0] for x in mutations}
        sequence[p-1]=b; mutations.append((p,a,b))
    assert ''.join(sequence)==q['sequence']==q['mutated_chains']['A']
    assert len(mutations)==q['mutation_order']
    for ch,seq in case['context']['fixed_partner_chains'].items():
        assert q['mutated_chains'][ch]==seq==case['reference_chains'][ch]
    return mutations

def predict(q):
    ms=validate(q)
    fold,bind=np.sum([effect(*m) for m in ms],axis=0)
    # Unknown WT folded fraction: average log effects across modest stability priors.
    # No assay-calibrated free energies or mutant-coordinate claims are made.
    wt_logits=np.array([-.65,0.,.65])
    wt_log_fold=-np.logaddexp(0.,-wt_logits)
    log_fold=-np.logaddexp(0.,fold-wt_logits)
    raw=float(np.mean(log_fold-wt_log_fold))-bind
    return float(np.tanh(raw/2.0)),dict(folding_penalty=float(fold),binding_penalty=float(bind),latent=float(raw))

assert case['context']['higher_is_better'] is True
assert manifest['support_count']==0 and not manifest['measurements_present']
assert not manifest['allowed_modalities']['msa'] and not manifest['allowed_modalities']['structure']
rows=[]; diagnostics=[]
for q in case['query']:
    s,d=predict(q); rows.append({'id':q['id'],'score':round(s,6)})
    diagnostics.append({'id':q['id'],'mutation':q['mutation'],**d})
assert len(rows)==250 and {r['id'] for r in rows}=={f'M{i:03d}' for i in range(1,251)}
assert len({r['id'] for r in rows})==len(rows)
assert all(math.isfinite(r['score']) and -1<=r['score']<=1 for r in rows)
summary=('Unsupervised sequence-only YAP WW-domain prior combining substitution compatibility, core/strand and turn stability, and PPxY-binding chemistry with fixed partner GTPPPPYTVG. Folding effects average three WT-stability assumptions; all scores share one monotonic scale, higher is better. All 250 mutation identities, sequences and partners were verified; input hashes are saved. No measurements, MSA or structure are available. Scores have substantial uncertainty, especially for multi-mutants; neither assay calibration nor validated epistasis is claimed.')
result={'predictions':rows,'summary':summary}
(ROOT/'checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'final_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
(ROOT/'input_audit.json').write_text(json.dumps({'sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['case.json','evidence_manifest.json']},'candidate_count':len(rows),'reference_length':len(wt),'variable_chain':'A','fixed_partners':case['context']['fixed_partner_chains'],'evidence':manifest,'validation':'all mutation identities, sequences, partners, IDs, score direction and finite range passed'},indent=2)+'\n')
print('Saved checkpoint_predictions.json and predict.py; all 250 candidates validated.')
for label,subset in [('top',sorted(diagnostics,key=lambda x:x['latent'],reverse=True)[:15]),('bottom',sorted(diagnostics,key=lambda x:x['latent'])[:10])]:
 print(label)
 for r in subset: print(r['id'],r['mutation'],round(r['latent'],3))
