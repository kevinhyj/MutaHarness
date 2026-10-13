"""Label-free stability ranking. No outcomes or outside data are used."""
import hashlib,json,math,re
from pathlib import Path
import numpy as np
from msa_tools import score_case

ROOT=Path(__file__).resolve().parent
AA='ACDEFGHIKLMNPQRSTVWY'
HYDRO=set('AVILMFWY')
CHARGE=set('DEKR')

def load_and_check():
    c=json.loads((ROOT/'case.json').read_text())
    manifest=json.loads((ROOT/'evidence_manifest.json').read_text())
    e=json.loads((ROOT/'msa_evidence.json').read_text())
    assert manifest['support_count']==0 and not manifest['measurements_present']
    assert c['context']['higher_is_better'] is True
    for name,meta in manifest['files'].items():
        if isinstance(meta,dict) and 'sha256' in meta:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256'],name
    assert e['reference_sequences_sha256']['__single__']==hashlib.sha256(c['reference_sequence'].encode()).hexdigest()
    recomputed=score_case(c)
    assert recomputed==e,'Fixed MSA evidence does not reproduce'
    expected={f'M{i:03d}' for i in range(1,251)}
    assert len(c['query'])==250 and {q['id'] for q in c['query']}==expected
    assert not c.get('reference_chains') and not c.get('fixed_partners'), 'Revisit partner handling'
    for q in c['query']:
        changes=[f'{w}{i}{a}' for i,(w,a) in enumerate(zip(c['reference_sequence'],q['sequence']),1) if w!=a]
        assert len(q['sequence'])==len(c['reference_sequence'])
        assert set(changes)==set(q['mutation'].split(':'))
        assert len(changes)==q['mutation_order']
        assert all(a in AA for a in q['sequence'])
    return c,e,manifest

def generic_prior(w,a):
    """Conservative uncertainty prior, not an assertion of neutral fitness."""
    cost=0.12
    if (w in HYDRO)!=(a in HYDRO): cost+=0.30
    if w in CHARGE and a in CHARGE:
        if (w in 'DE')!=(a in 'DE'): cost+=0.15
    if a=='P' and w!='P': cost+=0.7
    if a=='G' and w!='G': cost+=0.2
    if w in 'GP' and a not in 'GP': cost+=0.3
    return -cost

def baseline(c,e):
    rows=[];detail=[]
    for q,x in zip(c['query'],e['candidates']):
        assert q['id']==x['id']
        terms=[]
        for t in x['substitution_terms']:
            prior=generic_prior(t['WT'],t['ALT'])
            lr=t['sum_log_frequency_ratio']
            terms.append(prior if lr is None else 0.45*lr+0.3*prior)
        raw=math.fsum(terms)
        rows.append({'id':q['id'],'score':round(math.tanh(raw/5.0),7)})
        detail.append({'id':q['id'],'mutation':q['mutation'],'raw':raw,'msa_coverage':x['scores']['sum_log_frequency_ratio']['coverage']})
    return rows,detail

def validate(rows,c):
    assert len(rows)==250 and len({x['id'] for x in rows})==250
    assert {x['id'] for x in rows}=={q['id'] for q in c['query']}
    assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in rows)

if __name__=='__main__':
    c,e,m=load_and_check()
    rows,detail=baseline(c,e)
    validate(rows,c)
    result={'predictions':rows,'summary':'Initial label-free stability prior: additive, moderated homolog log-frequency ratios where observed; conservative amino-acid compatibility prior for missing sites. Source/reference hashes and mutation identities verified. No measured outcomes, coordinates or validated epistasis; uncertainty is high, especially at MSA-uncovered sites. One global tanh scale; higher is better.'}
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'checkpoint_diagnostics.json').write_text(json.dumps(detail,indent=2)+'\n')
    print('Checkpoint validated and saved:',len(rows),'candidates; score range',min(x['score'] for x in rows),max(x['score'] for x in rows))
