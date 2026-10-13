"""Label-free AMFR stability prioritization; no measured outcome fitting."""
import json, math, hashlib, re
from pathlib import Path
import msa_tools

ROOT=Path(__file__).resolve().parent

def load(name):
    return json.loads((ROOT/name).read_text())

def validate_inputs(case,manifest,evidence):
    wt=case['reference_sequence']
    assert case['context']['higher_is_better'] is True
    assert manifest['measurements_present'] is False and manifest['support_count']==0
    assert not case.get('reference_chains') and not case.get('fixed_partners'), 'Review partner context'
    expected={f'M{i:03d}' for i in range(1,251)}
    assert len(case['query'])==250 and {q['id'] for q in case['query']}==expected
    for q in case['query']:
        actual=[f'{a}{i}{b}' for i,(a,b) in enumerate(zip(wt,q['sequence']),1) if a!=b]
        assert len(q['sequence'])==len(wt)
        assert set(q['mutation'].split(':'))==set(actual)
        assert q['mutation_order']==len(actual)
    assert evidence['reference_sequences_sha256']=={'__single__':hashlib.sha256(wt.encode()).hexdigest()}
    for name,meta in manifest['files'].items():
        if isinstance(meta,dict) and 'sha256' in meta:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256'], name
    recomputed=msa_tools.score_case(case)
    assert recomputed==evidence, 'MSA tool and supplied evidence disagree'


def initial_predictions(case,evidence):
    # Missing terms get an explicit weak mutational prior, never a fake zero MSA term.
    rows=[]
    for q,e in zip(case['query'],evidence['candidates']):
        assert q['id']==e['id']
        terms=e['substitution_terms']
        value=0.0
        for t in terms:
            lr=t['sum_log_frequency_ratio']
            value+=lr/3.0 if lr is not None else -0.25
        rows.append({'id':q['id'],'score':round(math.tanh(value/3.0),6)})
    return {'predictions':rows,'summary':'First-pass additive natural-homolog log-frequency-ratio prior for proteolysis stability; absent MSA terms receive a weak generic mutation prior. One global tanh scale, higher is better. Mutation identities, source hashes, WT mapping and all 250 IDs checked. No assay labels, structure, partners or validated epistasis; rankings are uncertain.'}


def main():
    case,manifest,evidence=load('case.json'),load('evidence_manifest.json'),load('msa_evidence.json')
    validate_inputs(case,manifest,evidence)
    out=initial_predictions(case,evidence)
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Checkpoint saved:',len(out['predictions']),'finite scores; range',min(x['score'] for x in out['predictions']),max(x['score'] for x in out['predictions']))

if __name__=='__main__':main()
