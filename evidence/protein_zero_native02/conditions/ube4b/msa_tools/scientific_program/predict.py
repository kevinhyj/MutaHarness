"""Unsupervised evolutionary prior for UBE4B stability, without assay labels."""
import hashlib,json,math,re
from pathlib import Path
import msa_tools

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def load_validate():
    case=json.load(open(ROOT/'case.json'))
    manifest=json.load(open(ROOT/'evidence_manifest.json'))
    side=json.load(open(ROOT/'msa_evidence.json'))
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    assert case['context']['higher_is_better'] is True
    assert 'reference_chains' not in case, 'This predictor requires a single protein chain'
    assert len(case['query'])==250
    assert {q['id'] for q in case['query']}=={f'M{i:03d}' for i in range(1,251)}
    for path,meta in manifest['files'].items():
        if isinstance(meta,dict) and 'sha256' in meta:
            assert sha(ROOT/path)==meta['sha256'],path
    wt=case['reference_sequence']
    assert side['reference_sequences_sha256']['__single__']==hashlib.sha256(wt.encode()).hexdigest()
    for q in case['query']:
        changes=[f'{w}{p}{a}' for p,(w,a) in enumerate(zip(wt,q['sequence']),1) if w!=a]
        assert len(q['sequence'])==len(wt)
        assert set(changes)==set(q['mutation'].split(':')),(q['id'],changes)
        assert len(changes)==q['mutation_order']
    recomputed=msa_tools.score_case(case)
    assert recomputed==side,'MSA sidecar is not reproducible from the case'
    return case,manifest,side

def run_checkpoint():
    case,manifest,side=load_validate()
    predictions=[];details=[]
    for q,e in zip(case['query'],side['candidates']):
        assert q['id']==e['id']
        # Missing evolutionary terms use an explicit weak unfavorable prior,
        # rather than claiming that the partial sum is a complete MSA score.
        terms=[t['sum_log_frequency_ratio'] if t['sum_log_frequency_ratio'] is not None else -0.35 for t in e['substitution_terms']]
        raw=math.fsum(terms)
        score=math.tanh(raw/8.0)
        assert math.isfinite(score) and -1<=score<=1
        predictions.append({'id':q['id'],'score':score})
        details.append({'id':q['id'],'mutation':q['mutation'],'raw':raw,'complete_evolutionary_score':e['scores']['sum_log_frequency_ratio']['score'],'coverage':e['scores']['sum_log_frequency_ratio']['coverage']})
    summary='Label-free stability ranking from supplied weighted natural-homolog log-frequency ratios, summed across substitutions; absent terms receive a weak unfavorable prior. One global tanh scale; higher is better. Source hashes, WT mapping and all 250 mutation identities verified. No structure, measured outcomes or validated epistasis; evolutionary compatibility is an uncertain stability proxy.'
    result={'predictions':predictions,'summary':summary}
    json.dump(result,open(ROOT/'checkpoint_predictions.json','w'),indent=2,allow_nan=False)
    json.dump(details,open(ROOT/'checkpoint_diagnostics.json','w'),indent=2,allow_nan=False)
    return result

if __name__=='__main__':
    result=run_checkpoint()
    print('Checkpoint written:',len(result['predictions']),'scores; range',min(p['score'] for p in result['predictions']),max(p['score'] for p in result['predictions']))
